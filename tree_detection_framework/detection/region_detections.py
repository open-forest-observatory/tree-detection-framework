import copy
from pathlib import Path
from typing import Callable, List, Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio.transform
import shapely
from shapely.affinity import affine_transform

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.utils.geospatial import to_crs_multiple_geometry_columns
from tree_detection_framework.utils.raster import show_raster


def plot_detections(
    data_frame: gpd.GeoDataFrame,
    bounds: gpd.GeoSeries,
    CRS: Optional[pyproj.CRS] = None,
    plt_ax: Optional[plt.axes] = None,
    plt_show: bool = True,
    show_centroid: bool = False,
    visualization_column: Optional[str] = None,
    bounds_color: Optional[Union[str, np.array, pd.Series]] = None,
    detection_kwargs: dict = {},
    bounds_kwargs: dict = {},
    raster_file: Optional[PATH_TYPE] = None,
    raster_vis_downsample: float = 10.0,
) -> plt.axes:
    """Plot the detections and the bounds of the region

    Args:
        data_frame: (gpd.GeoDataFrame):
            The data representing the detections
        bounds: (gpd.GeoSeries):
            The spatial bounds of the predicted region
        CRS (Optional[pyproj.CRS], optional):
            What CRS to use. Defaults to None.
        plt_ax (Optional[plt.axes], optional):
            A pyplot axes to plot on. If not provided, one will be created. Defaults to None.
        plt_show (bool, optional):
            Whether to plot the result or just return it. Defaults to True.
        show_centroid (bool, optional):
            Whether to plot the centroid points within the boundaries. Defaults to False.
        visualization_column (Optional[str], optional):
            Which column to visualize from the detections dataframe. Defaults to None.
        bounds_color (Optional[Union[str, np.array, pd.Series]], optional):
            The color to plot the bounds. Must be accepted by the gpd.plot color argument.
            Defaults to None.
        detection_kwargs (dict, optional):
            Additional keyword arguments to pass to the .plot method for the detections.
            Defaults to {}.
        bounds_kwargs (dict, optional):
            Additional keyword arguments to pass to the .plot method for the bounds.
            Defaults to {}.
        raster_file (Optional[PATH_TYPE], optional):
            A path to a raster file to visualize the detections over if provided. Defaults to None.
        raster_vis_downsample (float, optional):
            The raster file is downsampled by this fraction before visualization to avoid
            excessive memory use or plotting time. Defaults to 10.0.

    Returns:
        plt.axes: The axes that were plotted on
    """

    # If no axes are provided, create new ones
    if plt_ax is None:
        _, plt_ax = plt.subplots(figsize=(10, 8))

    # Show the raster if provided
    if raster_file is not None:
        show_raster(
            raster_file_path=raster_file,
            downsample_factor=raster_vis_downsample,
            plt_ax=plt_ax,
            CRS=CRS,
        )

    # Plot the detections dataframe and the bounds on the same axes
    if "facecolor" not in detection_kwargs:
        # Plot with transperent faces unless requested
        detection_kwargs["facecolor"] = "none"

    data_frame.plot(
        ax=plt_ax, column=visualization_column, **detection_kwargs, legend=True
    )
    # Use the .boundary attribute to plot just the border. This works since it's a geoseries,
    # not a geodataframe
    bounds.boundary.plot(ax=plt_ax, color=bounds_color, **bounds_kwargs)

    # Plot the centroids if requested
    if show_centroid is True:
        for i, row in data_frame.iterrows():
            centroid = row.geometry.centroid
            x, y = centroid.x, centroid.y
            plt_ax.plot(x, y, "ro", markersize=1)
            plt_ax.text(
                x + 3, y, f"{i:03}", color="red", fontsize=4, ha="left", va="top"
            )

    # Show if requested
    if plt_show:
        plt.show()

    # Return the axes in case they need to be used later
    return plt_ax


class RegionDetections:
    detections: gpd.GeoDataFrame
    prediction_bounds_in_CRS: Union[shapely.Polygon, shapely.MultiPolygon, None]

    def __init__(
        self,
        detection_geometries: List[shapely.Geometry] | str,
        data: Union[dict, pd.DataFrame] = {},
        CRS: Optional[Union[pyproj.CRS, rasterio.CRS]] = None,
        geospatial_prediction_bounds: Optional[
            shapely.Polygon | shapely.MultiPolygon
        ] = None,
        geometry_columns: Optional[List[str]] = None,
    ):
        """Create a region detections object

        Args:
            detection_geometries (List[shapely.Geometry] | str | None):
                A list of shapely geometries for each detection. Alternatively, can be a string
                represting a key in data providing the same, or None if that key is named "geometry".
            data (Optional[dict | pd.DataFrame], optional):
                A dictionary mapping from str names for an attribute to a list of values for that
                attribute, one value per detection. Or a pandas dataframe. Passed to the data
                argument of gpd.GeoDataFrame. Defaults to {}.
            CRS (Optional[pyproj.CRS], optional):
                A coordinate reference system to interpret the data in. TODO, describe what happens
                if this is not set. Defaults to None.
            geometry_columns (Optional[List[str]], optional):
                Which columns to treat as geometric columns that should be transformed to different
                reference frames. If unset, all geometry-typed columns will be transformed. Defaults
                to None.
        """
        # Build a geopandas dataframe containing the geometries, additional attributes, and CRS
        self.detections = gpd.GeoDataFrame(
            data=data, geometry=detection_geometries, crs=CRS
        )

        # If no bounds are provided, compute them from the detections if possible
        if geospatial_prediction_bounds is None and len(self.detections) > 0:
            geospatial_prediction_bounds = self.detections.total_bounds
            geospatial_prediction_bounds = shapely.box(*geospatial_prediction_bounds)

        # Create a one-length geoseries for the bounds
        self.prediction_bounds_in_CRS = gpd.GeoSeries(
            data=[geospatial_prediction_bounds], crs=CRS
        )

        # Record which columns contain geometry attributes
        # TODO: if not specified, set it to all columns that are of type BaseGeometry
        self.geometry_columns = geometry_columns

    def apply_function_to_detections(
        self,
        func: Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame],
        inplace: bool = False,
    ) -> Optional["RegionDetections"]:
        """_summary_

        Args:
            func (Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame]):
                A function which takes in a GeoDataFrame and returns another GeoDataFrame.
            inplace (bool):
                Should the RegionDetections object be modified inplace, or a modified copy returned.
                Defaults to False.

        Returns:
            Optional[RegionDetections]:
                If inplace=False, returns a modified copy. Otherwise, returns None.
        """
        if inplace:
            # If inplace, just modify the detections directly
            self.detections = func(self.detections)
        else:
            # If not inplace, first deepcopy self to avoid any chance of modifying the original
            # .detections object
            modified_rd = copy.deepcopy(self)
            # Then apply the function
            modified_rd.detections = func(modified_rd.detections)

            return modified_rd

    def subset_detections(self, detection_indices) -> "RegionDetections":
        """Return a new RegionDetections object with only the detections indicated by the indices

        Args:
            detection_indices:
                Which detections to include. Can be any type that can be passed to pd.iloc.

        Returns:
            RegionDetections: The subset of detections cooresponding to these indeices
        """
        # Create a deep copy of the object
        subset_rd = copy.deepcopy(self)
        # Subset the detections dataframe to the requested rows
        subset_rd.detections = subset_rd.detections.iloc[detection_indices, :]

        return subset_rd

    def save(self, save_path: PATH_TYPE):
        """Saves the information to disk

        Args:
            save_path (PATH_TYPE):
                Path to a geofile to save the data to. The containing folder will be created if it
                doesn't exist.
        """
        # Convert to a Path object and create the containing folder if not present
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the detections to a file. Note that the bounds of the prediction region are currently lost.
        self.detections.to_file(save_path)

    def get_data_frame(self, CRS: Optional[pyproj.CRS] = None) -> gpd.GeoDataFrame:
        """Get the detections, optionally specifying a CRS

        Args:
            CRS (Optional[pyproj.CRS], optional):
                Requested CRS for the output detections. If un-set, the CRS of self.detections will
                be used. Defaults to None.

        Returns:
            gpd.GeoDataFrame: Detections in the requested CRS
        """
        # If no CRS is specified, return the data as-is, using the current CRS
        if CRS is None:
            return self.detections.copy()

        # Transform the data to the requested CRS. Note that if no CRS is provided initially,
        # this will error out. This is applied to all geometry-typed columns unless explicitly
        # stated by self.geometry_columns
        detections_in_new_CRS = to_crs_multiple_geometry_columns(
            self.detections,
            crs=CRS,
            columns_to_transform=self.geometry_columns,
            inplace=False,
        )
        return detections_in_new_CRS

    def convert_to_bboxes(self) -> "RegionDetections":
        """
        Return a copy of the RD with the geometry of all detections replaced by the minimum
        axis-aligned bounding rectangle. Also, rows with an empty geometry are removed.

        Returns:
            RegionDetections: An identical RD with all non-empty geometries represented as bounding boxes.
        """
        # Get the detections
        detections_df = self.get_data_frame()
        # Get non-empty rows in the dataframe, since conversion to bounding box only works for
        # non-empty polygons
        nonempty_rows = detections_df[~detections_df.geometry.is_empty]
        # Get the bounds and convert to shapely boxes
        bounds = nonempty_rows.bounds
        boxes = shapely.box(
            xmin=bounds.minx, ymin=bounds.miny, xmax=bounds.maxx, ymax=bounds.maxy
        )
        # Update the geometry
        # TODO make sure that thisn't updating the geometry of the orignal one
        nonempty_rows.geometry = boxes
        # Create a new RegionDetections object and update the detections on it
        bbox_rd = copy.deepcopy(self)
        bbox_rd.detections = nonempty_rows
        return bbox_rd

    def update_geometry_column(self, geometry_column: str) -> "RegionDetections":
        """Update the geometry to another column in the dataframe that contains shapely data

        Args:
            geometry_column (str): The name of a column containing shapely data

        Returns:
            RegionDetections: An updated RD with the geometry specified by the data in `geometry_column`
        """
        # Create a copy of the detections
        detections_df = self.get_data_frame().copy()
        # Set the geometry column to the specified one
        detections_df.geometry = detections_df[geometry_column]

        # Create a copy of the RD
        updated_geometry_rd = copy.deepcopy(self)
        # Update the detections and return
        updated_geometry_rd.detections = detections_df

        return updated_geometry_rd

    def get_bounds(self, CRS: Optional[pyproj.CRS] = None) -> gpd.GeoSeries:
        if CRS is None:
            # Get bounds in original CRS
            bounds = self.prediction_bounds_in_CRS.copy()
        else:
            # Get bounds in requested CRS
            bounds = self.prediction_bounds_in_CRS.to_crs(CRS)

        return bounds

    def get_CRS(self) -> Union[pyproj.CRS, None]:
        """Return the CRS of the detections dataframe

        Returns:
            Union[pyproj.CRS, None]: The CRS for the detections
        """
        return self.detections.crs

    def plot(
        self,
        CRS: Optional[pyproj.CRS] = None,
        plt_ax: Optional[plt.axes] = None,
        plt_show: bool = True,
        show_centroid: bool = False,
        visualization_column: Optional[str] = None,
        bounds_color: Optional[Union[str, np.array, pd.Series]] = None,
        detection_kwargs: dict = {},
        bounds_kwargs: dict = {},
        raster_file: Optional[PATH_TYPE] = None,
        raster_vis_downsample: float = 10.0,
    ) -> plt.axes:
        """Plot the detections and the bounds of the region

        Args:
            CRS (Optional[pyproj.CRS], optional):
                What CRS to use. Defaults to None.
            plt_ax (Optional[plt.axes], optional):
                A pyplot axes to plot on. If not provided, one will be created. Defaults to None.
            plt_show (bool, optional):
                Whether to plot the result or just return it. Defaults to True.
            show_centroid (bool, optional):
                Whether to plot the centroid points within the boundaries. Defaults to False.
            visualization_column (Optional[str], optional):
                Which column to visualize from the detections dataframe. Defaults to None.
            bounds_color (Optional[Union[str, np.array, pd.Series]], optional):
                The color to plot the bounds. Must be accepted by the gpd.plot color argument.
                Defaults to None.
            detection_kwargs (dict, optional):
                Additional keyword arguments to pass to the .plot method for the detections.
                Defaults to {}.
            bounds_kwargs (dict, optional):
                Additional keyword arguments to pass to the .plot method for the bounds.
                Defaults to {}.
            raster_file (Optional[PATH_TYPE], optional):
                A path to a raster file to visualize the detections over if provided. Defaults to None.
            raster_vis_downsample (float, optional):
                The raster file is downsampled by this fraction before visualization to avoid
                excessive memory use or plotting time. Defaults to 10.0.

        Returns:
            plt.axes: The axes that were plotted on
        """

        # Get the dataframe and the bounds
        data_frame = self.get_data_frame(CRS=CRS)
        bounds = self.get_bounds(CRS)

        # Perform plotting and return the axes
        return plot_detections(
            data_frame=data_frame,
            bounds=bounds,
            CRS=data_frame.crs,
            plt_ax=plt_ax,
            plt_show=plt_show,
            show_centroid=show_centroid,
            visualization_column=visualization_column,
            detection_kwargs=detection_kwargs,
            bounds_kwargs=bounds_kwargs,
            raster_file=raster_file,
            raster_vis_downsample=raster_vis_downsample,
            bounds_color=bounds_color,
        )


class RegionDetectionsSet:
    region_detections: List[RegionDetections]

    def __init__(self, region_detections: List[RegionDetections]):
        """Create a set of detections to conveniently perform operations on all of them, for example
        merging all regions into a single dataframe with an additional column indicating which
        region each detection belongs to.

        Args:
            region_detections (List[RegionDetections]): A list of individual detections
        """
        self.region_detections = region_detections

    def apply_function_to_detections(
        self,
        func: Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame],
        inplace: bool = False,
    ) -> Optional["RegionDetectionsSet"]:
        """
        See documentation for RegionDetections.apply_function_to_detections
        """
        # Convert each detection
        modified_region_detections = [
            rd.apply_function_to_detections(func=func, inplace=inplace)
            for rd in self.region_detections
        ]

        # If inplace, the individual RegionDetections objects will already have been updated and
        # None should be returned for consistency
        if inplace:
            return None

        # Create and return the new RDS
        modified_rds = RegionDetectionsSet(modified_region_detections)
        return modified_rds

    def all_regions_have_CRS(self) -> bool:
        """Check whether all sub-regions have a non-None CRS

        Returns:
            bool: Whether all sub-regions have a valid CRS.
        """
        # Get the CRS for each sub-region
        regions_CRS_values = [rd.detections.crs for rd in self.region_detections]
        # Only valid if no CRS value is None
        valid = not (None in regions_CRS_values)

        return valid

    def get_default_CRS(self, check_all_have_CRS=True) -> pyproj.CRS:
        """Find the CRS of the first sub-region to use as a default

        Args:
            check_all_have_CRS (bool, optional):
                Should an error be raised if not all regions have a CRS set.

        Returns:
            pyproj.CRS: The CRS given by the first sub-region.
        """
        if check_all_have_CRS and not self.all_regions_have_CRS():
            raise ValueError(
                "Not all regions have a CRS set and a default one was requested"
            )
        # Check that every region is geospatial
        regions_CRS_values = [rd.detections.crs for rd in self.region_detections]
        # The default is the to the CRS of the first region
        # TODO in the future it could be something else like the most common
        CRS = regions_CRS_values[0]

        return CRS

    def get_region_detections(self, index: int) -> RegionDetections:
        """Get a single region detections object by index

        Args:
            index (int): Which one to select

        Returns:
            RegionDetections: The RegionDetections object from the list of objects in the set
        """
        return self.region_detections[index]

    def merge(
        self,
        region_ID_key: Optional[str] = "region_ID",
        CRS: Optional[pyproj.CRS] = None,
    ):
        """Get the merged detections across all regions with an additional field specifying which region
        the detection came from.

        Args:
            region_ID_key (Optional[str], optional):
                Create this column in the output dataframe identifying which region that data came
                from using a zero-indexed integer. Defaults to "region_ID".
            CRS (Optional[pyproj.CRS], optional):
                Requested CRS for merged detections. If un-set, the CRS of the first region will
                be used. Defaults to None.

        Returns:
            gpd.GeoDataFrame: Detections in the requested CRS
        """

        # Get the detections from each region detection object as geodataframes
        detection_geodataframes = [
            rd.get_data_frame(CRS=CRS) for rd in self.region_detections
        ]

        # Add a column to each geodataframe identifying which region detection object it came from
        # Note that dataframes in the original list are updated
        # TODO consider a more sophisticated ID
        for ID, gdf in enumerate(detection_geodataframes):
            gdf[region_ID_key] = ID

        # Concatenate the geodataframes together
        concatenated_geodataframes = pd.concat(
            detection_geodataframes, ignore_index=True
        )

        # Add a globally unique 5-digit ID column
        concatenated_geodataframes["unique_ID"] = concatenated_geodataframes.index.map(
            lambda i: f"{i:05d}"
        )

        ## Merge_bounds
        # Get the merged bounds
        merged_bounds = self.get_bounds(CRS=CRS)
        # Convert to a single shapely object
        merged_bounds_shapely = merged_bounds.geometry[0]

        # Use the geometry column of the concatenated dataframes
        merged_region_detections = RegionDetections(
            detection_geometries="geometry",
            data=concatenated_geodataframes,
            CRS=CRS,
            geospatial_prediction_bounds=merged_bounds_shapely,
        )

        return merged_region_detections

    def get_data_frame(
        self,
        CRS: Optional[pyproj.CRS] = None,
        merge: bool = False,
        region_ID_key: str = "region_ID",
    ) -> gpd.GeoDataFrame | List[gpd.GeoDataFrame]:
        """Get the detections, optionally specifying a CRS

        Args:
            CRS (Optional[pyproj.CRS], optional):
                Requested CRS for the output detections. If un-set, the CRS of self.detections will
                be used. Defaults to None.
            merge (bool, optional):
                If true, return one dataframe. Else, return a list of individual dataframes.
            region_ID_key (str, optional):
                Use this column to identify which region each detection came from. Defaults to
                "region_ID"

        Returns:
            gpd.GeoDataFrame | List[gpd.GeoDataFrame]:
                If merge=True, then one dataframe with an addtional column specifying which region each
                detection came from. If merge=False, then a list of dataframes for each region.
        """
        if merge:
            # Merge all of the detections into one RegionDetection
            merged_detections = self.merge(region_ID_key=region_ID_key, CRS=CRS)
            # get the dataframe. It is already in the requested CRS in the current implementation.
            data_frame = merged_detections.get_data_frame()
            return data_frame

        # Get a list of dataframes from each region
        list_of_region_data_frames = [
            rd.get_data_frame(CRS=CRS) for rd in self.region_detections
        ]
        return list_of_region_data_frames

    def convert_to_bboxes(self) -> "RegionDetectionsSet":
        """Convert all the RegionDetections to bounding box representations

        Returns:
            RegionDetectionsSet:
                A new RDS where each RD has all empty geometries dropped and all remaning ones
                represented by an axis-aligned rectangle.
        """
        # Convert each detection
        converted_detections = [rd.convert_to_bboxes() for rd in self.region_detections]
        # Return the new RDS
        bboxes_rds = RegionDetectionsSet(converted_detections)
        return bboxes_rds

    def update_geometry_column(self, geometry_column: str) -> "RegionDetectionsSet":
        """
        Update the geometry to another column in the dataframe that contains shapely data for each RD

        Args:
            geometry_column (str): The name of a column containing shapely data

        Returns:
            RegionDetectionsSet: An updated RDS with the geometry specified by the data in `geometry_column`
        """
        # Convert each detection
        converted_detections = [
            rd.update_geometry_column(geometry_column=geometry_column)
            for rd in self.region_detections
        ]
        # Return the new RDS
        updated_geometry_rds = RegionDetectionsSet(converted_detections)
        return updated_geometry_rds

    def get_bounds(
        self, CRS: Optional[pyproj.CRS] = None, union_bounds: bool = True
    ) -> gpd.GeoSeries:
        """Get the bounds corresponding to the sub-regions.

        Args:
            CRS (Optional[pyproj.CRS], optional):
                The CRS to return the bounds in. If not set, it will be the bounds of the first
                region. Defaults to None.
            union_bounds (bool, optional):
                Whether to return the spatial union of all bounds or a series of per-region bounds.
                Defaults to True.

        Returns:
            gpd.GeoSeries: Either a one-length series of merged bounds if merge=True or a series
            of bounds per region.
        """

        region_bounds = [rd.get_bounds(CRS=CRS) for rd in self.region_detections]
        # Create a geodataframe out of these region bounds
        all_region_bounds = gpd.GeoSeries(pd.concat(region_bounds), crs=CRS)

        # If the union is not requested, return the individual bounds
        if not union_bounds:
            return all_region_bounds

        # Compute the union of all bounds
        merged_bounds = gpd.GeoSeries([all_region_bounds.geometry.union_all()], crs=CRS)

        return merged_bounds

    def disjoint_bounds(self) -> bool:
        """Determine whether the bounds of the sub-regions are disjoint

        Returns:
            bool: Are they disjoint
        """
        # Get the bounds for each individual region
        bounds = self.get_bounds(union_bounds=False)
        # Get the union of all bounds
        union_bounds = bounds.union_all()

        # Find the sum of areas for each region
        sum_individual_areas = bounds.area.sum()
        # And the area of the union
        union_area = union_bounds.area

        # If the two areas are the same (down to numeric errors) then there are no overlaps
        disjoint = np.allclose(sum_individual_areas, union_area)

        return disjoint

    def save(
        self,
        save_path: PATH_TYPE,
        CRS: Optional[pyproj.CRS] = None,
        region_ID_key: Optional[str] = "region_ID",
    ):
        """
        Save the data to a geospatial file by calling get_data_frame with merge=True and then saving
        to the specified file. The containing folder is created if it doesn't exist.

        Args:
            save_path (PATH_TYPE):
               File to save the data to. The containing folder will be created if it does not exist.
            CRS (Optional[pyproj.CRS], optional):
                See get_data_frame.
            region_ID_key (Optional[str], optional):
                See get_data_frame.
        """
        # Get the concatenated dataframes
        concatenated_geodataframes = self.get_data_frame(
            CRS=CRS, region_ID_key=region_ID_key, merge=True
        )

        # Ensure that the folder to save them to exists
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)

        # Save the data to the geofile
        concatenated_geodataframes.to_file(save_path)

    def plot(
        self,
        CRS: Optional[pyproj.CRS] = None,
        plt_ax: Optional[plt.axes] = None,
        plt_show: bool = True,
        visualization_column: Optional[str] = None,
        bounds_color: Optional[Union[str, np.array, pd.Series]] = None,
        detection_kwargs: dict = {},
        bounds_kwargs: dict = {},
        raster_file: Optional[PATH_TYPE] = None,
        raster_vis_downsample: float = 10.0,
    ) -> plt.axes:
        """Plot each of the region detections using their .plot method

        Args:
            CRS (Optional[pyproj.CRS], optional):
                The CRS to use for plotting all regions. If unset, the default one for this object
                will be selected. Defaults to None.
            plt_ax (Optional[plt.axes], optional):
                The axes to plot on. Will be created if not provided. Defaults to None.
            plt_show (bool, optional):
                See RegionDetections.plot. Defaults to True.
            visualization_column (Optional[str], optional):
                See regiondetections.plot. Defaults to None.
            bounds_color (Optional[Union[str, np.array, pd.Series]], optional):
                See regiondetections.plot. Defaults to None.
            detection_kwargs (dict, optional):
                See regiondetections.plot. Defaults to {}.
            bounds_kwargs (dict, optional):
                See regiondetections.plot. Defaults to {}.
            raster_file (Optional[PATH_TYPE], optional):
                See regiondetections.plot. Defaults to None.
            raster_vis_downsample (float, optional):
                See regiondetections.plot. Defaults to 10.0.

        Returns:
            plt.axes: The axes that have been plotted on.
        """
        # Extract the bounds for each of the sub-regions
        bounds = self.get_bounds(CRS=CRS, union_bounds=False)
        data_frame = self.get_data_frame(CRS=CRS, merge=True)

        # Perform plotting and return the axes
        return plot_detections(
            data_frame=data_frame,
            bounds=bounds,
            CRS=data_frame.crs,
            plt_ax=plt_ax,
            plt_show=plt_show,
            visualization_column=visualization_column,
            bounds_color=bounds_color,
            detection_kwargs=detection_kwargs,
            bounds_kwargs=bounds_kwargs,
            raster_file=raster_file,
            raster_vis_downsample=raster_vis_downsample,
        )


def reproject_detections(
    region_detections: RegionDetections,
    target_crs: pyproj.CRS,
) -> RegionDetections:
    """
    Reprojects the RegionDetections object to a target CRS.

    Args:
        region_detections (RegionDetections): The detections to reproject.
        target_crs (pyproj.CRS): The target CRS to reproject to.

    Returns:
        RegionDetections: Detections after reprojection.
    """

    # Get the detections in the target CRS
    projected_detections = region_detections.get_data_frame(CRS=target_crs)
    # Get the bounds in the target CRS
    projected_bounds = region_detections.get_bounds(CRS=target_crs)[0]

    # Create a new RegionDetections object with the projected detections
    projected_region_detections = RegionDetections(
        detection_geometries=None,
        data=projected_detections,
        CRS=target_crs,
        geospatial_prediction_bounds=projected_bounds,
    )

    return projected_region_detections
