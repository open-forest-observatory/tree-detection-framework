import copy
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio.transform
import shapely
from shapely.affinity import affine_transform

from tree_detection_framework.constants import PATH_TYPE


class RegionDetections:
    detections: gpd.GeoDataFrame
    pixel_to_CRS_transform: rasterio.transform.AffineTransformer
    prediction_bounds_in_CRS: Union[shapely.Polygon, shapely.MultiPolygon, None]

    def __init__(
        self,
        detection_geometries: List[shapely.Geometry] | str,
        data: Union[dict, pd.DataFrame] = {},
        input_in_pixels: bool = True,
        CRS: Optional[Union[pyproj.CRS, rasterio.CRS]] = None,
        pixel_to_CRS_transform: Optional[rasterio.transform.AffineTransformer] = None,
        pixel_prediction_bounds: Optional[
            shapely.Polygon | shapely.MultiPolygon
        ] = None,
        geospatial_prediction_bounds: Optional[
            shapely.Polygon | shapely.MultiPolygon
        ] = None,
    ):
        """Create a region detections object

        Args:
            detection_geometries (List[shapely.Geometry] | str | None):
                A list of shapely geometries for each detection. Alternatively, can be a string
                represting a key in data providing the same, or None if that key is named "geometry".
                The coordinates can either be provided in pixel coordinates or in the coordinates
                of a CRS. input_in_pixels should be set accordingly.
            data (Optional[dict | pd.DataFrame], optional):
                A dictionary mapping from str names for an attribute to a list of values for that
                attribute, one value per detection. Or a pandas dataframe. Passed to the data
                argument of gpd.GeoDataFrame. Defaults to {}.
            input_in_pixels (bool, optional):
                Whether the detection_geometries should be interpreted in pixels or geospatial
                coordinates.
            CRS (Optional[pyproj.CRS], optional):
                A coordinate reference system to interpret the data in. If input_in_pixels is False,
                then the input data will be interpreted as values in this CRS. If input_in_pixels is
                True and CRS is None, then the data will be interpreted as pixel coordinates with no
                georeferencing information. If input_in_pixels is True and CRS is not None, then
                the data will be attempted to be geometrically transformed into the CRS using either
                pixel_to_CRS_transform if set, or the relationship between the pixel and geospatial
                bounds of the region. Defaults to None.
            pixel_to_CRS_transform (Optional[rasterio.transform.AffineTransformer], optional):
                An affine transformation mapping from the pixel coordinates to those of the CRS.
                Only meaningful if `CRS` is set as well. Defaults to None.
            pixel_prediction_bounds (Optional[shapely.Polygon | shapely.MultiPolygon], optional):
                The pixel bounds of the region that predictions were generated for. For example, a
                square starting at (0, 0) and extending to the size in pixels of the tile.
                Defaults to None.
            geospatial_prediction_bounds (Optional[shapely.Polygon | shapely.MultiPolygon], optional):
                Only meaningful if CRS is set. In that case, it represents the spatial bounds of the
                prediction region. If pixel_to_CRS_transform is None, and both pixel_- and
                geospatial_prediction_bounds are not None, then the two bounds will be used to
                compute the transform. Defaults to None.
        """
        # Build a geopandas dataframe containing the geometries, additional attributes, and CRS
        self.detections = gpd.GeoDataFrame(
            data=data, geometry=detection_geometries, crs=CRS
        )

        # If the pixel_to_CRS_transform is None but can be computed from the two bounds, do that
        if (
            input_in_pixels
            and (pixel_to_CRS_transform is None)
            and (pixel_prediction_bounds is not None)
            and (geospatial_prediction_bounds is not None)
        ):
            # We assume that these two array are exactly corresponding, representing the same shape
            # in the two coordinate frames and also the same starting vertex.
            # Drop the last entry because it is a duplicate of the first one
            geospatial_corners_array = shapely.get_coordinates(
                geospatial_prediction_bounds
            )[:-1]
            pixel_corners_array = shapely.get_coordinates(pixel_prediction_bounds)[:-1]

            # If they don't have the same number of vertices, this can't be the case
            if len(geospatial_corners_array) != len(pixel_corners_array):
                raise ValueError("Bounds had different lengths")

            # Representing the correspondences as ground control points
            ground_control_points = [
                rasterio.control.GroundControlPoint(
                    col=pixel_vertex[0],
                    row=pixel_vertex[1],
                    x=geospatial_vertex[0],
                    y=geospatial_vertex[1],
                )
                for pixel_vertex, geospatial_vertex in zip(
                    pixel_corners_array, geospatial_corners_array
                )
            ]
            # Solve the affine transform that best transforms from the pixel to geospatial coordinates
            pixel_to_CRS_transform = rasterio.transform.from_gcps(ground_control_points)

        # Error checking
        if (pixel_to_CRS_transform is not None) and (CRS is None):
            raise ValueError(
                "The geometric transform to map to a CRS was provided but the CRS was not specified"
            )

        if (pixel_to_CRS_transform is None) and (CRS is not None) and input_in_pixels:
            raise ValueError(
                "The input was in pixels and a CRS was specified but no geommetric transformation was provided to transform the pixel values to that CRS"
            )

        # Set the transform
        self.pixel_to_CRS_transform = pixel_to_CRS_transform

        # If the inputs are provided in pixels, apply the transform to the predictions
        if input_in_pixels:
            # Get the transform in the format expected by shapely
            shapely_transform = pixel_to_CRS_transform.to_shapely()
            # Apply this transformation to the geometry of the dataframe
            self.detections.geometry = self.detections.geometry.affine_transform(
                matrix=shapely_transform
            )

        # Handle the bounds
        # If the bounds are provided as geospatial coordinates, use those directly
        if geospatial_prediction_bounds is not None:
            prediction_bounds_in_CRS = geospatial_prediction_bounds
        # If the bounds are provided in pixels and a transform to geospatial is provided, use that
        # this assumes that a CRS is set based on previous checks
        elif (pixel_to_CRS_transform is not None) and (
            pixel_prediction_bounds is not None
        ):
            prediction_bounds_in_CRS = affine_transform(
                geom=pixel_prediction_bounds, matrix=pixel_to_CRS_transform.to_shapely()
            )
        # If there is no CRS and pixel bounds are provided, use these directly
        # The None CRS implies pixels, so this still has the intended meaning
        elif CRS is None and pixel_prediction_bounds is not None:
            prediction_bounds_in_CRS = pixel_prediction_bounds
        # Else set the bounds to None (unknown)
        else:
            prediction_bounds_in_CRS = None

        # Create a one-length geoseries for the bounds
        self.prediction_bounds_in_CRS = gpd.GeoSeries(
            data=[prediction_bounds_in_CRS], crs=CRS
        )

    def subset_detections(self, detection_indices) -> "RegionDetections":
        """Return a new Reg

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

        # Save the detections to a file. Note that the bounds of the prediction region and
        # information about the transform to pixel coordinates are currently lost.
        self.detections.to_file(save_path)

    def get_data_frame(
        self, CRS: Optional[pyproj.CRS] = None, as_pixels: Optional[bool] = False
    ) -> gpd.GeoDataFrame:
        """Get the detections, optionally specifying a CRS or pixel coordinates

        Args:
            CRS (Optional[pyproj.CRS], optional):
                Requested CRS for the output detections. If un-set, the CRS of self.detections will
                be used. Defaults to None.
            as_pixels (Optional[bool], optional):
                Whether to return the values in pixel coordinates. Defaults to False.

        Returns:
            gpd.GeoDataFrame: Detections in the requested CRS or in pixel coordinates with a None .crs
        """
        # If the data is requested in pixel coordinates, transform it appropriately
        if as_pixels:
            if (self.detections.crs is not None) and (
                self.pixel_to_CRS_transform is None
            ):
                raise ValueError(
                    "Pixel coordinates were requested but data is in geospatial units with no transformation to pixels"
                )

            # Compute the inverse transform using ~ to map from the CRS to pixels instead of the
            # other way around. Also get this transform in the shapely convention.
            CRS_to_pixel_transform_shapely = (~self.pixel_to_CRS_transform).to_shapely()
            # Create a new geodataframe with the transformed coordinates
            # Start by copying the old dataframe
            pixel_coordinate_detections = self.detections.copy()
            # Since the units are pixels, it no longer has a CRS, so set it to None
            pixel_coordinate_detections.crs = None
            # Transform the geometry to pixel coordinates
            pixel_coordinate_detections.geometry = (
                pixel_coordinate_detections.geometry.affine_transform(
                    CRS_to_pixel_transform_shapely
                )
            )
            return pixel_coordinate_detections

        # Return the data in geospatial coordinates
        else:
            # If no CRS is specified, return the data as-is, using the current CRS
            if CRS is None:
                return self.detections.copy()

            # Transform the data to the requested CRS. Note that if no CRS is provided initially,
            # this will error out
            detections_in_new_CRS = self.detections.copy().to_crs(CRS)
            return detections_in_new_CRS

    def get_bounds(
        self, CRS: Optional[pyproj.CRS] = None, as_pixels: Optional[bool] = False
    ) -> gpd.GeoSeries:
        if CRS is None:
            # Get bounds in original CRS
            bounds = self.prediction_bounds_in_CRS.copy()
        else:
            # Get bounds in requested CRS
            bounds = self.prediction_bounds_in_CRS.to_crs(CRS)

        if as_pixels:
            # Invert the transform
            CRS_to_pixel_transform_shapely = (~self.pixel_to_CRS_transform).to_shapely()
            # Transform the bounds into pixel coordinates
            bounds.geometry = bounds.geometry.affine_transform(
                CRS_to_pixel_transform_shapely
            )
            # Set the CRS to None since this is now pixels
            # Note that this does not reproject
            bounds.set_crs(None)

        return bounds


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
            gpd.GeoDataFrame: Detections in the requested CRS or in pixel coordinates with a None .crs
        """
        # Check that every region is geospatial
        regions_CRS_values = [rd.detections.crs for rd in self.region_detections]
        if None in regions_CRS_values:
            raise ValueError("Merging requires every region to have a CRS")

        # If no CRS is provided default to the CRS of the first region
        if CRS is None:
            CRS = regions_CRS_values[0]

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
        concatenated_geodataframes = pd.concat(detection_geodataframes)

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
            input_in_pixels=False,
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

    def get_bounds(self, CRS: Optional[pyproj.CRS] = None) -> gpd.GeoSeries:
        region_bounds = [rd.get_bounds(CRS=CRS) for rd in self.region_detections]
        # Create a geodataframe out of these region bounds
        all_region_bounds = gpd.GeoSeries(pd.concat(region_bounds), crs=CRS)
        # Compute the union of all bounds
        merged_bounds = gpd.GeoSeries([all_region_bounds.geometry.union_all()], crs=CRS)

        return merged_bounds

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
