from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from shapely.affinity import affine_transform

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.utils.geometric import get_shapely_transform_from_matrix


class RegionDetections:
    detections: gpd.GeoDataFrame
    pixel_to_CRS_transform: np.ndarray
    prediction_bounds: Union[shapely.Polygon, shapely.MultiPolygon, None]

    def __init__(
        self,
        detection_geometries: List[shapely.Geometry],
        attributes: dict = {},
        pixel_to_CRS_transform: Optional[np.array] = None,
        CRS: Optional[pyproj.CRS] = None,
        prediction_bounds: Optional[shapely.Polygon | shapely.MultiPolygon] = None,
    ):
        """Create a region detections object

        Args:
            detection_geometries (List[shapely.Geometry]):
                A list of shapely geometries for each detection. The coordinates can either be
                provided in pixel coordinates or in the coordinates of a CRS.
            attributes (Optional[dict], optional):
                A dictionary mapping from str names for an attribute to a list of values for that
                attribute. Defaults to {}.
            CRS (Optional[pyproj.CRS], optional):
                A coordinate reference system to interpret the data in. If pixel_to_CRS_transform is
                set, this matrix will be first applied to the values before they are interpreted in
                this CRS. Otherwise, the values will be used directly. If None, the coordinates will
                be assumed to be in pixel coordinate with no georeferencing.
            pixel_to_CRS_transform (Optional[np.array], optional):
                Only meaningful if `CRS` is set as well. A 2x3 transform matrix mapping the
                coordinates in pixels to the coordinates of the CRS. If un-set, the input data will
                be assumed to already be in the coordinates of the given CRS. Defaults to None.
            prediction_bounds (Optional[shapely.Polygon | shapely.MultiPolygon], optional):
                The spatial bounds of the region that predictions were generated for. For example,
                the bounds of a tile. Defaults to None.
        """

        # Check if a pixel to CRS transform is provided
        if pixel_to_CRS_transform:
            # Convert from the matrix representation to what is expected by shapely
            shapely_transform = get_shapely_transform_from_matrix(
                pixel_to_CRS_transform
            )
            # Apply this transformation to each geometry to get the detections in a given CRS
            detection_geometries = [
                affine_transform(geom=geom, matrix=shapely_transform)
                for geom in detection_geometries
            ]
            # Apply the same transform to the detection bounds
            prediction_bounds = affine_transform(
                geom=prediction_bounds, matrix=shapely_transform
            )

        # Set the transform and bounds
        self.pixel_to_CRS_transform = pixel_to_CRS_transform
        self.prediction_bounds = prediction_bounds

        # Build a geopandas dataframe containing the geometries, additional attributes, and CRS
        self.detections = gpd.GeoDataFrame(
            data=attributes, geometry=detection_geometries, crs=CRS
        )

    def save(self, save_path: PATH_TYPE):
        """Saves the information to disk

        Args:
            save_path (PATH_TYPE):
                Path to a geofile to save the data to. The containing folder will be created if it
                doesn't exist.
        """
        # Convert to a Path object and create the containing folder if not present
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save the detections to a file. Note that the bounds and the CRS are currently lost.
        self.detections.to_file(save_path)

    def get_detections(
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

            # Add a row of 0, 0, 1 to the transform
            transform_3x3 = np.concatenate(
                self.pixel_to_CRS_transform, np.expand_dims([0, 0, 1], 0)
            )
            # Compute the matrix inverse of the transform to get the map from the CRS reference
            # frame to pixels rather than the other way around
            CRS_to_pixel_transform_matrix = np.linalg.inv(transform_3x3)
            # Geopandas also uses the shapely conventions, so convert the matrix into this form
            CRS_to_pixel_transform_shapely = get_shapely_transform_from_matrix(
                CRS_to_pixel_transform_matrix
            )
            # Create a new geodataframe with the transformed coordinates
            pixel_coordinate_detections = self.detections.affine_transform(
                CRS_to_pixel_transform_shapely
            )
            # This no longer has a CRS, so set it to None
            pixel_coordinate_detections.crs = None
            return pixel_coordinate_detections.copy()

        # Return the data in geospatial coordinates
        else:
            # If no CRS is specified, return the data as-is, using the current CRS
            if CRS is None:
                return self.detections.copy()

            # Transform the data to the requested CRS. Note that if no CRS is provided initially,
            # this will error out
            detections_in_new_CRS = self.detections.to_crs(CRS)
            return detections_in_new_CRS.copy()


class RegionDetectionsSet:
    region_detections: List[RegionDetections]

    def __init__(self, region_detections: List[RegionDetections]):
        """Create a set of detections to conveniently perform operations on all of them

        Args:
            region_detections (List[RegionDetections]): A list of individual detections
        """
        self.region_detections = region_detections

    def save(self, save_path: PATH_TYPE, region_ID_key: Optional[str] = "region_ID"):
        """
        Save the data to a geospatial file, by adding an additional attribute to specify the region
        and then merging all the regions together.

        Args:
            save_path (PATH_TYPE):
               File to save the data to. The containing folder will be created if it does not exist.
            region_ID_key (Optional[str], optional):
                Create this column in the output dataframe identifying which region that data came
                from using a zero-indexed integer. Defaults to "region_ID".
        """
        # Get the detections from each region detection object as geodataframes
        detection_geodataframes = [rd.get_detections() for rd in self.region_detections]

        # Add a column to each geodataframe identifying which region detection object it came from
        # Note that dataframes in the original list are updated
        for i, gdf in enumerate(detection_geodataframes):
            gdf[region_ID_key] = i

        # Concatenate the geodataframes together
        concatenated_geodataframes = pd.concatenate(detection_geodataframes)

        # Ensure that the folder to save them to exists
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)

        # Save the data to the geofile
        concatenated_geodataframes.to_file(save_path)
