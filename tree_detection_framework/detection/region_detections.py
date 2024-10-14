from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import pyproj
import shapely
from shapely.affinity import affine_transform

from tree_detection_framework.constants import PATH_TYPE


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
            # Format the transform in the format expected by shapely: [a, b, d, e, xoff, y_off]
            shapely_transform = [
                pixel_to_CRS_transform[0, 0],
                pixel_to_CRS_transform[0, 1],
                pixel_to_CRS_transform[1, 0],
                pixel_to_CRS_transform[1, 1],
                pixel_to_CRS_transform[0, 2],
                pixel_to_CRS_transform[1, 2],
            ]
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
        raise NotImplementedError()

    def convert_pixels_to_CRS(self) -> gpd.GeoDataFrame:
        """
        Use the `self.CRS_to_pixels_transform` to convert the predictions from pixels to coordinates
        of self.CRS


        Returns:
            gpd.GeoDataFrame: Dataframe of detections with the CRS=self.CRS.
        """
        raise NotImplementedError()

    def get_detections(
        self, CRS: Optional[pyproj.CRS] = None, as_pixels: Optional[bool] = False
    ) -> gpd.GeoDataFrame:
        """Get the detections, optionally in a specfied CRS

        Args:
            CRS (Optional[pyproj.CRS], optional):
                Requested CRS for the output detections. If un-set, `self.CRS` will be used.
                Defaults to None.
            as_pixels (Optional[bool], optional):
                Whether to return the values in pixel coordinates. Defaults to False.

        Returns:
            gpd.GeoDataFrame: Detections in the requested CRS
        """
        raise NotImplementedError()


class RegionDetectionsSet:
    def __init__(self, region_detections: List[RegionDetections]):
        """Create a set of detections to conveniently perform operations on all of them

        Args:
            region_detections (List[RegionDetections]): A list of individual detections
        """
        raise NotImplementedError()

    def save(self, save_path: PATH_TYPE):
        """Save the data to a geospatial file

        Args:
            save_path (PATH_TYPE):
               File to save the data to. The containing folder will be created if it does not exist.
        """
        raise NotImplementedError()
