from typing import List, Optional

import geopandas as gpd
import numpy as np
import pyproj
import shapely

from tree_detection_framework.constants import PATH_TYPE


class RegionDetections:
    def __init__(
        self,
        detections: gpd.GeoDataFrame,
        pixel_to_CRS_transform: Optional[np.array] = None,
        CRS: Optional[pyproj.CRS] = None,
        prediction_bounds: Optional[shapely.Polygon | shapely.MultiPolygon] = None,
    ):
        """Create a region detections object

        Args:
            detections (gpd.GeoDataFrame):
                The dataframe of detected shapes, potentially with additional attributes such as
                class or confidence. If no CRS is set, the data will be assumed to be in pixel
                coordinates.
            pixel_to_CRS_transform (Optional[np.array], optional):
                Only meaningful if `CRS` is set as well. A 2x3 transform matrix mapping the
                coordinates in pixels to the coordinates of the CRS. If un-set, will assumed to be
                the identity transform. Defaults to None.
            CRS (Optional[pyproj.CRS], optional):
                A coordinate reference system to interpret the data in, used in conjuction with the
                `pixel_to_CRS_transform`. Only used if the CRS of `detections` is un-set. Defaults
                to None.
            prediction_bounds (Optional[shapely.Polygon | shapely.MultiPolygon], optional):
                The spatial bounds of the region that predictions were generated for. For example,
                the bounds of a tile. Defaults to None.
        """
        raise NotImplementedError()

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
