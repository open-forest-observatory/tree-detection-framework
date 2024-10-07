from tree_detection_framework.constants import PATH_TYPE
from typing import List
import geopandas as gpd
import pyproj
import numpy as np


class RegionDetections:
    def __init__(
        self, detections: gpd.GeoDataFrame, pixel_to_CRS_transform: np.array, CRS: pyproj.CRS
    ):
        raise NotImplementedError()

    def save(self, save_path: PATH_TYPE):
        raise NotImplementedError()


class RegionDetectionsSet:
    def __init__(self, region_detections: List[RegionDetections]):
        raise NotImplementedError()

    def save(self, save_path: PATH_TYPE):
        raise NotImplementedError()
