from typing import Union
from pathlib import Path
import geopandas as gpd
import shapely
import numpy as np

PATH_TYPE = Union[str, Path]
BOUNDARY_TYPE = Union[
    PATH_TYPE, shapely.Polygon, shapely.MultiPolygon, gpd.GeoDataFrame, gpd.GeoSeries
]
ARRAY_TYPE = np.typing.ArrayLike
