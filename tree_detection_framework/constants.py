from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import shapely

PATH_TYPE = Union[str, Path]
BOUNDARY_TYPE = Union[
    PATH_TYPE, shapely.Polygon, shapely.MultiPolygon, gpd.GeoDataFrame, gpd.GeoSeries
]
ARRAY_TYPE = np.typing.ArrayLike
