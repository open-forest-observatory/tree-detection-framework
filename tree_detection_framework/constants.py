from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy.typing
import shapely
import torch

PATH_TYPE = Union[str, Path]
BOUNDARY_TYPE = Union[
    PATH_TYPE, shapely.Polygon, shapely.MultiPolygon, gpd.GeoDataFrame, gpd.GeoSeries
]
ARRAY_TYPE = numpy.typing.ArrayLike

DATA_FOLDER = Path(Path(__file__).parent, "..", "data").resolve()

DEFAULT_DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
