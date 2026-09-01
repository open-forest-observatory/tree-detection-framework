from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy.typing
import shapely
import torch

PATH_TYPE = Union[str, Path]
"""A path to a file or folder, given either as a string or a `pathlib.Path`."""

BOUNDARY_TYPE = Union[
    PATH_TYPE, shapely.Polygon, shapely.MultiPolygon, gpd.GeoDataFrame, gpd.GeoSeries
]
"""A spatial region, such as a region of interest to restrict processing to.

May be given as a path to a geospatial vector file, an in-memory shapely geometry, or a geopandas
`GeoDataFrame` or `GeoSeries`.
"""

ARRAY_TYPE = numpy.typing.ArrayLike
"""Any object numpy can interpret as an array, such as a numpy array, list, or tuple."""

DATA_FOLDER = Path(Path(__file__).parent, "..", "data").resolve()
"""The `data` folder at the top level of the project.

The example notebooks read their inputs from here, and the example data should be extracted into
this folder.
"""

CHECKPOINTS_FOLDER = Path(Path(__file__).parent, "..", "checkpoints").resolve()
"""The `checkpoints` folder at the top level of the project.

The default weights paths for Detectree2, SAM2, and SAM3 are resolved against this folder.
"""

DEFAULT_DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
"""The device detectors run on by default: the GPU if one is available, otherwise the CPU."""
