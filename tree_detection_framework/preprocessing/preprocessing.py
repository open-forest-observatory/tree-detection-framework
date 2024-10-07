from typing import List, Optional

import pyproj
import shapely
from torch.utils.data import DataLoader

from tree_detection_framework.constants import ARRAY_TYPE, BOUNDARY_TYPE, PATH_TYPE


def create_spatial_split(
    region_to_be_split: BOUNDARY_TYPE, split_fractions: ARRAY_TYPE
) -> List[shapely.MultiPolygon]:
    """Creates non-overlapping spatial splits

    Args:
        region_to_be_split (BOUNDARY_TYPE):
            A spatial region to be split up. May be defined as a shapely object, geopandas object,
            or a path to a geospatial file. In any case, the union of all the elements will be taken.
        split_fractions (ARRAY_TYPE):
            A sequence of fractions to split the input region into. If they don't sum to 1, the total
            wlil be normalized.

    Returns:
        List[shapely.MultiPolygon]:
            A list of regions representing spatial splits of the input. The area of each one is
            controlled by the corresponding element in split_fractions.

    """
    raise NotImplementedError()


def create_dataloader(
    raster_folder_path: PATH_TYPE,
    chip_size: float,
    chip_stride: float,
    use_units_meters: bool = False,
    region_of_interest: Optional[BOUNDARY_TYPE] = None,
    output_resolution: Optional[float] = None,
    output_CRS: Optional[pyproj.CRS] = None,
    vector_label_folder_path: Optional[PATH_TYPE] = None,
    vector_label_attribute: Optional[str] = None,
) -> DataLoader:
    """
    Create a tiled dataloader using torchgeo. Contains raster data data and optionally vector labels

    Args:
        raster_folder_path (PATH_TYPE): Path to the folder or raster files
        chip_size (float):
            Dimension of the chip. May be pixels or meters, based on `use_units_meters`.
        chip_stride (float):
            Stride of the chip. May be pixels or meters, based on `use_units_meters`.
        use_units_meters (bool, optional):
            Use units of meters rather than pixels when interpreting the `chip_size` and `chip_stride`.
            Defaults to False.
        region_of_interest (Optional[BOUNDARY_TYPE], optional):
            Only data from this spatial region will be included in the dataloader. Defaults to None.
        output_resolution (Optional[float], optional):
            Spatial resolution the data in meters/pixel. If un-set, will be the resolution of the
            first raster data that is read. Defaults to None.
        output_CRS: (Optional[pyproj.CRS], optional):
            The coordinate reference system to use for the output data. If un-set, will be the CRS
            of the first tile found. Defaults to None.
        vector_label_folder_path (Optional[PATH_TYPE], optional):
            A folder of geospatial vector files that will be used for the label. If un-set, the
            dataloader will not be labeled. Defaults to None.
        vector_label_attribute (Optional[str], optional):
            Attribute to read from the vector data, such as the class or instance ID. Defaults to None.

    Returns:
        DataLoader:
            A dataloader containing tiles from the raster data and optionally corresponding labels
            from the vector data.
    """
    raise NotImplementedError()


def visualize_dataloader(dataloader: DataLoader, n_tiles: int):
    """Show samples from the dataloader

    Args:
        dataloader (DataLoader): The dataloader to visualize
        n_tiles (int): The number of randomly-sampled tiles to show
    """
    raise NotImplementedError()


def save_dataloader_contents(dataloader: DataLoader, save_folder: PATH_TYPE):
    """Save contents of the dataloader to a folder

    Args:
        dataloader (DataLoader): Dataloader to save the contents of
        save_folder (PATH_TYPE): Folder to save data to. Will be created if it doesn't exist.
    """
    raise NotImplementedError()
