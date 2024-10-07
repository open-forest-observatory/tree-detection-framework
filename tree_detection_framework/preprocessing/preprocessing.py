from typing import Optional, List
from torch.utils.data import DataLoader
import shapely

from tree_detection_framework.constants import PATH_TYPE, BOUNDARY_TYPE, ARRAY_TYPE


def create_spatial_split(
    region_to_be_split: BOUNDARY_TYPE, split_fractions: ARRAY_TYPE
) -> List[shapely.MultiPolygon]:
    raise NotImplementedError()


def create_dataloader(
    raster_folder_path: PATH_TYPE,
    chip_size: float,
    chip_stride: float,
    region_of_interest: Optional[BOUNDARY_TYPE] = None,
    output_resolution: Optional[float] = None,
    vector_file_path: Optional[PATH_TYPE] = None,
    use_units_meters: bool = False,
) -> DataLoader:
    raise NotImplementedError()


def visualize_dataloader(dataloader: DataLoader, n_tiles):
    raise NotImplementedError()


def save_dataloader_contents(dataloader: DataLoader, save_folder: PATH_TYPE):
    raise NotImplementedError()
