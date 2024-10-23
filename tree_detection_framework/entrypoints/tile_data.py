import argparse
from typing import Optional

import pyproj

from tree_detection_framework.constants import ARRAY_TYPE, BOUNDARY_TYPE, PATH_TYPE
from tree_detection_framework.preprocessing.preprocessing import (
    create_dataloader,
    save_dataloader_contents,
    visualize_dataloader,
)


def tile_data(
    raster_folder_path: PATH_TYPE,
    chip_size: float,
    chip_stride: Optional[float] = None,
    chip_overlap_percentage: float = None,
    use_units_meters: bool = False,
    region_of_interest: Optional[BOUNDARY_TYPE] = None,
    output_resolution: Optional[float] = None,
    output_CRS: Optional[pyproj.CRS] = None,
    vector_label_folder_path: Optional[PATH_TYPE] = None,
    vector_label_attribute: Optional[str] = None,
    visualize_n_tiles: Optional[int] = None,
    save_folder: Optional[PATH_TYPE] = None,
    save_n_tiles: Optional[int] = None,
    random_sample: bool = False,
    batch_size: int = 1,
):
    """
    Entrypoint script for testing preprocessing functions.
    It enables creating a dataloader, visualizing sample tiles from the dataloader, and saving the contents of the dataloader to disk.

    Args:
        raster_folder_path (PATH_TYPE): Path to the folder or raster files.
        chip_size (float):
            Dimension of the chip. May be pixels or meters, based on `use_units_meters`.
        chip_stride (Optional[float], optional):
            Stride of the chip. May be pixels or meters, based on `use_units_meters`. If used,
            `chip_overlap_percentage` should not be set. Defaults to None.
        chip_overlap_percentage (Optional[float], optional):
            Percent overlap of the chip from 0-100. If used, `chip_stride` should not be set.
            Defaults to None.
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
        visualize_n_tiles (int):
            The number of randomly-sampled tiles to display.
        save_folder (Optional[PATH_TYPE], optional):
            Folder to save data to. Will be created if it doesn't exist.
        save_n_tiles (Optional[int], optional):
            Number of tiles to save. Whether they are the first tiles or random is controlled by
            `random_sample`. If unset, all tiles will be saved. Defaults to None.
        random_sample (bool, optional):
            If `save_n_tiles` is set, should the tiles be randomly sampled rather than taken from the
            beginning of the dataloader. Defaults to False.
        batch_size (int, optional):
            Number of images to load in a batch. Defaults to 1.
    """
    # Create the dataloader by passing folder path to raster data and optionally a path to the vector data folder.
    dataloader = create_dataloader(
        raster_folder_path=raster_folder_path,
        chip_size=chip_size,
        chip_stride=chip_stride,
        chip_overlap_percentage=chip_overlap_percentage,
        use_units_meters=use_units_meters,
        region_of_interest=region_of_interest,
        output_resolution=output_resolution,
        output_CRS=output_CRS,
        vector_label_folder_path=vector_label_folder_path,
        vector_label_attribute=vector_label_attribute,
        batch_size=batch_size,
    )

    # If `visualize_n_tiles` is specified, display those many number of tiles.
    if visualize_n_tiles is not None:
        visualize_dataloader(dataloader=dataloader, n_tiles=visualize_n_tiles)

    # If path to save tiles is given, save all the tiles (or `n_tiles`) from the dataloader to disk. Tiles can be randomly sampled or ordered.
    if save_folder is not None:
        save_dataloader_contents(
            dataloader=dataloader,
            save_folder=save_folder,
            n_tiles=save_n_tiles,
            random_sample=random_sample,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chipping orthomosaic images")

    parser.add_argument(
        "--raster-folder-path",
        type=str,
        required=True,
        help="Path to the folder or raster files.",
    )

    parser.add_argument(
        "--chip-size",
        type=float,
        required=True,
        help="Dimension of the chip. May be pixels or meters, based on --use-units-meters.",
    )

    parser.add_argument(
        "--chip-stride",
        type=float,
        required=False,
        help="Stride of the chip. May be pixels or meters, based on --use-units-meters. If used, --chip-overlap-percentage should not be set.",
    )

    parser.add_argument(
        "--chip-overlap-percentage",
        type=float,
        required=False,
        help="Percent overlap of the chip from 0-100. If used, --chip-stride should not be set.",
    )

    parser.add_argument(
        "--use-units-meters",
        action="store_true",
        help="Use units of meters rather than pixels when interpreting the --chip-size and --chip-stride.",
    )

    parser.add_argument(
        "--region-of-interest",
        type=str,
        required=False,
        help="Only data from this spatial region will be included in the dataloader. Should be specified as minx,miny,maxx,maxy.",
    )

    parser.add_argument(
        "--output-resolution",
        type=float,
        required=False,
        help="Spatial resolution of the data in meters/pixel. If un-set, will be the resolution of the first raster data that is read.",
    )

    parser.add_argument(
        "--output-CRS",
        type=str,
        required=False,
        help="The coordinate reference system to use for the output data. If un-set, will be the CRS of the first tile found.",
    )

    parser.add_argument(
        "--vector-label-folder-path",
        type=str,
        required=False,
        help="A folder of geospatial vector files that will be used for the label. If un-set, the dataloader will not be labeled.",
    )

    parser.add_argument(
        "--vector-label-attribute",
        type=str,
        default="treeID",
        help="Attribute to read from the vector data, such as the class or instance ID. Defaults to 'treeID'.",
    )

    parser.add_argument(
        "--visualize-n-tiles",
        type=int,
        required=False,
        help="The number of randomly-sampled tiles to display.",
    )

    parser.add_argument(
        "--save-folder",
        type=str,
        required=False,
        help="Folder to save data to. Will be created if it doesn't exist.",
    )

    parser.add_argument(
        "--save-n-tiles",
        type=int,
        required=False,
        help="Number of tiles to save. Whether they are the first tiles or random is controlled by --random-sample. If unset, all tiles will be saved.",
    )

    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="If --save-n-tiles is set, should the tiles be randomly sampled rather than taken from the beginning of the dataloader.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        help="Number of images to load in a batch. Defaults to 1.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    tile_data(**args.__dict__)
