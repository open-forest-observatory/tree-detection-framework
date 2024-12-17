import json
import logging
import random
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pyproj
import rasterio
import shapely
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import IntersectionDataset, stack_samples, unbind_samples
from torchgeo.samplers import GridGeoSampler, Units
from torchvision.transforms import ToPILImage

from tree_detection_framework.constants import ARRAY_TYPE, BOUNDARY_TYPE, PATH_TYPE
from tree_detection_framework.preprocessing.derived_geodatasets import (
    CustomImageDataset,
    CustomRasterDataset,
    CustomVectorDataset,
)
from tree_detection_framework.utils.geospatial import get_projected_CRS
from tree_detection_framework.utils.raster import plot_from_dataloader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
    chip_stride: Optional[float] = None,
    chip_overlap_percentage: float = None,
    use_units_meters: bool = False,
    region_of_interest: Optional[BOUNDARY_TYPE] = None,
    output_resolution: Optional[float] = None,
    output_CRS: Optional[pyproj.CRS] = None,
    vector_label_folder_path: Optional[PATH_TYPE] = None,
    vector_label_attribute: Optional[str] = None,
    batch_size: int = 1,
) -> DataLoader:
    """
    Create a tiled dataloader using torchgeo. Contains raster data data and optionally vector labels

    Args:
        raster_folder_path (PATH_TYPE): Path to the folder or raster files
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
        batch_size (int, optional):
            Number of images to load in a batch. Defaults to 1.

    Returns:
        DataLoader:
            A dataloader containing tiles from the raster data and optionally corresponding labels
            from the vector data.
    """

    # changes: 1. bounding box included in every sample as a df / np array
    # 2. TODO: float or uint8 images
    # match with the param dict from the model, else error out
    # Stores image data
    raster_dataset = CustomRasterDataset(
        paths=raster_folder_path, res=output_resolution
    )

    # Stores label data
    vector_dataset = (
        CustomVectorDataset(
            paths=vector_label_folder_path,
            res=output_resolution,
            label_name=vector_label_attribute,
        )
        if vector_label_folder_path is not None
        else None
    )

    units = Units.CRS if use_units_meters == True else Units.PIXELS
    logging.info(f"Units = {units}")

    if use_units_meters and raster_dataset.crs.is_geographic:
        # Reproject the dataset to a meters-based CRS
        logging.info("Projecting to meters-based CRS...")
        lat, lon = raster_dataset.bounds[2], raster_dataset.bounds[0]

        # Return a new projected CRS value with meters units
        projected_crs = get_projected_CRS(lat, lon)

        # Type conversion to rasterio.crs
        projected_crs = rasterio.crs.CRS.from_wkt(projected_crs.to_wkt())

        # Recreating the raster and vector dataset objects with the new CRS value
        raster_dataset = CustomRasterDataset(
            paths=raster_folder_path, crs=projected_crs
        )
        vector_dataset = (
            CustomVectorDataset(
                paths=vector_label_folder_path,
                crs=projected_crs,
                label_name=vector_label_attribute,
            )
            if vector_label_folder_path is not None
            else None
        )

    # Create an intersection dataset that combines raster and label data if given. Otherwise, proceed with just raster_dataset.
    final_dataset = (
        IntersectionDataset(raster_dataset, vector_dataset)
        if vector_label_folder_path is not None
        else raster_dataset
    )

    if chip_overlap_percentage:
        # Calculate `chip_stride` if `chip_overlap_percentage` is provided
        chip_stride = chip_size * (1 - chip_overlap_percentage / 100.0)
        logging.info(f"Calculated stride based on overlap: {chip_stride}")

    elif chip_stride is None:
        raise ValueError(
            "Either 'chip_stride' or 'chip_overlap_percentage' must be provided."
        )

    logging.info(f"Stride = {chip_stride}")

    # GridGeoSampler to get contiguous tiles
    sampler = GridGeoSampler(
        final_dataset, size=chip_size, stride=chip_stride, units=units
    )
    dataloader = DataLoader(
        final_dataset, batch_size=batch_size, sampler=sampler, collate_fn=stack_samples
    )

    return dataloader


def create_image_dataloader(
    folder_path: Path,
    chip_size: int,
    chip_stride: Optional[int] = None,
    chip_overlap_percentage: Optional[float] = None,
    batch_size: int = 1,
) -> DataLoader:
    """
    Create a dataloader for a folder of normal images (e.g., JPGs), tiling them into smaller patches.

    Args:
        folder_path (Path):
            Path to the folder containing image files.
        chip_size (int):
            Size of the tiles (width, height) in pixels.
        chip_stride (Optional[int], optional):
            Stride of the tiling (horizontal, vertical) in pixels.
        chip_overlap_percentage (Optional[float], optional):
            Percent overlap of the chip from 0-100. If used, `chip_stride` should not be set.
        batch_size (int, optional):
            Number of tiles in a batch. Defaults to 1.

    Returns:
        DataLoader: A dataloader containing the tiles and associated metadata.
    """

    logging.info("Units set in PIXELS")

    if chip_overlap_percentage:
        # Calculate `chip_stride` if `chip_overlap_percentage` is provided
        chip_stride = chip_size * (1 - chip_overlap_percentage / 100.0)
        chip_stride = int(chip_stride)
        logging.info(f"Calculated stride based on overlap: {chip_stride}")

    elif chip_stride is None:
        raise ValueError(
            "Either 'chip_stride' or 'chip_overlap_percentage' must be provided."
        )

    dataset = CustomImageDataset(
        folder_path=folder_path,
        chip_size=chip_size,
        chip_stride=chip_stride,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CustomImageDataset.collate_as_defaultdict,
    )
    return dataloader


def visualize_dataloader(dataloader: DataLoader, n_tiles: int):
    """Show samples from the dataloader.

    Args:
        dataloader (DataLoader): The dataloader to visualize.
        n_tiles (int): The number of randomly-sampled tiles to show.
    """
    # Get a random sample of `n_tiles` index values to visualize
    tile_indices = random.sample(range(len(dataloader.sampler)), n_tiles)

    # Get a list of all tile bounds from the sampler
    list_of_bboxes = list(dataloader.sampler)

    for i in tile_indices:
        sample_bbox = list_of_bboxes[i]

        # Get the referenced sample from the dataloader
        sample = dataloader.dataset[sample_bbox]

        # Plot the sample image.
        plot_from_dataloader(sample)
        plt.axis("off")
        plt.show()


def save_dataloader_contents(
    dataloader: DataLoader,
    save_folder: PATH_TYPE,
    n_tiles: Optional[int] = None,
    random_sample: bool = False,
):
    """Save contents of the dataloader to a folder.

    Args:
        dataloader (DataLoader):
            Dataloader to save the contents of.
        save_folder (PATH_TYPE):
            Folder to save data to. Will be created if it doesn't exist.
        n_tiles (Optional[int], optional):
            Number of tiles to save. Whether they are the first tiles or random is controlled by
            `random_sample`. If unset, all tiles will be saved. Defaults to None.
        random_sample (bool, optional):
            If `n_tiles` is set, should the tiles be randomly sampled rather than taken from the
            beginning of the dataloader. Defaults to False.
    """
    # Create save directory if it doesn't exist
    destination_folder = Path(save_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)

    transform_to_pil = ToPILImage()

    # TODO: handle batch_size > 1
    # Collect all batches from the dataloader
    all_batches = list(dataloader)

    # Flatten the list of batches into individual samples
    all_samples = [sample for batch in all_batches for sample in unbind_samples(batch)]

    # If `n_tiles` is set, limit the number of tiles to save
    if n_tiles is not None:
        if random_sample:
            # Randomly sample `n_tiles`. If `n_tiles` is greater than available samples, include all samples.
            selected_samples = random.sample(
                all_samples, min(n_tiles, len(all_samples))
            )
        else:
            # Take first `n_tiles`
            selected_samples = all_samples[:n_tiles]
    else:
        selected_samples = all_samples

    # Counter for saved tiles
    saved_tiles_count = 0

    # Iterate over the selected samples
    for sample in selected_samples:
        image = sample["image"]
        image_tensor = torch.clamp(image / 255.0, min=0, max=1)
        pil_image = transform_to_pil(image_tensor)

        # Save the image tile
        pil_image.save(destination_folder / f"tile_{saved_tiles_count}.png")

        # Prepare tile metadata
        metadata = {
            "crs": sample["crs"].to_string(),
            "bounds": list(sample["bounds"]),
        }

        # If dataset includes labels, save crown metadata
        if isinstance(dataloader.dataset, IntersectionDataset):
            shapes = sample["shapes"]
            crowns = [
                {"ID": tree_id, "crown": polygon.wkt} for polygon, tree_id in shapes
            ]
            metadata["crowns"] = crowns

        # Save metadata to a JSON file
        with open(destination_folder / f"tile_{saved_tiles_count}.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # Increment the saved tile count
        saved_tiles_count += 1

        # Stop once the desired number of tiles is saved
        if n_tiles is not None and saved_tiles_count >= n_tiles:
            break

    print(f"Saved {saved_tiles_count} tiles to {save_folder}")
