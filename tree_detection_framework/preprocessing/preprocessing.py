import json
import logging
import random
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
    CustomRasterDataset,
    CustomVectorDataset,
)
from tree_detection_framework.utils.geospatial import get_projected_CRS


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

    Returns:
        DataLoader:
            A dataloader containing tiles from the raster data and optionally corresponding labels
            from the vector data.
    """
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
            "Either 'chip_size' or 'chip_overlap_percentage' must be provided."
        )

    logging.info(f"Stride = {chip_stride}")

    # GridGeoSampler to get contiguous tiles
    sampler = GridGeoSampler(
        final_dataset, size=chip_size, stride=chip_stride, units=units
    )
    dataloader = DataLoader(final_dataset, sampler=sampler, collate_fn=stack_samples)

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

        # Plot the sample image. `dataloader.dataset.datasets[0]` selects the raster_dataset.
        dataloader.dataset.datasets[0].plot(sample)
        plt.axis("off")
        plt.show()


def save_dataloader_contents(
    dataloader: DataLoader,
    save_folder: PATH_TYPE,
    n_tiles: Optional[int] = None,
    random_sample: bool = False,
):
    """Save contents of the dataloader to a folder

    Args:
        dataloader (DataLoader):
            Dataloader to save the contents of
        save_folder (PATH_TYPE):
            Folder to save data to. Will be created if it doesn't exist.
        n_tiles (Optional[int], optional):
            How many tiles to saved. Whether they are the first tiles or random is controlled by
            `random_sample`. If unset, all tiles will be saved. Defaults to None.
        random_sample: (bool, optional):
            If `n_tiles` is set, should the tiles be randomly sampled rather than taken from the
            beginning of the dataloader. Defaults to False.
    """
    # Creates save directory if it doesn't exist
    save_folder.mkdir(parents=True, exist_ok=True)

    transform_to_pil = ToPILImage()
    for i, batch in enumerate(dataloader):
        sample = unbind_samples(batch)[0]

        image = sample["image"]
        image_tensor = torch.clamp(image / 255.0, min=0, max=1)
        pil_image = transform_to_pil(image_tensor)
        pil_image.save(save_folder / f"tile_{i}.png")

        # Prepare tile metadata
        metadata = {
            "crs": sample["crs"].to_string(),
            "bounds": list(sample["bounds"]),
        }

        # if vector dataset is part of the dataloader (i.e., includes 2 datasets), save crown metadata as well
        if len(dataloader.dataset.datasets) == 2:
            # Extract shapes (polygons and tree IDs)
            shapes = sample["shapes"]

            crowns = [
                {"ID": tree_id, "crown": polygon.wkt} for polygon, tree_id in shapes
            ]

            # Add crowns to the metadata
            metadata["crowns"] = crowns

        # Save tile metadata to a json file
        with open(save_folder / f"tile_{i}.json", "w") as f:
            json.dump(metadata, f, indent=4)

    logging.info(f"Saved {i + 1} tiles to {save_folder}")
