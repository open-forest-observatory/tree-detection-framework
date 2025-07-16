import json
import logging
import random
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import pyproj
import rasterio
import shapely
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import (
    BoundingBox,
    GeoDataset,
    IntersectionDataset,
    stack_samples,
    unbind_samples,
)
from torchgeo.samplers import GeoSampler, GridGeoSampler, Units
from torchgeo.samplers.utils import _to_tuple, tile_to_chips
from torchvision.transforms import ToPILImage

from tree_detection_framework.constants import ARRAY_TYPE, BOUNDARY_TYPE, PATH_TYPE
from tree_detection_framework.detection.region_detections import RegionDetectionsSet
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


class UnboundedGridGeoSampler(GridGeoSampler):
    """
    A variant of GridGeoSampler that optionally includes tiles smaller than the chip size.

    Default GridGeoSampler skips tiles that are too small to contain a full chip. Setting
    `include_smaller_tiles=True` overrides this behavior, and ensures all tiles are included.
    This is useful for generating a single chip from an entire raster.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: tuple[float, float] | float,
        stride: tuple[float, float] | float,
        roi: BoundingBox | None = None,
        units: Units = Units.PIXELS,
        include_smaller_tiles: bool = True,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            stride: distance to skip between each patch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` and ``stride`` are in pixel or CRS units
            allow_smaller_tiles: If True, includes all tiles regardless of size.
                If False, behaves like GridGeoSampler. Defaults to True.
        """
        GeoSampler.__init__(self, dataset, roi)
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.stride = (self.stride[0] * self.res, self.stride[1] * self.res)

        # If including smaller tiles, append all hits regardless of size
        if include_smaller_tiles is True:
            self.hits = list(self.index.intersection(tuple(self.roi), objects=True))
        else:
            # Otherwise, behave like GridGeoSampler
            self.hits = []
            for hit in self.index.intersection(tuple(self.roi), objects=True):
                bounds = BoundingBox(*hit.bounds)
                if (
                    bounds.maxx - bounds.minx >= self.size[1]
                    and bounds.maxy - bounds.miny >= self.size[0]
                ):
                    self.hits.append(hit)

        self.length = 0
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            self.length += rows * cols


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
    resolution: Optional[float] = None,
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
        resolution (Optional[float], optional):
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
    raster_dataset = CustomRasterDataset(paths=raster_folder_path, res=resolution)

    # Stores label data
    vector_dataset = (
        CustomVectorDataset(
            paths=vector_label_folder_path,
            res=resolution,
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

    # Create a sampler to generate contiguous tiles of the input dataset
    sampler = UnboundedGridGeoSampler(
        final_dataset, size=chip_size, stride=chip_stride, units=units
    )
    dataloader = DataLoader(
        final_dataset, batch_size=batch_size, sampler=sampler, collate_fn=stack_samples
    )

    return dataloader


def create_intersection_dataloader(
    raster_data: PATH_TYPE,
    vector_data: Union[PATH_TYPE, RegionDetectionsSet],
    chip_size: float,
    chip_stride: float = None,
    chip_overlap_percentage: float = None,
    use_units_meters: bool = False,
    resolution: Optional[float] = None,
    output_CRS: Optional[pyproj.CRS] = None,
):
    """
    Create a dataloader for raster data with vector labels. Used to combine detected geometry outputs from
    one detector with the input for the next detector.

    Args:
        raster_data (PATH_TYPE): Path to raster file or the folder containing raster files.
        vector_data (Union[PATH_TYPE, RegionDetectionsSet]):
            Path to the vector file or outputs from a Detector as a RegionDetectionsSet object.
        chip_size (float): Size of the chips in pixels or meters.
        chip_stride (float, optional): Stride of the chips in pixels or meters. Defaults to None.
        chip_overlap_percentage (float, optional): Percent overlap of the chips from 0-100. Defaults to None.
        use_units_meters (bool, optional): Use meters as units for chip size and stride. Defaults to False.
        resolution (Optional[float], optional):
            Spatial resolution of the output data in meters/pixel. If un-set, will be the resolution
            of the first raster data that is read. Defaults to None.

    Returns:
        DataLoader: A dataloader containing raster data and vector labels.
    """

    if chip_overlap_percentage:
        # Calculate `chip_stride` if `chip_overlap_percentage` is provided
        chip_stride = chip_size * (1 - chip_overlap_percentage / 100.0)
        logging.info(f"Calculated stride based on overlap: {chip_stride}")

    elif chip_stride is None:
        raise ValueError(
            "Either 'chip_stride' or 'chip_overlap_percentage' must be provided."
        )

    logging.info(f"Stride = {chip_stride}")

    units = Units.CRS if use_units_meters == True else Units.PIXELS
    logging.info(f"Units = {units}")

    kwargs = {}
    if resolution is not None:
        kwargs["res"] = resolution
    if output_CRS is not None:
        kwargs["crs"] = output_CRS

    # Create the vector and raster datasets
    vector_data = CustomVectorDataset(vector_data, **kwargs)
    raster_data = CustomRasterDataset(raster_data, **kwargs)

    # Create an intersection dataset that combines the datasets
    # Attributes such as resolution will be taken from the first dataset
    intersection_data = IntersectionDataset(vector_data, raster_data)

    # Create a sampler to generate contiguous tiles of the input dataset
    sampler = UnboundedGridGeoSampler(
        intersection_data, size=chip_size, stride=chip_stride, units=units
    )
    dataloader = DataLoader(
        intersection_data, sampler=sampler, collate_fn=stack_samples
    )

    return dataloader


def create_image_dataloader(
    images_dir: Union[PATH_TYPE, List[str]],
    chip_size: int,
    chip_stride: Optional[int] = None,
    chip_overlap_percentage: Optional[float] = None,
    labels_dir: Optional[List[str]] = None,
    batch_size: int = 1,
) -> DataLoader:
    """
    Create a dataloader for a folder of normal images (e.g., JPGs), tiling them into smaller patches.

    Args:
        images_dir (Union[Path, List[str]]):
            Path to the folder containing image files, or list of paths to image files.
        chip_size (int):
            Size of the tiles (width, height) in pixels.
        chip_stride (Optional[int], optional):
            Stride of the tiling (horizontal, vertical) in pixels.
        chip_overlap_percentage (Optional[float], optional):
            Percent overlap of the chip from 0-100. If used, `chip_stride` should not be set.
        labels_dir (Optional[List[str]], optional):
            List of paths to tree crown label files corresponding to the images.
            This will be used as ground truth during evaluation
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
        images_dir=images_dir,
        chip_size=chip_size,
        chip_stride=chip_stride,
        labels_dir=labels_dir,
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
