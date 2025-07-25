import logging
import os.path
import tempfile
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Any, List, Optional, Union

import fiona
import fiona.transform
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely.geometry
import torch
from PIL import Image
from shapely.affinity import affine_transform
from torch.utils.data import DataLoader, Dataset
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import (
    IntersectionDataset,
    RasterDataset,
    VectorDataset,
    stack_samples,
)
from torchgeo.datasets.utils import BoundingBox, array_to_tensor
from torchgeo.samplers import GridGeoSampler, Units
from torchvision import transforms

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.detection.region_detections import RegionDetectionsSet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define a namedtuple to store bounds of tiles images from the `CustomImageDataset`
# Note that the y min bounds are reversed from the expected convention. This is because they are
# measured in pixel coordinates, which start at the top and go down. So this convention matches
# how the geospatial bounding box is represented.
bounding_box = namedtuple("bounding_box", ["minx", "maxx", "miny", "maxy"])


class CustomRasterDataset(RasterDataset):
    """
    Custom dataset class for orthomosaic raster images. This class extends the `RasterDataset` from `torchgeo`.

    Attributes:
        filename_glob (str): Glob pattern to match files in the directory.
        is_image (bool): Indicates that the data being loaded is image data.
        separate_files (bool): True if data is stored in a separate file for each band, else False.
    """

    filename_glob: str = "*.tif"  # To match all TIFF files
    is_image: bool = True
    separate_files: bool = False


class CustomVectorDataset(VectorDataset):
    """
    Custom dataset class for vector data which act as labels for the raster data. This class extends the `VectorDataset` from `torchgeo`.
    """

    def __init__(
        self,
        vector_data: Union[PATH_TYPE, RegionDetectionsSet],
        **kwargs,
    ):

        self._tempfile = None  # Will hold a NamedTemporaryFile if we generate one

        if isinstance(vector_data, RegionDetectionsSet):
            # Get the merged GeoDataFrame for the RegionDetectionsSet
            vector_data_gdf = vector_data.merge().get_data_frame()

            # Save GeoDataFrame to a temp file that persists until object is deleted
            self._tempfile = tempfile.NamedTemporaryFile(suffix=".geojson", delete=True)
            vector_data_gdf.to_file(self._tempfile.name, driver="GeoJSON")
            vector_data = self._tempfile.name

            # Update CRS to be used for the created VectorDataset object
            kwargs["crs"] = vector_data_gdf.crs

            logging.info(f"RegionDetectionsSet temporarily saved to: {vector_data}")

        super().__init__(paths=vector_data, **kwargs)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.
           This function is largely based on the `__getitem__` method from TorchGeo's `VectorDataset`.
           Modifications have been made to include the following keys within the returned dictionary:
            1. 'shapes' as polygons per tile represented in pixel coordinates.
            2. 'bounding_boxes' as bounding box of every detected polygon per tile in pixel coordinates.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        shapes = []
        attributes = []
        for filepath in filepaths:
            with fiona.open(filepath) as src:
                # We need to know the bounding box of the query in the source CRS
                (minx, maxx), (miny, maxy) = fiona.transform.transform(
                    self.crs.to_dict(),
                    src.crs,
                    [query.minx, query.maxx],
                    [query.miny, query.maxy],
                )

                # Filter geometries to those that intersect with the bounding box
                for feature in src.filter(bbox=(minx, miny, maxx, maxy)):
                    # Warp geometries to requested CRS
                    shape = fiona.transform.transform_geom(
                        src.crs, self.crs.to_dict(), feature["geometry"]
                    )
                    label = self.get_label(feature)
                    shapes.append((shape, label))
                    attributes.append(feature["properties"])  # Non-geometry properties

        # Rasterize geometries
        width = (query.maxx - query.minx) / self.res
        height = (query.maxy - query.miny) / self.res
        transform = rasterio.transform.from_bounds(
            query.minx, query.miny, query.maxx, query.maxy, width, height
        )
        if shapes:
            masks = rasterio.features.rasterize(
                shapes, out_shape=(round(height), round(width)), transform=transform
            )
        else:
            # If no features are found in this query, return an empty mask
            # with the default fill value and dtype used by rasterize
            masks = np.zeros((round(height), round(width)), dtype=np.uint8)

        # Use array_to_tensor since rasterize may return uint16/uint32 arrays.
        masks = array_to_tensor(masks)

        masks = masks.to(self.dtype)

        # Beginning of additions made to this function

        # Invert the transform to convert geo coordinates to pixel values
        inverse_transform = ~transform

        # Convert `fiona` type shapes to `shapely` shape objects for easier manipulation
        shapely_shapes = [(shapely.geometry.shape(sh), i) for sh, i in shapes]

        # Apply the inverse transform to each shapely shape, converting geo coordinates to pixel coordinates
        pixel_transformed_shapes = [
            (affine_transform(sh, inverse_transform.to_shapely()), i)
            for sh, i in shapely_shapes
        ]

        # Convert each polygon to an axis-aligned bounding box of format (x_min, y_min, x_max, y_max) in pixel coordinates
        bounding_boxes = []
        for polygon, _ in pixel_transformed_shapes:
            x_min, y_min, x_max, y_max = polygon.bounds
            bounding_boxes.append([x_min, y_min, x_max, y_max])

        # Convert list of fiona.Properties into dict of lists
        if attributes:
            all_keys = attributes[0].keys()
            # Create a dictionary where each key is an attribute name, and each value is a list
            # containing the values of that attribute from all features in the current tile.
            attributes_dict = {
                key: [attr[key] for attr in attributes] for key in all_keys
            }
        else:
            attributes_dict = {}

        # Add shapes, bounding_boxes, and attributes to the dictionary.
        sample = {
            "mask": masks,
            "crs": self.crs,
            "bounds": query,
            "shapes": pixel_transformed_shapes,
            "bounding_boxes": bounding_boxes,
            "attributes": attributes_dict,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class CustomImageDataset(Dataset):
    def __init__(
        self,
        images_dir: Union[PATH_TYPE, List[PATH_TYPE]],
        chip_size: int,
        chip_stride: int,
        labels_dir: Optional[List[PATH_TYPE]] = None,
    ):
        """
        Dataset for creating a dataloader from a folder of individual images, with an option to create tiles.

        Args:
            images_dir (Union[Path, List[PATH_TYPE]]): Path to a folder containing potentially nested
                image files, or list of paths to image files. By 'nested' image files, that means that
                images could reside in arbitrarily nested subdirectories and this dataset should iterate
                over all of them. E.g.
                    images_dir/
                        subdir1/
                            [images]
                        subdir2/
                            [images]
            chip_size (int): Dimension of each image chip (width, height) in pixels.
            chip_stride (int): Stride to take while chipping the images (horizontal, vertical) in pixels.
            labels_dir (Optional[List[PATH_TYPE]]): List of paths to annotation .geojson files corresponding to the images.
                Should have same file name as the image.
        """
        self.chip_size = chip_size
        self.chip_stride = chip_stride

        if not isinstance(images_dir, list):
            self.images_root_dir = Path(images_dir)
            # Get a list of all image paths
            image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
            self.image_paths = sorted(
                [
                    path
                    for path in self.images_root_dir.glob("**/*")
                    if path.suffix.lower() in image_extensions
                ]
            )

            if len(self.image_paths) == 0:
                raise ValueError(f"No image files found in {self.images_root_dir}")
        else:
            # Save this input as the list of image paths we want to iterate over
            self.image_paths = images_dir
            # Calculate the root of the images
            self.images_root_dir = Path(os.path.commonpath(self.image_paths))

        self.labels_paths = labels_dir
        self.tile_metadata = self._get_metadata(image_root=self.images_root_dir)

    def _get_metadata(self, image_root: PATH_TYPE):
        """
        Create a metadata entry for each image. If label paths are provided, check that
        each maps to an image and include it in the entry.

        Arguments:
            image_root (PATH_TYPE): Should be the root directory where all images
                are nested under. We assert that the labels files should have
                a similar root, and the labels structure beneath that root should
                match that of the images.
        """

        # Prepare some necessary structures if labels are provided
        if self.labels_paths is not None:
            label_root = Path(os.path.commonpath(self.labels_paths))
            label_pathset = set([str(path) for path in self.labels_paths])

        metadata = []
        for img_path in self.image_paths:

            # If label paths were given, ensure the paths corresponds to the same
            # nested path as each image
            label_path = None
            if self.labels_paths is not None:
                # Match the label file by replacing the image extension with `.geojson`
                # and mimicking the directory structure
                subpath = Path(img_path).relative_to(image_root)
                label_path = (label_root / subpath).with_suffix(".geojson")
                if str(label_path) not in label_pathset:
                    raise ValueError(
                        f"Label file [{label_path}] not found for image: [{img_path}]"
                    )

            tile_idx = 0  # A unique tile index value within this image, resets for every new image
            with Image.open(img_path) as img:
                img_width, img_height = img.size

                # Generate tile coordinates
                for y in range(0, img_height, self.chip_stride):
                    for x in range(0, img_width, self.chip_stride):
                        # Add metadata for the current tile
                        metadata.append((tile_idx, img_path, label_path, x, y))
                        tile_idx += 1

        return metadata

    def __len__(self):
        return len(self.tile_metadata)

    def __getitem__(self, idx):

        img_idx, img_path, label_path, x, y = self.tile_metadata[idx]

        with Image.open(img_path) as img:
            img = img.convert("RGB")

            # Check if the tile extends beyond the image boundary
            tile_width = min(self.chip_size, img.width - x)
            tile_height = min(self.chip_size, img.height - y)

            # If the tile fits within the image, return the cropped image
            if tile_width == self.chip_size and tile_height == self.chip_size:
                tile = img.crop((x, y, x + self.chip_size, y + self.chip_size))
            else:
                # Create a white square tile of shape 'chip_size'
                tile = Image.new(
                    "RGB", (self.chip_size, self.chip_size), (255, 255, 255)
                )

                # Crop the image section and paste onto the white image
                img_section = img.crop((x, y, x + tile_width, y + tile_height))
                tile.paste(img_section, (0, 0))

            # Convert to tensor
            if not isinstance(tile, torch.Tensor):
                tile = transforms.ToTensor()(tile)

            metadata = {
                "image_index": img_idx,
                "source_image": str(img_path),
                "image_bounds": bounding_box(
                    0,
                    float(img.width),
                    float(img.height),
                    0,
                ),
            }
            if self.labels_paths is not None:
                metadata["annotations"] = str(label_path)

            return {
                "image": tile,
                "metadata": metadata,
                # Bounds includes bounding box values for the whole tile including white padded region if any
                "bounds": bounding_box(
                    float(x),
                    float(x + self.chip_size),
                    float(y + self.chip_size),
                    float(y),
                ),
                "crs": None,
            }

    @staticmethod
    def collate_as_defaultdict(batch):
        # Stack images from batch into a single tensor
        images = torch.stack([item["image"] for item in batch])
        # Collect metadata as a list
        metadata = [item["metadata"] for item in batch]
        bounds = [item["bounds"] for item in batch]
        crs = [item["crs"] for item in batch]
        return defaultdict(
            lambda: None,
            {"image": images, "metadata": metadata, "bounds": bounds, "crs": crs},
        )


class CustomDataModule(GeoDataModule):
    # TODO: Add docstring
    def __init__(
        self,
        output_res: float,
        train_raster_path: str,
        vector_label_name: str,
        train_vector_path: str,
        size: int,
        stride: int,
        batch_size: int = 2,
        val_raster_path: Optional[str] = None,
        val_vector_path: Optional[str] = None,
        test_raster_path: Optional[str] = None,
        test_vector_path: Optional[str] = None,
    ) -> None:
        super().__init__(dataset_class=IntersectionDataset)
        self.output_res = output_res
        self.vector_label_name = vector_label_name
        self.size = size
        self.stride = stride
        self.batch_size = batch_size

        # Paths for train, val and test dataset
        self.train_raster_path = train_raster_path
        self.val_raster_path = val_raster_path
        self.test_raster_path = test_raster_path
        self.train_vector_path = train_vector_path
        self.val_vector_path = val_vector_path
        self.test_vector_path = test_vector_path

    def create_intersection_dataset(
        self, raster_path: str, vector_path: str
    ) -> IntersectionDataset:
        raster_data = CustomRasterDataset(paths=raster_path, res=self.output_res)
        vector_data = CustomVectorDataset(
            paths=vector_path, res=self.output_res, label_name=self.vector_label_name
        )
        return raster_data & vector_data  # IntersectionDataset

    def setup(self, stage=None):
        # create the data based on the stage the Trainer is in
        if stage == "fit":
            self.train_data = self.create_intersection_dataset(
                self.train_raster_path, self.train_vector_path
            )
        if stage == "validate" or stage == "fit":
            self.val_data = self.create_intersection_dataset(
                self.val_raster_path, self.val_vector_path
            )
        if stage == "test":
            self.test_data = self.create_intersection_dataset(
                self.test_raster_path, self.test_vector_path
            )

    def train_dataloader(self) -> DataLoader:
        sampler = GridGeoSampler(self.train_data, size=self.size, stride=self.stride)
        return DataLoader(
            self.train_data,
            sampler=sampler,
            collate_fn=stack_samples,
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        sampler = GridGeoSampler(
            self.val_data, size=self.size, stride=self.stride, units=Units.CRS
        )
        return DataLoader(
            self.val_data,
            sampler=sampler,
            collate_fn=stack_samples,
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> DataLoader:
        sampler = GridGeoSampler(
            self.test_data, size=self.size, stride=self.stride, units=Units.CRS
        )
        return DataLoader(
            self.test_data,
            sampler=sampler,
            collate_fn=stack_samples,
            batch_size=self.batch_size,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        return batch
