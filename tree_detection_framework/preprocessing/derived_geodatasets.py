from typing import Any, Optional

import fiona
import fiona.transform
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely.geometry
from shapely.affinity import affine_transform
from torch.utils.data import DataLoader
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import (
    IntersectionDataset,
    RasterDataset,
    VectorDataset,
    stack_samples,
)
from torchgeo.datasets.utils import BoundingBox, array_to_tensor
from torchgeo.samplers import GridGeoSampler, Units


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

    def plot_rgb(self, sample):
        """
        Plots an image from the dataset.

        Args:
            sample (dict): A dictionary containing the tile to plot. The 'image' key should have a tensor of shape (C, H, W).

        Returns:
            matplotlib.figure.Figure: A figure containing the plotted image.
        """
        # Reorder and rescale the image
        image = sample["image"].permute(1, 2, 0)
        image = image.byte().numpy()

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)
        return fig


class CustomVectorDataset(VectorDataset):
    """
    Custom dataset class for vector data which act as labels for the raster data. This class extends the `VectorDataset` from `torchgeo`.
    """

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

        # Add `shapes` and `bounding_boxes` to the dictionary.
        sample = {
            "mask": masks,
            "crs": self.crs,
            "bounds": query,
            "shapes": pixel_transformed_shapes,
            "bounding_boxes": bounding_boxes,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


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
