from typing import Any

import fiona
import fiona.transform
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely.geometry
from shapely.affinity import affine_transform
from torchgeo.datasets import RasterDataset, VectorDataset
from torchgeo.datasets.utils import BoundingBox, array_to_tensor


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

    def plot(self, sample):
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
           This function is largely based on the `__getitem__` method from torchgeo's `VectorDataset`, with custom modifications for this implementation.

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

        # Add 'shapes' containing polygons and corresponding ID values
        sample = {
            "mask": masks,
            "crs": self.crs,
            "bounds": query,
            "shapes": pixel_transformed_shapes,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
