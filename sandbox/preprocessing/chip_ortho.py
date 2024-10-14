import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import fiona
import fiona.transform
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import shapely.geometry
import torch
from shapely.affinity import affine_transform
from shapely.geometry import box
from torch.utils.data import DataLoader
from torchgeo.datasets import (IntersectionDataset, RasterDataset,
                               VectorDataset, stack_samples, unbind_samples)
from torchgeo.datasets.utils import BoundingBox, array_to_tensor
from torchgeo.samplers import GridGeoSampler, Units
from torchvision.transforms import ToPILImage

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


def chip_orthomosaics(
    raster_path: str,
    size: float,
    vector_path: Optional[str] = None,
    id_column_name: str = "treeID",
    stride: Optional[float] = None,
    overlap_percent: Optional[float] = None,
    res: Optional[float] = None,
    use_units_meters: bool = False,
    save_dir: Optional[str] = None,
    visualize_n: Optional[int] = None,
) -> DataLoader:
    """
    Splits an orthomosaic image into smaller tiles with optional reprojection to a meters-based CRS. Tiles can be saved to a directory and visualized.

    Args:
        raster_path (str): Path to the folder containing the orthomosaic files.
        size (float): Tile size in units of pixels or meters, depending on `use_units_meters`.
        vector_path (str, optional): Path to the folder containing the vector data files.
        id_column_name (str): Column name in the vector dataframe containing IDs for the tree polygons. Defaults to 'treeID'.
        stride (float, optional): The distance between the start of one tile and the next in pixels or meters.
        overlap (float, optional): Percentage overlap between consecutive tiles (0-100%). Used to calculate stride if provided.
        res (float, optional): Resolution of the dataset in units of the CRS (if not specified, defaults to the resolution of the first image).
        use_units_meters (bool, optional): Whether to use meters instead of pixels for tile size and stride.
        save_dir (str, optional): Directory where the tiles and metadata should be saved.
        visualize_n (int, optional): Number of randomly selected tiles to visualize.

    Returns:
        A dataloader with chipped orthomosaic tiles.

    Raises:
        ValueError: If neither `stride` nor `overlap` are provided.
    """

    # Stores image data
    raster_dataset = CustomRasterDataset(paths=raster_path, res=res)

    # Stores label data (hardcoded label_name for now)
    vector_dataset = (
        CustomVectorDataset(paths=vector_path, res=res, label_name=id_column_name)
        if vector_path is not None
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
        raster_dataset = CustomRasterDataset(paths=raster_path, crs=projected_crs)
        vector_dataset = (
            CustomVectorDataset(
                paths=vector_path, crs=projected_crs, label_name=id_column_name
            )
            if vector_path
            else None
        )

    # Create an intersection dataset that combines raster and label data if given. Otherwise, proceed with just raster_dataset.
    final_dataset = (
        IntersectionDataset(raster_dataset, vector_dataset)
        if vector_path is not None
        else raster_dataset
    )

    # Calculate stride if overlap is provided
    if overlap_percent:
        stride = size * (1 - overlap_percent / 100.0)
        logging.info(f"Calculated stride based on overlap: {stride}")
    elif stride is None:
        raise ValueError("Either 'stride' or 'overlap' must be provided.")
    logging.info(f"Stride = {stride}")

    # GridGeoSampler to get contiguous tiles
    sampler = GridGeoSampler(final_dataset, size=size, stride=stride, units=units)
    dataloader = DataLoader(final_dataset, sampler=sampler, collate_fn=stack_samples)

    if visualize_n:
        # Randomly pick indices for visualizing tiles if visualize_n is specified
        visualize_indices = random.sample(range(len(sampler)), visualize_n)

        for i in visualize_indices:
            plot(get_sample_from_index(raster_dataset, sampler, i))
            plt.axis("off")
            plt.show()

    if save_dir:
        # Creates save directory if it doesn't exist
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        transform_to_pil = ToPILImage()
        for i, batch in enumerate(dataloader):
            sample = unbind_samples(batch)[0]

            image = sample["image"]
            image_tensor = torch.clamp(image / 255.0, min=0, max=1)
            pil_image = transform_to_pil(image_tensor)
            pil_image.save(Path(save_dir) / f"tile_{i}.png")

            # Prepare to save tile metadata
            metadata = {
                "crs": sample["crs"].to_string(),
                "bounds": list(sample["bounds"]),
            }

            if vector_path:
                # Extract shapes (polygons and tree IDs)
                shapes = sample["shapes"]

                crowns = [
                    {"ID": tree_id, "crown": polygon.wkt} for polygon, tree_id in shapes
                ]

                # Add crowns to the metadata
                metadata["crowns"] = crowns

            # Save tile metadata to a json file
            with open(Path(save_dir) / f"tile_{i}.json", "w") as f:
                json.dump(metadata, f, indent=4)

        logging.info(f"Saved {i + 1} tiles to {save_dir}")

    return dataloader


# Helper functions


def get_sample_from_index(
    dataset: CustomRasterDataset, sampler: GridGeoSampler, index: int
) -> Dict:
    # Access the specific index from the sampler containing bounding boxes
    sample_indices = list(sampler)
    sample_idx = sample_indices[index]

    # Get the sample from the dataset using this index
    sample = dataset[sample_idx]
    return sample


def plot(sample: Dict) -> plt.Figure:
    image = sample["image"].permute(1, 2, 0)
    image = image.byte().numpy()
    fig, ax = plt.subplots()
    ax.imshow(image)
    return fig


def get_projected_CRS(
    lat: float, lon: float, assume_western_hem: bool = True
) -> pyproj.CRS:
    if assume_western_hem and lon > 0:
        lon = -lon
    epgs_code = 32700 - round((45 + lat) / 90) * 100 + round((183 + lon) / 6)
    crs = pyproj.CRS.from_epsg(epgs_code)
    return crs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chipping orthomosaic images")
    parser.add_argument(
        "--raster-path",
        type=str,
        required=True,
        help="Path to folder containing single or multiple orthomosaic images.",
    )
    parser.add_argument(
        "--vector-path",
        type=str,
        required=False,
        help="Path to folder containing single or multiple vector datafiles.",
    )
    parser.add_argument(
        "--id-column-name",
        type=str,
        default="treeID",
        help="Column name in the vector dataframe containing IDs for the tree polygons. Defaults to 'treeID'.",
    )
    parser.add_argument(
        "--res",
        type=float,
        required=False,
        help="Resolution of the dataset in units of CRS (defaults to the resolution of the first file found)",
    )
    parser.add_argument(
        "--size",
        type=float,
        required=True,
        help="Single value used for height and width dim",
    )
    parser.add_argument(
        "--stride",
        type=float,
        required=False,
        help="Distance to skip between each patch",
    )
    parser.add_argument(
        "--overlap-percent",
        type=float,
        required=False,
        help="Percentage overlap between the tiles (0-100%)",
    )
    parser.add_argument(
        "--use-units-meters",
        action="store_true",
        help="Whether to set units for tile size and stide as meters",
    )
    parser.add_argument(
        "--save-dir", type=str, required=False, help="Directory to save chips"
    )
    parser.add_argument(
        "--visualize-n", type=int, required=False, help="Number of tiles to visualize"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    chip_orthomosaics(**args.__dict__)
