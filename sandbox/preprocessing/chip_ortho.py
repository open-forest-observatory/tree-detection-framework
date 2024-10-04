import argparse
import json
import random
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
from typing import Optional, Dict

import matplotlib.pyplot as plt
import pyproj
import rasterio
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, VectorDataset, IntersectionDataset, stack_samples, unbind_samples
from torchgeo.samplers import GridGeoSampler, Units
from torchvision.transforms import ToPILImage


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
    filename_glob = "*.gpkg" #".*\.(gpkg|geojson)$"


def chip_orthomosaics(
    raster_path: str,
    vector_path: str,
    size: float,
    stride: Optional[float] = None,
    overlap_percent: Optional[float] = None,
    res: Optional[float] = None,
    use_units_meters: bool = False,
    save_dir: Optional[str] = None,
    visualize_n: Optional[int] = None,
) -> None:
    """
    Splits an orthomosaic image into smaller tiles with optional reprojection to a meters-based CRS. Tiles can be saved to a directory and visualized.

    Args:
        path (str): Path to the folder containing the orthomosaic files.
        size (float): Tile size in units of pixels or meters, depending on `use_units_meters`.
        stride (float, optional): The distance between the start of one tile and the next in pixels or meters.
        overlap (float, optional): Percentage overlap between consecutive tiles (0-100%). Used to calculate stride if provided.
        res (float, optional): Resolution of the dataset in units of the CRS (if not specified, defaults to the resolution of the first image).
        use_units_meters (bool, optional): Whether to use meters instead of pixels for tile size and stride.
        save_dir (str, optional): Directory where the tiles and metadata should be saved.
        visualize_n (int, optional): Number of randomly selected tiles to visualize.

    Raises:
        ValueError: If neither `stride` nor `overlap` are provided.
    """

    # Stores image data
    raster_dataset = CustomRasterDataset(paths=raster_path, res=res)

    # Stores label data
    vector_dataset = CustomVectorDataset(paths=vector_path, res=res)

    units = Units.CRS if use_units_meters == True else Units.PIXELS
    print("Units = ", units)

    if use_units_meters and raster_dataset.crs.is_geographic:
        # Reproject the dataset to a meters-based CRS
        print("Projecting to meters-based CRS...")
        lat, lon = raster_dataset.bounds[2], raster_dataset.bounds[0]

        # Return a new projected CRS value with meters units
        projected_crs = get_projected_CRS(lat, lon)

        # Type conversion to rasterio.crs
        projected_crs = rasterio.crs.CRS.from_wkt(projected_crs.to_wkt())

        # Recreating the raster and vector dataset objects with the new CRS value
        raster_dataset = CustomRasterDataset(paths=raster_path, crs=projected_crs)
        vector_dataset = CustomVectorDataset(paths=vector_path, crs=projected_crs)
    
    # Create an intersection dataset that combines raster and label data
    intersection = IntersectionDataset(raster_dataset, vector_dataset)

    # Calculate stride if overlap is provided
    if overlap_percent:
        stride = size * (1 - overlap_percent / 100.0)
        print("Calculated stride based on overlap: " + str(stride))
    elif stride is None:
        raise ValueError("Either 'stride' or 'overlap' must be provided.")
    print("Stride = ", stride)

    # GridGeoSampler to get contiguous tiles
    sampler = GridGeoSampler(intersection, size=size, stride=stride, units=units)
    dataloader = DataLoader(intersection, sampler=sampler, collate_fn=stack_samples)

    if visualize_n:
        # Randomly pick indices for visualizing tiles if visualize_n is specified
        visualize_indices = random.sample(range(len(sampler)), visualize_n)

        for i in visualize_indices:
            plot(get_sample_from_index(raster_dataset, sampler, i))
            plt.axis("off")
            plt.show()    

    if save_dir:
        # Creates save directory if it doesn't exist
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        transform_to_pil = ToPILImage()
        for i, batch in enumerate(dataloader):
            sample = unbind_samples(batch)[0]

            # Save image as PNG
            image = sample["image"]
            image_tensor = torch.clamp(image / 255.0, min=0, max=1)
            pil_image = transform_to_pil(image_tensor)
            pil_image.save(Path(save_dir) / f"tile_{i}.png")

            # Save the mask as a numpy file
            mask = sample["mask"].squeeze().numpy()
            np.save(Path(save_dir) / f"tile_{i}_mask.npy", mask)

            # Save per-tile geojson files with crowns and other information
            tile_bounds = sample["bounds"]

            # Convert tile bounds to Polygon
            tile_bounds = box(tile_bounds.minx, tile_bounds.miny, tile_bounds.maxx, tile_bounds.maxy)

            # Read original vector data, create a new dataframe for the tile
            crowns_gdf = gpd.read_file(vector_dataset.files[0])
            tile_bbox = gpd.GeoDataFrame(
                geometry=[tile_bounds],
                crs=crowns_gdf.crs
            )

            # Spatial join of tile bounds and tree bounds within that region
            crowns_tile = gpd.sjoin(crowns_gdf, tile_bbox, how="inner", predicate='intersects')
            if not crowns_tile.empty:
                crowns_tile.to_file(Path(save_dir) / f"tile_{i}_crowns.geojson", driver="GeoJSON")
            else:
                print("No crowns found for tile "+str(i))
        print("Saved " + str(i + 1) + " tiles to " + save_dir)


# Helper functions

def get_sample_from_index(dataset: CustomRasterDataset, sampler: GridGeoSampler, index: int) -> Dict:
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


def get_projected_CRS(lat: float, lon: float, assume_western_hem: bool = True) -> pyproj.CRS:
    if assume_western_hem and lon > 0:
        lon = -lon
    epgs_code = 32700 - round((45 + lat) / 90) * 100 + round((183 + lon) / 6)
    crs = pyproj.CRS.from_epsg(epgs_code)
    return crs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chipping orthomosaic images")
    parser.add_argument(
        "--raster-path", type=str, required=True, help="Path to folder containing single or multiple orthomosaic images."
    )
    parser.add_argument(
        "--vector-path", type=str, required=True, help="Path to folder containing single or multiple vector datafiles."
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
