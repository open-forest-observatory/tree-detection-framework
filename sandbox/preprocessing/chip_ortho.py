import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pyproj
import rasterio
import torch
from rasterio.warp import Resampling, calculate_default_transform, reproject
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples
from torchgeo.samplers import GridGeoSampler, Units
from torchvision.transforms import ToPILImage


class CustomOrthoDataset(RasterDataset):
    """
    Custom dataset class for orthomosaic raster images. This class extends the `RasterDataset` from `torchgeo`.
    
    Attributes:
        filename_glob (str): Glob pattern to match files in the directory.
        is_image (bool): Indicates that the data being loaded is image data.
        separate_files (bool): True if data is stored in a separate file for each band, else False.
    """

    filename_glob = '*.tif'  # To match all TIFF files
    is_image = True
    separate_files = False

def chip_orthomosaics(path, size, stride=None, overlap=None, units="pixel", res=None, use_units_meters=False, save_dir=None, visualize_n=None):
    """
    Splits an orthomosaic image into smaller tiles with optional reprojection to a meters-based CRS. Tiles can be saved to a directory and visualized.

    Args:
        path (str): Path to the folder containing the orthomosaic files.
        size (float): Tile size in units of pixels or meters, depending on `use_units_meters`.
        stride (float, optional): The distance between the start of one tile and the next in pixels or meters. 
        overlap (float, optional): Percentage overlap between consecutive tiles (0-100%). Used to calculate stride if provided.
        units (str, optional): Unit of measurement for the tile size and stride ('pixel' or 'meters'). Default is 'pixel'.
        res (float, optional): Resolution of the dataset in units of the CRS (if not specified, defaults to the resolution of the first image).
        use_units_meters (bool, optional): Whether to use meters instead of pixels for tile size and stride. 
        save_dir (str, optional): Directory where the tiles and metadata should be saved.
        visualize_n (int, optional): Number of randomly selected tiles to visualize.
    
    Raises:
        ValueError: If neither `stride` nor `overlap` are provided.

    Returns:
        None
    """

    # Create dataset instance
    dataset = CustomOrthoDataset(paths=path, res=res)
    units = Units.CRS if use_units_meters == True else Units.PIXELS
    print("Units = ", units)

    if use_units_meters and dataset.crs.is_geographic:
        # Reproject the dataset to a meters-based CRS
        lat, lon = dataset.bounds[1], dataset.bounds[0]
        projected_crs = get_projected_CRS(lat, lon)
        reprojected_path = reproject_raster_to_crs(path, projected_crs)
        dataset = CustomOrthoDataset(paths=reprojected_path, res=res)

    # Calculate stride if overlap is provided
    if overlap:
        stride = size * (1 - overlap / 100.0)
        print("Calculated stride based on overlap: "+str(stride))
    elif stride is None:
        raise ValueError("Either 'stride' or 'overlap' must be provided.")
    print("Stride = ", stride)
    
    #GridGeoSampler to get contiguous tiles
    sampler = GridGeoSampler(dataset, size=size, stride=stride, units=units)
    dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

    total_tiles = len(sampler)
    # Randomly pick indices for visualization if visualize_n is specified
    visualize_indices = random.sample(range(total_tiles), visualize_n) if visualize_n else []

    # Creates save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(dataloader):
        sample = unbind_samples(batch)[0]

        # Saving logic
        if save_dir:
            image = sample['image']
            image_tensor = torch.clamp(image / 255.0, min=0, max=1)
            pil_image = ToPILImage()(image_tensor)
            pil_image.save(Path(save_dir) / f'tile_{i}.png')

            # Save tile metadata to a json file
            metadata = {
                "crs": sample['crs'].to_string(),  
                "bounds": list(sample['bounds']),
            }
            with open(Path(save_dir) / f'tile_{i}.json', 'w') as f:
                json.dump(metadata, f, indent=4)

        # Visualization logic
        if visualize_n and i in visualize_indices:
            plot(sample)
            plt.axis('off')
            plt.show()

    # Action summary
    if save_dir:
        print("Saved " + str(i + 1) + " tiles to " + save_dir)
    if visualize_n:
        print("Visualized " + str(len(visualize_indices)) + " tiles")


# Helper functions (could be moved to a separate utils file)

def plot(sample):
    image = sample['image'].permute(1, 2, 0)
    image = image.byte().numpy()
    fig, ax = plt.subplots()
    ax.imshow(image)
    return fig

def get_projected_CRS(lat, lon, assume_western_hem=True):
    if assume_western_hem and lon > 0:
        lon = -lon
    epgs_code = 32700 - round((45 + lat) / 90) * 100 + round((183 + lon) / 6)
    crs = pyproj.CRS.from_epsg(epgs_code)
    return crs

def reproject_raster_to_crs(dataset_path, projected_crs):

    with rasterio.open(dataset_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, projected_crs, src.width, src.height, *src.bounds
        )
        
        # Set up the metadata for the reprojected dataset
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': projected_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Create a new reprojected file (path - undecided)
        with rasterio.open('reprojected.tif', 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=projected_crs,
                    resampling=Resampling.nearest
                )
    
    return 'reprojected.tif' # New path


def parse_args():
    parser = argparse.ArgumentParser(description="Chipping orthomosaic images")
    parser.add_argument('--path', type=str, required=True, help="Path to folder containing orthomosaic")
    parser.add_argument('--res', type=float, required=False, help="Resolution of the dataset in units of CRS (defaults to the resolution of the first file found)")
    parser.add_argument('--size', type=float, required=True, help="Single value used for height and width dim")
    parser.add_argument('--stride', type=float, required=False, help="Distance to skip between each patch")
    parser.add_argument('--overlap', type=float, required=False, help="Percentage overlap between the tiles (0-100%)")
    parser.add_argument('--use-units-meters', action='store_true', help="Whether to set units for tile size and stide as meters")
    parser.add_argument('--save-dir', type=str, required=False, help="Directory to save chips")
    parser.add_argument('--visualize-n', type=int, required=False, help="Number of tiles to visualize")
    # to add: arg to accept different regex patterns

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    chip_orthomosaics(**args.__dict__)

