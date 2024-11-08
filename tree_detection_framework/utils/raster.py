import tempfile
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import rasterio as rio
import rasterio.plot
from rasterio.warp import Resampling, calculate_default_transform, reproject

from tree_detection_framework.constants import PATH_TYPE


# Copied from https://github.com/open-forest-observatory/geograypher/blob/2900ede9a00ac8bdce22c43e4abb6d74876390f6/geograypher/utils/geospatial.py#L333
def load_downsampled_raster_data(dataset_filename: PATH_TYPE, downsample_factor: float):
    """Load a raster file spatially downsampled

    Args:
        dataset (PATH_TYPE): Path to the raster
        downsample_factor (float): Downsample factor of 10 means that pixels are 10 times larger

    Returns:
        np.array: The downsampled array in the rasterio (c, h, w) convention
        rio.DatasetReader: The reader with the transform updated
        rio.Transform: The updated transform
    """
    # Open the dataset handler. Note that this doesn't read into memory.
    dataset = rio.open(dataset_filename)

    # resample data to target shape
    data = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height / downsample_factor),
            int(dataset.width / downsample_factor),
        ),
        resampling=rio.enums.Resampling.bilinear,
    )

    # scale image transform
    updated_transform = dataset.transform * dataset.transform.scale(
        (dataset.width / data.shape[-1]), (dataset.height / data.shape[-2])
    )
    # Return the data and the transform
    return data, dataset, updated_transform


def reproject_raster(
    input_file: PATH_TYPE, output_file: PATH_TYPE, dst_crs: PATH_TYPE
) -> PATH_TYPE:
    """_summary_

    Args:
        input_file (PATH_TYPE): _description_
        output_file (PATH_TYPE): _description_
        dst_crs (PATH_TYPE): _description_

    Returns:
        PATH_TYPE: _description_
    """
    # Taken from here: https://rasterio.readthedocs.io/en/latest/topics/reproject.html
    # Open the source raster
    with rasterio.open(input_file, "r") as src:
        # If it is in the desired CRS, then return the input file since this is a no-op
        if dst_crs == src.crs:
            return input_file

        # Calculate the parameters of the transform for the data in the new CRS
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        # Create updated metadata for this new file
        kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

        # Open the output file
        with rasterio.open(output_file, "w", **kwargs) as dst:
            # Perform reprojection per band
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )
    # Return the output file path that the reprojected raster was written to.
    return output_file


def show_raster(
    raster_file_path: PATH_TYPE,
    downsample_factor: float = 10.0,
    plt_ax: Optional[plt.axes] = None,
    CRS: Optional[pyproj.CRS] = None,
):
    """Show a raster, optionally downsampling or reprojecting it

    Args:
        raster_file_path (PATH_TYPE):
            Path to the raster file
        downsample_factor (float):
            How much to downsample the raster before visualization, this makes it faster and consume
            less memory
        plt_ax (Optional[plt.axes], optional):
            Axes to plot on, otherwise the current ones are used. Defaults to None.
        CRS (Optional[pyproj.CRS], optional):
            The CRS to reproject the data to if set. Defaults to None.
    """
    # If the CRS is set, ensure the dat matches it
    if CRS is not None:
        # Create a temporary file to write to
        temp_output_filename = tempfile.NamedTemporaryFile(suffix=".tif")
        # Get the name of this file
        temp_name = temp_output_filename.name
        # Reproject the raster. If the CRS was the same as requested, the original raster path will
        # be returned. Otherwise, the reprojected raster will be written to the temp file and that
        # path will be returned.
        raster_file_path = reproject_raster(
            input_file=raster_file_path, output_file=temp_name, dst_crs=CRS
        )
    # Load the downsampled image
    img, _, transform = load_downsampled_raster_data(
        raster_file_path, downsample_factor=downsample_factor
    )
    # Plot the image
    rio.plot.show(source=img, transform=transform, ax=plt_ax)
