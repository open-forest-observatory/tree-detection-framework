import copy
from typing import List, Optional

import numpy as np
import geopandas as gpd
import pyproj


def get_projected_CRS(
    lat: float, lon: float, assume_western_hem: bool = True
) -> pyproj.CRS:
    """
    Returns a projected Coordinate Reference System (CRS) based on latitude and longitude.

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        assume_western_hem (bool): Assumes the longitude is in the Western Hemisphere. Defaults to True.

    Returns:
        pyproj.CRS: The projected CRS corresponding to the UTM zone for the given latitude and longitude.
    """

    if assume_western_hem and lon > 0:
        lon = -lon
    epgs_code = 32700 - round((45 + lat) / 90) * 100 + round((183 + lon) / 6)
    crs = pyproj.CRS.from_epsg(epgs_code)
    return crs


def to_crs_multiple_geometry_columns(
    data: gpd.GeoDataFrame,
    crs: pyproj.CRS,
    columns_to_transform: Optional[List[str]] = None,
    inplace: bool = False,
) -> Optional[gpd.GeoDataFrame]:
    """Behaves like gpd.GeoDataFrame.to_crs() except supports transforming multiple geometry columns

    Args:
        data (gpd.GeoDataFrame): Input data to transform
        crs (pyproj.CRS): The output CRS
        columns_to_transform (Optional[List[str]], optional):
            Which columns to transform. If unset, all geometry-typed columns will be. Defaults to
            None.
        inplace (bool, optional):
            Should the input geodataframe be modified. Defaults to False.

    Returns:
        Optional[gpd.GeoDataFrame]: Returns the modified geodataframe if inplace=False, else None
    """
    # If columns to transform are not explicitly provided, assume any geometry-typed columns should be transformed
    if columns_to_transform is None:
        columns_to_transform = data.columns[data.dtypes == "geometry"].tolist()

    # Create a copy of the data unless we want to modify the original data
    if not inplace:
        data = copy.deepcopy(data)

    # For each column we should transform, do the projection
    for col in columns_to_transform:
        data[col] = data[col].to_crs(crs)

    # Only return if not inplace, consistent with most other libraries
    if not inplace:
        return data

# Taken from here:
# https://stackoverflow.com/questions/6430091/efficient-distance-calculation-between-n-points-and-a-reference-in-numpy-scipy
# This is drop-in replacement for scipy.cdist
def cdist(x, y):
    """
    Compute pair-wise distances between points in x and y.

    Parameters:
        x (ndarray): Numpy array of shape (n_samples_x, n_features).
        y (ndarray): Numpy array of shape (n_samples_y, n_features).

    Returns:
        ndarray: Numpy array of shape (n_samples_x, n_samples_y) containing
        the pair-wise distances between points in x and y.
    """
    # Reshape x and y to enable broadcasting
    x_reshaped = x[:, np.newaxis, :]  # Shape: (n_samples_x, 1, n_features)
    y_reshaped = y[np.newaxis, :, :]  # Shape: (1, n_samples_y, n_features)

    # Compute pair-wise distances using Euclidean distance formula
    pairwise_distances = np.sqrt(np.sum((x_reshaped - y_reshaped) ** 2, axis=2))

    return pairwise_distances
