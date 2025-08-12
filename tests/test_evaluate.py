from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Point

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.detection.region_detections import (
    RegionDetections,
    RegionDetectionsSet,
)
from tree_detection_framework.evaluation.evaluate import match_points
from tree_detection_framework.utils.geospatial import get_projected_CRS

CRS = get_projected_CRS(lat=38.6274, lon=90.1982)
INPUT_TYPES = ["RegionDetections", "RegionDetectionsSet", "GeoDataFrame"]


def create_test_chm_tif(
    temp_dir, height_values=None, bounds=(-1, -1, 100, 100), crs=CRS
):
    """
    Create a test CHM TIF file with specified height values.

    Args:
        temp_dir (PATH_TYPE): Directory to save the TIF file.
        height_values (np.ndarray, optional): 2D array of height values.
            If None, creates a simple 10x10 grid with heights 1-15m.
        bounds (tuple): (minx, miny, maxx, maxy) bounds in CRS units.
            Defaults to (-1, -1, 100, 100).
        crs (str): Coordinate reference system for the raster.

    Returns:
        pathlib.Path: Path to the created TIF file
    """
    if height_values is None:
        # Create a simple 10x10 grid with heights from 5 to 15 meters
        height_values = np.linspace(1, 15, 100).reshape(10, 10)

    # Create transform from bounds
    minx, miny, maxx, maxy = bounds
    height, width = height_values.shape
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Write the TIF file
    tif_path = Path(temp_dir) / "test_chm.tif"
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=height_values.dtype,
        crs=crs,
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(height_values, 1)
    return tif_path


def make_inputs(input_type, points, heights, crs=CRS):
    if input_type == "GeoDataFrame":
        return gpd.GeoDataFrame({"geometry": points, "height": heights}, crs=crs)
    elif input_type.startswith("RegionDetections"):
        rd = RegionDetections(points, data={"height": heights}, CRS=crs)
        if input_type == "RegionDetections":
            return rd
        elif input_type == "RegionDetectionsSet":
            return RegionDetectionsSet([rd])
    raise ValueError(f"Unknown input_type: {input_type}")


class TestMatchPoints:

    def check_types(self, matches):
        for match in matches:
            assert isinstance(match, tuple)
            assert len(match) == 3
            assert isinstance(match[0], (int, np.integer))
            assert isinstance(match[1], (int, np.integer))
            assert isinstance(match[2], float)
            # All elements (indices and distance) should be non-negative
            assert np.all(np.array(match) >= 0)

    @pytest.mark.parametrize("type1", INPUT_TYPES)
    @pytest.mark.parametrize("type2", INPUT_TYPES)
    @pytest.mark.parametrize("height_threshold", [lambda x: 0.5 * x, 2.0, np.inf])
    @pytest.mark.parametrize("distance_threshold", [lambda x: 0.2 * x + 2, 2.0, np.inf])
    @pytest.mark.parametrize("use_height_in_distance", [1.0, 0.0])
    @pytest.mark.parametrize(
        "add_kwargs",
        [
            {},
            {"height_column_1": "height", "height_column_2": "height"},
            {"fillin_method": "chm", "chm_path": "temporary"},
        ],
    )
    def test_basic_match(
        self,
        tmp_path,
        type1,
        type2,
        height_threshold,
        distance_threshold,
        use_height_in_distance,
        add_kwargs,
    ):
        """
        In all argument combinations, we should have a three matches
        with no distance.
        """

        # Avoid certain combinations
        if len(add_kwargs) == 0 and callable(distance_threshold):
            return

        # Points with the same location and height
        set1 = make_inputs(
            type1, points=[Point(10, 0), Point(0, 0), Point(30, 0)], heights=[20, 10, 5]
        )
        set2 = make_inputs(
            type2, points=[Point(0, 0), Point(10, 0), Point(30, 0)], heights=[10, 20, 5]
        )

        # Build the function arguments
        kwargs = {
            "height_threshold": height_threshold,
            "distance_threshold": distance_threshold,
            "use_height_in_distance": use_height_in_distance,
        }
        for key, value in add_kwargs.items():
            kwargs[key] = value
        # Special case the CHM since it requires the tmp_path fixture and can't be
        # built before the function
        if "chm_path" in kwargs:
            kwargs["chm_path"] = create_test_chm_tif(tmp_path)

        # Call match_points
        matches = match_points(set1, set2, **kwargs)
        self.check_types(matches)

        # Check that the points are matched up as expected
        assert len(matches) == 3
        for expected_indices, match in zip(([0, 1], [1, 0], [2, 2]), matches):
            idx1, idx2, distance = match
            assert [idx1, idx2] == expected_indices
            assert np.isclose(distance, 0.0)
