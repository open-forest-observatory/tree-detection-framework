from itertools import cycle, repeat
from typing import List, Tuple, Union

import geopandas as gpd
import numpy as np
import pytest
from PIL import Image
from shapely.geometry import box

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.detection.region_detections import (
    RegionDetections,
    RegionDetectionsSet,
)
from tree_detection_framework.postprocessing.postprocessing import (
    remove_masked_detections,
    remove_maskfile_detections,
)
from tree_detection_framework.utils.geometric import ellipse_mask


def make_detections(
    corners: List[Tuple[float, float, float, float]], return_gdf: bool
) -> Union[RegionDetections, gpd.GeoDataFrame]:
    """
    Create a GeoDataFrame with rectangular polygons from bounding box corners.

    Arguments:
        corners: List of (minx, miny, maxx, maxy) tuples defining rectangular areas.
        gdf (bool): If true return a GDF with these corner detections directly, if
            False return the gdf packaged in a RegionDetections class

    Returns:
        A GeoDataFrame with one row per rectangle.
    """
    polygons = [box(minx, miny, maxx, maxy) for minx, miny, maxx, maxy in corners]
    gdf = gpd.GeoDataFrame({"geometry": polygons})

    if return_gdf is True:
        return gdf
    else:
        return RegionDetections(detection_geometries=None, data=gdf)


def make_mask_image(
    path: PATH_TYPE,
    valid_classes: List[int],
    corners: List[Tuple[int, int, int, int]],
    shape: List[int],
):
    """
    Helper function to save an array with multiple polygonal valid masked areas.

    Arguments:
        path: Where to save the resulting file.
        valid_classes: List of values to use as "good" in the mask. For example,
            if 0 was ground in a mask and 1 was above-ground, valid_classes would be
            [1]. If there are multiple valid classes (perhaps [1, 2, 3] all indicate
            that an area is good) then this builder function will cycle through the
            valid values when masking areas. Should be uint8.
        corners: List of (minx, miny, maxx, maxy) tuples defining rectangular areas
            we want to mask.
        shape: The (H, W) of the array we want to build.

    Returns: None, but a (H, W, 1) image will be saved to the given path.
    """

    mask = np.zeros(shape, dtype=np.uint8)

    for value, (minx, miny, maxx, maxy) in zip(cycle(valid_classes), corners):
        mask[miny:maxy, minx:maxx] = value

    # Convert to (H, W, 1) and save using PIL
    mask_img = Image.fromarray(mask[:, :, None].astype(np.uint8).squeeze(), mode="L")
    mask_img.save(path)


def get_detections(corners, use_rds, file_dir, geo_extension, N=3):
    """
    Helper function to turn corner values into either a list of RegionDetectionsSet
    objects, or a list paths to saved geospatial files
    """

    if use_rds is True:
        detection_list = [
            RegionDetectionsSet(
                region_detections=[
                    make_detections(corners, return_gdf=False) for _ in range(N)
                ]
            )
            for _ in range(N)
        ]
    else:
        detection_data = [make_detections(corners, return_gdf=True) for _ in range(N)]
        # Write the data out as geospatial files and return the file paths
        detection_list = [file_dir / f"image_{i}.{geo_extension}" for i in range(N)]
        for path, gdf in zip(detection_list, detection_data):
            gdf.to_file(path)

    return detection_list


def check_expected_rds(rds, use_rds, expected_indices):
    """Helper file to check the final RDS indices."""

    if use_rds is True:
        # In this case we should have recieved a list of RDS objects, one
        # per image (and on RD per chip)
        assert isinstance(rds, RegionDetectionsSet)
        for rd in rds.region_detections:
            assert set(rd.get_data_frame().index) == set(expected_indices)
    else:
        # In this case we should have recieved a list of GDF dataframes, one
        # per image
        assert isinstance(rds, gpd.GeoDataFrame)
        assert set(rds.index) == set(expected_indices)


class TestRemoveMaskfileDetections:
    @pytest.mark.parametrize(
        "use_rds,geo_extension",
        ([True, None], [False, ".gpkg"], [False, ".geojson"], [False, ".shp"]),
    )
    @pytest.mark.parametrize("flat", (True, False))
    @pytest.mark.parametrize("valid_classes", ([2], [1, 2, 3], [100, 5, 20]))
    def test_basic(self, tmp_path, valid_classes, flat, use_rds, geo_extension):

        # Test the following code on N different masks. It doesn't matter that they
        # are all the same, I just want to make sure that it works on multiple
        N = 3

        # Test whether this works in a flat directory where the files are directly
        # present (flat = True) and when given nested subdirectory files (flat = False)
        if flat is True:
            file_dir = tmp_path
        else:
            file_dir = tmp_path / "mission" / "00" / "folder"
            file_dir.mkdir(parents=True)

        # Make the detections that we want to filter
        corners = [
            (0, 0, 10, 10),  # 100% good
            (41, 0, 51, 10),  # 90% good
            (0, 22, 10, 32),  # 80% good
            (60, 27, 70, 37),  # 70% good
            (46, 35, 56, 45),  # 60% good
            (45, 25, 55, 35),  # 50% good
            (80, 56, 90, 66),  # 40% good
            (97, 30, 107, 40),  # 30% good
            (110, 52, 120, 62),  # 20% good
            (91, 70, 101, 80),  # 10% good
            (0, 50, 10, 60),  # 0% good
        ]

        # Test both that the function works when given a list of RDS objects, and also that it
        # works when given a list of geospatial files.
        detection_list = get_detections(corners, use_rds, file_dir, geo_extension)

        # Fake paths for images corresponding to these detections
        image_paths = [file_dir / f"image_{i}.jpg" for i in range(N)]

        # Make the masks that we want to filter on
        for i in range(N):
            make_mask_image(
                path=file_dir / f"image_{i}.png",
                valid_classes=valid_classes,
                corners=[(0, 0, 50, 30), (50, 30, 100, 60), (100, 60, 120, 100)],
                shape=[100, 120],
            )

        # Filter the detections. We know that as the threshold goes up we should
        # get fewer detections remaining
        for threshold, max_expected_index in (
            (0.05, 9),
            (0.15, 8),
            (0.25, 7),
            (0.35, 6),
            (0.45, 5),
            (0.55, 4),
            (0.65, 3),
            (0.75, 2),
            (0.85, 1),
            (0.95, 0),
        ):
            filtered_detection_list = remove_maskfile_detections(
                region_detection_sets=detection_list,
                image_root=tmp_path,
                image_paths=image_paths,
                valid_classes=valid_classes,
                mask_root=tmp_path,
                mask_extension=".png",
                threshold=threshold,
            )

            # Check that we kept the expected rows. For example, if the threshold is 0.75, we
            # would expect to keep rows [0, 1, 2]
            for rds in filtered_detection_list:
                check_expected_rds(rds, use_rds, range(max_expected_index + 1))


class TestRemoveMaskedDetections:

    @pytest.mark.parametrize(
        "use_rds,geo_extension",
        ([True, None], [False, ".gpkg"], [False, ".geojson"], [False, ".shp"]),
    )
    @pytest.mark.parametrize("ellipse_radius", [10, 30, 50])
    @pytest.mark.parametrize("flat", (True, False))
    def test_basic(self, tmp_path, flat, ellipse_radius, use_rds, geo_extension):

        # Test the following code on N different masks. It doesn't matter that they
        # are all the same, I just want to make sure that it works on multiple
        N = 3

        # Test whether this works in a flat directory where the files are directly
        # present (flat = True) and when given nested subdirectory files (flat = False)
        if flat is True:
            file_dir = tmp_path
        else:
            file_dir = tmp_path / "mission" / "00" / "folder"
            file_dir.mkdir(parents=True)

        # Make the detections that we want to filter
        corners = [
            (0, 0, 10, 10),  # Always out
            (140, 140, 150, 150),  # Always out
            (70, 70, 80, 80),  # Always in
            (50, 70, 60, 80),  # In for [30, 50] radii
            (90, 70, 100, 80),  # In for [30, 50] radii
            (70, 50, 80, 60),  # In for [30, 50] radii
            (70, 90, 80, 100),  # In for [30, 50] radii
            (110, 70, 120, 80),  # In for [50] radius
            (30, 70, 40, 80),  # In for [50] radius
            (70, 110, 80, 120),  # In for [50] radius
            (70, 30, 80, 40),  # In for [50] radius
        ]

        # Test both that the function works when given a list of RDS objects, and also that it
        # works when given a list of geospatial files.
        detection_list = get_detections(corners, use_rds, file_dir, geo_extension)

        # Create a variably sized ellipse mask
        ellipse = ellipse_mask(
            image_shape=(150, 150),
            center=(75, 75),
            axes=(ellipse_radius, ellipse_radius),
        )

        # Filter the detections. We know that as the threshold goes up we should
        # get fewer detections remaining
        filtered_detection_list = remove_masked_detections(
            region_detection_sets=detection_list,
            mask_iterator=repeat(ellipse),
            threshold=0.5,
        )

        # Check that we kept the expected rows. Based on the radius of the
        # constructed ellipse we know what the kept indices should be.
        expected_map = {10: [2], 30: [2, 3, 4, 5, 6], 50: [2, 3, 4, 5, 6, 7, 8, 9, 10]}
        for rds in filtered_detection_list:
            check_expected_rds(rds, use_rds, expected_map[ellipse_radius])
