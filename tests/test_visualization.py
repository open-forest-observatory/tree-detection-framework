import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from matplotlib import colormaps
from PIL import Image
from shapely.geometry import Polygon

from tree_detection_framework.detection.region_detections import RegionDetectionsSet
from tree_detection_framework.utils.visualization import show_filtered_detections


# Helper to create a dummy image file
def save_dummy_image(imdir, size=(20, 20), color=(255, 255, 255)):
    img = Image.new("RGB", size, color)
    path = imdir / "test_image.png"
    img.save(path)
    return path


def save_dummy_gdf(gdir, N):
    # Create DataFrame
    df = pd.DataFrame(
        {
            "unique_ID": list(range(N)),
            "geometry": [
                Polygon(
                    [
                        (3 * i, 3 * i),
                        (3 * i + 2, 3 * i),
                        (3 * i + 2, 3 * i + 2),
                        (3 * i, 3 * i + 2),
                    ]
                )
                for i in range(N)
            ],
        }
    )
    # Save GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    path = gdir / f"test_gdf_{N}.gpkg"
    gdf.to_file(path)
    return path


class TestShowFilteredDetections:

    @pytest.mark.parametrize("mask_dtype", [int, bool])
    @pytest.mark.parametrize(
        "mask_colormap",
        [
            None,
            {
                0: (np.array(colormaps["tab20"](0)) * 255).astype(np.uint8),
                1: (np.array(colormaps["tab20"](1)) * 255).astype(np.uint8),
            }
        ],
    )
    def test_basic_functionality(self, tmp_path, mask_dtype, mask_colormap):

        # Create dummy files
        impath = save_dummy_image(tmp_path)
        gdf1 = save_dummy_gdf(tmp_path, 3)
        gdf2 = save_dummy_gdf(tmp_path, 2)

        # Create mask
        mask = np.zeros((20, 20), dtype=mask_dtype)
        mask[:10, :10] = 1

        # Run function
        det_img, mask_img = show_filtered_detections(
            impath=impath,
            detection1=gdf1,
            detection2=gdf2,
            mask=mask,
            mask_colormap=mask_colormap,
        )
        assert det_img.shape == (20, 20, 3)
        assert mask_img.shape == (20, 20, 3)

        # Spot check a couple of areas. We know there were three detections, two
        # matches and a no-match
        greenish = det_img[[1, 4], [1, 4]]
        assert np.all(greenish[:, 1] > greenish[:, 0])
        assert np.all(greenish[:, 1] > greenish[:, 2])
        reddish = det_img[[7], [7]]
        assert np.all(reddish[:, 0] > reddish[:, 1])
        assert np.all(reddish[:, 0] > reddish[:, 2])

        # Check the mask map in a few spots
        if mask_colormap is None:
            # In this case the color is alpha blended with the image (white) so it
            # won't be an exact match
            assert np.argmax(mask_img[-1, -1]) == np.argmax(colormaps["tab20"](0)[:3])
            assert np.argmax(mask_img[0, 0]) == np.argmax(colormaps["tab20"](1)[:3])
            # The tab20[1] should be brighter
            assert np.sum(mask_img[0, 0]) > np.sum(mask_img[-1, -1])
        else:
            assert np.allclose(mask_img[-5:, -5:].reshape(-1, 3), mask_colormap[0][:3])
            assert np.allclose(mask_img[:5, :5].reshape(-1, 3), mask_colormap[1][:3])

        assert np.any(mask_img != 255)  # Should have some non-white pixels
