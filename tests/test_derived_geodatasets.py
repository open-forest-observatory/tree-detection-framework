from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tree_detection_framework.preprocessing.derived_geodatasets import (
    CustomImageDataset,
)


def create_image(path: Path, size=(100, 100), color="blue"):
    # Note that size is in (x, y) = (width, height)
    img = Image.new("RGB", size, color=color)
    img.save(path)
    return path


def create_nested_dirs(root):
    """Create a commonly used subdir structure of
    tmp_path/
        root/
            subdir/
    """
    root.mkdir(exist_ok=True)
    subdir = root / "subdir"
    subdir.mkdir()
    return root, subdir


class TestCustomImageDataset:

    @pytest.mark.parametrize(
        "image_extension", [".JPG", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    )
    @pytest.mark.parametrize(
        "from_dir,as_str",
        [
            (True, None),
            (False, True),
            (False, False),
        ],
    )
    def test_dataset_basic(self, tmp_path, image_extension, from_dir, as_str):
        """
        Test the nested files
        root/
            img1.jpg
            subdir/
                img2.jpg
        """

        root, subdir = create_nested_dirs(tmp_path / "images")
        img1 = create_image(
            root / f"img1{image_extension}", size=(128, 64), color="blue"
        )
        img2 = create_image(
            subdir / f"img1{image_extension}", size=(128, 64), color="red"
        )

        # Either serve the images from a root directory or as a list of paths
        if from_dir:
            source = root
        else:
            if as_str:
                source = [str(img1), str(img2)]
            else:
                source = [img1, img2]

        # Create the dataset
        dataset = CustomImageDataset(
            images_dir=source,
            chip_size=32,
            chip_stride=32,
            labels_dir=None,
        )

        # Check that we have the proper number of chips, (128, 64) is 4x2
        # and there were two images
        assert len(dataset) == 16

        # Check that we got colors from both images
        count = {"red": 0, "blue": 0}
        for chip in dataset:
            # Reshape from (C, H, W) to (C, H * W) and average over pixels
            color = chip["image"].reshape(3, -1).mean(axis=1)
            if np.allclose(color, [1, 0, 0], atol=0.01):
                count["red"] += 1
            elif np.allclose(color, [0, 0, 1], atol=0.01):
                count["blue"] += 1

            # Check some typing
            assert isinstance(chip["image"], torch.Tensor)
            # And existence of certain fields
            assert "metadata" in chip
            assert "annotations" not in chip["metadata"]
            assert chip["crs"] is None
            # Check that the chip and image bounds are consistent
            for exp_x, exp_y, bounds in (
                (32, 32, chip["bounds"]),
                (128, 64, chip["metadata"]["image_bounds"]),
            ):
                # See explanatory comments on derived_geodatasets.bounding_box on why
                # we are doing miny - maxy
                assert np.isclose(bounds.maxx - bounds.minx, exp_x)
                assert np.isclose(bounds.miny - bounds.maxy, exp_y)

        assert count == {"red": 8, "blue": 8}

    @pytest.mark.parametrize("as_str", [True, False])
    def test_dataset_labels_bad(self, tmp_path, as_str):
        """
        Check that if labels are given but they don't match the image structure,
        an error is raised.
        """

        # Two images, one in a subdirectory
        imroot, imsub = create_nested_dirs(tmp_path / "images")
        img1 = create_image(imroot / "img1.jpg", size=(128, 64), color="blue")
        img2 = create_image(imsub / "img1.jpg", size=(128, 64), color="red")

        # Two label files
        labroot, labsub = create_nested_dirs(tmp_path / "labels")
        label_paths = [labroot / "img1.geojson", labroot / "img2.geojson"]
        for path in label_paths:
            path.write_text('{"type": "FeatureCollection", "features": []}')
        if as_str:
            label_paths = [str(path) for path in label_paths]

        with pytest.raises(ValueError):
            CustomImageDataset(
                images_dir=[img1, img2],
                chip_size=32,
                chip_stride=32,
                labels_dir=label_paths,
            )

    @pytest.mark.parametrize(
        "from_dir,as_str",
        [
            (True, None),
            (False, True),
            (False, False),
        ],
    )
    def test_dataset_labels(self, tmp_path, from_dir, as_str):
        """
        Check that if labels are given an annotation field is added
        """

        # Two images, one in a subdirectory
        imroot, imsub = create_nested_dirs(tmp_path / "images")
        img1 = create_image(imroot / "img1.jpg", size=(128, 64))
        img2 = create_image(imsub / "img1.jpg", size=(128, 64))

        # Two label files
        labroot, labsub = create_nested_dirs(tmp_path / "labels")
        label_paths = [labroot / "img1.geojson", labsub / "img1.geojson"]
        for path in label_paths:
            path.write_text('{"type": "FeatureCollection", "features": []}')
        if as_str:
            label_paths = [str(path) for path in label_paths]

        # Either serve the images from a root directory or as a list of paths
        if from_dir:
            source = imroot
        else:
            if as_str:
                source = [str(img1), str(img2)]
            else:
                source = [img1, img2]

        dataset = CustomImageDataset(
            images_dir=source,
            chip_size=32,
            chip_stride=32,
            labels_dir=label_paths,
        )
        assert len(dataset) == 16

        output_paths = []
        for chip in dataset:
            assert "annotations" in chip["metadata"]
            assert isinstance(chip["metadata"]["annotations"], str)
            output_paths.append(chip["metadata"]["annotations"])

        # Multiple chips will have a single label file, so we can check sets - are
        # all of the input labels matched to some chip?
        assert set(output_paths) == set(map(str, label_paths))

    def test_dataset_padding_at_edge(self, tmp_path):
        img_path = create_image(tmp_path / "small.jpg", size=(40, 40), color="black")

        dataset = CustomImageDataset(
            images_dir=[str(img_path)],
            chip_size=32,
            chip_stride=30,
        )

        # Padding should turn this into 4 chips, 3 heavily padded
        assert len(dataset) == 4

        means = []
        for chip in dataset:
            assert chip["image"].shape[1:] == (
                32,
                32,
            ), "Tile should always match chip_size even with padding"
            means.append(chip["image"].mean())

        # As the image fills with white padding the average should go up
        assert np.allclose(sorted(means), [0, 0.7, 0.7, 0.9], atol=0.05)
