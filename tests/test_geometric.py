import numpy as np

from tree_detection_framework.utils.geometric import ellipse_mask


class TestEllipseMask:
    def test_basic(self):

        # Create the ellipse. Note that the center is defined in terms of
        # (x, y), but then we will need to index into the image with (y, x)
        # a.k.a. (i, j)
        shape = (40, 60)
        center = (10, 20)
        axes = (10, 3)
        mask = ellipse_mask(shape, center, axes)

        # Check types and shapes
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == shape

        # Check a few points in the mask
        assert bool(mask[20, 10]) is True
        # X
        assert bool(mask[20, 8]) is True
        assert bool(mask[20, 18]) is True
        assert bool(mask[20, 22]) is False
        # Y
        assert bool(mask[18, 10]) is True
        assert bool(mask[22, 10]) is True
        assert bool(mask[15, 10]) is False
        assert bool(mask[25, 10]) is False

    def test_angles(self):

        # Create the ellipse with two angles
        shape = (60, 60)
        center = (30, 30)
        axes = (300, 5)
        angle = np.deg2rad(45)
        mask_p45 = ellipse_mask(shape, center, axes, angle)
        mask_n45 = ellipse_mask(shape, center, axes, -angle)

        # Check the diagonals for +45
        assert bool(mask_p45[5, 5]) is False
        assert bool(mask_p45[55, 55]) is False
        assert bool(mask_p45[30, 30]) is True
        assert bool(mask_p45[55, 5]) is True
        assert bool(mask_p45[5, 55]) is True

        # Check the diagonals for -45
        assert bool(mask_n45[5, 5]) is True
        assert bool(mask_n45[55, 55]) is True
        assert bool(mask_n45[30, 30]) is True
        assert bool(mask_n45[55, 5]) is False
        assert bool(mask_n45[5, 55]) is False
