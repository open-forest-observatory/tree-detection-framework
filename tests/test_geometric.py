import numpy as np

from tree_detection_framework.utils import ellipse_mask


class TestEllipseMask:
    def test_basic(self):

        # Create the ellipse
        shape = (40, 60)
        center = (10, 20)
        axes = (10, 3)
        mask = ellipse_mask(shape, center, axes)

        # Check types and shapes
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == shape

        # Check a few points in the mask
        assert mask[10, 20] is True
        # X
        assert mask[8, 20] is True
        assert mask[18, 20] is True
        assert mask[22, 20] is False
        # Y
        assert mask[10, 18] is True
        assert mask[10, 22] is True
        assert mask[10, 15] is False
        assert mask[10, 25] is False

    def test_angles(self):

        # Create the ellipse with two angles
        shape = (60, 60)
        center = (30, 30)
        axes = (300, 5)
        angle = np.deg2rad(45)
        mask_p45 = ellipse_mask(shape, center, axes, angle)
        mask_n45 = ellipse_mask(shape, center, axes, -angle)

        # Check the diagonals for +45
        assert mask_p45[5, 5] is False
        assert mask_p45[55, 55] is False
        assert mask_p45[30, 30] is True
        assert mask_p45[55, 5] is True
        assert mask_p45[5, 55] is True

        # Check the diagonals for -45
        assert mask_p45[5, 5] is True
        assert mask_p45[55, 55] is True
        assert mask_p45[30, 30] is True
        assert mask_p45[55, 5] is False
        assert mask_p45[5, 55] is False
