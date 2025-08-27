from typing import Tuple

import cv2
import numpy as np
import shapely
from contourpy import contour_generator


def get_shapely_transform_from_matrix(matrix_transform: np.ndarray):
    """
    Take a matrix transform and convert it into format expected by shapely: [a, b, d, e, xoff, y_off]

    Args:
        matrix_transform (np.ndarray):
            (2, 3) or (3, 3) 2D transformation matrix such that the matrix-vector product produces
            the transformed value.
    """
    shapely_transform = [
        matrix_transform[0, 0],
        matrix_transform[0, 1],
        matrix_transform[1, 0],
        matrix_transform[1, 1],
        matrix_transform[0, 2],
        matrix_transform[1, 2],
    ]
    return shapely_transform


def mask_to_shapely(
    mask: np.ndarray, simplify_tolerance: float = 0, backend: str = "contourpy"
) -> shapely.MultiPolygon:
    """
    Convert a binary mask to a Shapely MultiPolygon representing positive regions,
    with optional simplification.

    Args:
        mask (np.ndarray): A (n, m) A mask with boolean values.
        simplify_tolerance (float): Tolerance for simplifying polygons. A value of 0 means no simplification.
        backend (str): The backend to use for contour extraction. Choose from "cv2" and "contourpy". Defaults to contourpy.

    Returns:
        shapely.MultiPolygon: A MultiPolygon representing the positive regions.
    """
    if not np.any(mask):
        return shapely.Polygon()  # Return an empty Polygon if the mask is empty.

    if backend == "cv2":
        # CV2-based approach
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        for contour in contours:
            contour = np.squeeze(contour)
            # Skip invalid contours
            if (contour.ndim != 2) or (contour.shape[0] < 3):
                continue

            # Convert the contour to a shapely geometry
            shape = shapely.Polygon(contour)

            if isinstance(shape, shapely.MultiPolygon):
                # Append all individual polygons
                polygons.extend(shape.geoms)
            elif isinstance(shape, shapely.Polygon):
                # Append the polygon
                polygons.append(shape)

        # Combine all polygons into a MultiPolygon
        multipolygon = shapely.MultiPolygon(polygons)

        if simplify_tolerance > 0:
            multipolygon = multipolygon.simplify(simplify_tolerance)

        return multipolygon

    elif backend == "contourpy":
        # ContourPy-based approach
        filled = contour_generator(
            z=mask, fill_type="ChunkCombinedOffsetOffset"
        ).filled(0.5, np.inf)
        chunk_polygons = [
            shapely.from_ragged_array(
                shapely.GeometryType.POLYGON, points, (offsets, outer_offsets)
            )
            for points, offsets, outer_offsets in zip(*filled)
        ]

        multipolygon = shapely.unary_union(chunk_polygons)

        # Simplify the resulting MultiPolygon if needed
        if simplify_tolerance > 0:
            multipolygon = multipolygon.simplify(simplify_tolerance)

        return multipolygon

    else:
        raise ValueError(
            f"Unsupported backend: {backend}. Choose 'cv2' or 'contourpy'."
        )


def ellipse_mask(
    image_shape: Tuple[int, int],
    center: Tuple[int, int],
    axes: Tuple[int, int],
    angle_rad: float = 0.0,
):
    """Return a boolean mask with True inside the specified ellipse.

    Args:
        image_shape (Tuple[int, int]):
            (Height, width) tuple for the shape of the mask we want to return
        center (Tuple[int, int]):
            (x0, y0) center of the ellipse in x and y (pixels)
        axes (Tuple[int, int]):
            (a, b) x and y axis lengths (before rotation) in pixels
        angle_rad (float):
            Rotation angle of the semi-major axis, in radians. CCW from x-axis
            (a.k.a. right-hand rule out of the image). Defaults to 0.

    Returns:
        mask: np.ndarray of shape (H, W), dtype=bool
    """
    H, W = image_shape
    y, x = np.ogrid[:H, :W]
    x0, y0 = center
    a, b = axes
    # We need to invert the angle because images have Y pointing down
    # (left-hand reference frame) but for ease of use we are going to
    # specify the angle as if Y was up as the docstring describes.
    cos_t, sin_t = np.cos(-angle_rad), np.sin(-angle_rad)

    # Shift and rotate the coordinates
    x_shift = x - x0
    y_shift = y - y0

    x_rot = cos_t * x_shift + sin_t * y_shift
    y_rot = -sin_t * x_shift + cos_t * y_shift

    # Ellipse equation
    mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1
    return mask.astype(bool)
