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


def mask_to_shapely(mask: np.ndarray) -> shapely.MultiPolygon:
    """Convert a binary mask to a shapely polygon representing the positive regions

    Args:
        mask (np.ndarary):
            A (n, m) array where positive values are > 0.5 and negative values are < 0.5

    Returns:
        shapely.MultiPolygon:
            A multipolygon representing the positive regions. Holes and multiple disconnected
            components are properly handled.
    """
    # This generally follows the example here:
    # https://contourpy.readthedocs.io/en/v1.3.0/user_guide/external/shapely.html#filled-contours-to-shapely

    # If mask is empty, return an empty Polygon
    if not np.any(mask):
        return shapely.Polygon()

    # Extract the contours and create a filled contour for the regions above 0.5
    filled = contour_generator(z=mask, fill_type="ChunkCombinedOffsetOffset").filled(
        0.5, np.inf
    )

    # Create a polygon for each of the disconnected regions, called chunks in ContourPy
    # This iterates over the elements in three lists, the points, offsets, and outer offsets
    chunk_polygons = [
        shapely.from_ragged_array(
            shapely.GeometryType.POLYGON, points, (offsets, outer_offsets)
        )
        for points, offsets, outer_offsets in zip(*filled)
    ]
    # Union these regions to get a single multipolygon
    multipolygon = shapely.unary_union(chunk_polygons)

    return multipolygon
