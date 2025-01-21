import cv2
import numpy as np
import shapely


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
    mask: np.ndarray, simplify_tolerance: float = 0
) -> shapely.MultiPolygon:
    """
    Convert a binary mask to a Shapely MultiPolygon representing positive regions,
    with optional simplification.

    Args:
        mask (np.ndarray): A (n, m) array where positive values are > 0.5 and negative values are < 0.5.
        simplify_tolerance (float): Tolerance for simplifying polygons. A value of 0 means no simplification.

    Returns:
        shapely.MultiPolygon: A MultiPolygon representing the positive regions.
    """
    if not np.any(mask):
        return shapely.Polygon()  # Return an empty Polygon if the mask is empty.

    # Find contours in the mask
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

        # Simplify polygons if tolerance value provided
        if simplify_tolerance > 0:
            shape = shape.simplify(simplify_tolerance)

        if isinstance(shape, shapely.MultiPolygon):
            # Append all individual polygons
            polygons.extend(shape.geoms)
        elif isinstance(shape, shapely.Polygon):
            # Append the polygon
            polygons.append(shape)

    # Combine all polygons into a MultiPolygon
    multipolygon = shapely.MultiPolygon(polygons)

    return multipolygon
