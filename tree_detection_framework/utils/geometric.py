from collections import defaultdict
from typing import List, Tuple

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from contourpy import contour_generator
from shapely import plotting


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


def ordered_voronoi(points, tolerance: float = 1e-6):
    """
    The shapely version we are using does not order the voronoi polygons in the same way as the
    input points. This wrapper does that.
    """
    voronoi = shapely.voronoi_polygons(points, tolerance=tolerance)

    ordered_voronoi = []

    for point in points.geoms:
        matching_polygon = list(
            filter(lambda poly: shapely.contains(poly, point), voronoi.geoms)
        )
        if len(matching_polygon) > 1:
            raise ValueError(f"Multiple Voronoi polygons contain the point {point}")
        # Take the matching element, or empty (point with no values) if numerical issues means there is no match
        matching_polygon = (
            matching_polygon[0]
            if len(matching_polygon) > 0
            else shapely.geometry.Point()
        )
        ordered_voronoi.append(matching_polygon)
    return shapely.GeometryCollection(ordered_voronoi)


def split_overlapping_region(
    poly1: shapely.Polygon,
    poly2: shapely.Polygon,
    epsilon: float = 1e-6,
    vis=False,
    segmentize_multiplier: float = 20.0,
) -> Tuple[shapely.Polygon, shapely.Polygon]:
    """
    Take two potentially-overlapping polygons and return the non-overlapping version of them such
    that any overlapping regions are split between the two. This is done by finding any overlapping
    regions and splitting them with a straight line that is equidistant to the two non-overlapping
    portions of the polygons.

    Args:
        poly1 (shapely.Polygon): First polygon
        poly2 (shapely.Polygon): Second polygon
        epsilon (float, optional): Used for numerical stability. Defaults to 1e-6.
        vis (bool, optional): Whether to show intermediate results. Defaults to False.
        segmentize_multiplier (float, optional):
            Use roughly this many points per dimension when calling "segmentize" to densify the polygon
            boundary. Defaults to 20.0.

    Returns:
        Tuple[shapely.Polygon, shapely.Polygon]: The non-overlapping versions of the input polygons
    """
    # Compute the intersection between the two polygons
    intersection = shapely.intersection(poly1, poly2)
    intersection = intersection.buffer(epsilon)

    # If there is no intersection, return the initial regions unchanged
    if intersection.area == 0:
        return (poly1, poly2)

    # Subtract the (slightly buffered) intersection from both input polygons
    p1_min_inter = poly1.difference(intersection)
    p2_min_inter = poly2.difference(intersection)

    # Compute the scale of the objects to understand how densely to sample
    p1_size = max(
        p1_min_inter.bounds[2] - p1_min_inter.bounds[0],
        p1_min_inter.bounds[3] - p1_min_inter.bounds[1],
    )
    p2_size = max(
        p2_min_inter.bounds[2] - p2_min_inter.bounds[0],
        p2_min_inter.bounds[3] - p2_min_inter.bounds[1],
    )
    # Get the densified boundary points, dropping the duplicated start/end point
    boundary1 = list(
        shapely.segmentize(
            p1_min_inter.exterior, p1_size / segmentize_multiplier
        ).coords
    )[:-1]
    boundary2 = list(
        shapely.segmentize(
            p2_min_inter.exterior, p2_size / segmentize_multiplier
        ).coords
    )[:-1]

    # Compute IDs identifying which polygon each vertex corresponds to
    vert_IDs = np.concatenate(
        [np.zeros(len(boundary1), dtype=int), np.ones(len(boundary2), dtype=int)]
    )

    # Create a multipoint with the vertices from both polygons
    all_verts = shapely.MultiPoint(boundary1 + boundary2)

    # Compute the voronoi tesselation with the polygons ordered consistently with the input verts
    voronoi = ordered_voronoi(all_verts)

    voronoi_gdf = gpd.GeoDataFrame(data={"IDs": vert_IDs}, geometry=list(voronoi.geoms))

    voronoi_gdf.geometry = voronoi_gdf.make_valid()
    merged = voronoi_gdf.dissolve("IDs")
    merged["IDs"] = merged.index
    if vis:
        ax = merged.plot("IDs")
        ax.set_title("Voronoi tessellation colored by original polygon")
        plt.show()

    poly1_merged = merged.iloc[0].geometry
    poly2_merged = merged.iloc[1].geometry

    poly1_clipped = poly1.intersection(poly1_merged)
    poly2_clipped = poly2.intersection(poly2_merged)

    # Remove extranous points by simplifying
    poly1_clipped = shapely.make_valid(shapely.simplify(poly1_clipped, epsilon))
    poly2_clipped = shapely.make_valid(shapely.simplify(poly2_clipped, epsilon))

    if vis:
        _, ax = plt.subplots()
        plotting.plot_polygon(poly1_clipped, ax=ax, color="b")
        plotting.plot_polygon(poly2_clipped, ax=ax, color="r")
        plt.title("Non-overlapping regions")
        plt.show()

    return poly1_clipped, poly2_clipped


def make_polygon_set_nonoverlapping(
    polygons: List[shapely.geometry.polygon.Polygon], epsilon: float = 1e-6, vis=False
) -> List[shapely.geometry.polygon.Polygon]:
    """
    Take a set of polygons and return an updated set representing the "core" nonoverlapping regions
    such that each core region represents the space which is farthest interior to that polygon,
    compared to all other polygons.

    Args:
        polygons (List[shapely.geometry.polygon.Polygon]):
            A list of polygons which may have pairwise overlaps. Note, this approach may fail if
            two polygons are identical or a strict subset of another one.
        epsilon (float, optional): Used for numerical stability. Defaults to 1e-6.
        vis (bool, optional): Show the pairwise compuations. Defaults to False.

    Raises:
        NotImplementedError: If there are multiple polygons in an intersecting region.

    Returns:
        List[shapely.geometry.polygon.Polygon]: The core regions, ordered the same way as the input
    """
    # If there are zero or one polygons, just return that unchanged
    if len(polygons) <= 1:
        return polygons

    nonoverlapping_regions = defaultdict(list)

    for i, first_poly in enumerate(polygons):
        for j, second_poly in enumerate(polygons[:i]):
            first_poly_nonoverlapping, second_poly_nonoverlapping = (
                split_overlapping_region(
                    first_poly, second_poly, epsilon=epsilon, vis=vis
                )
            )
            nonoverlapping_regions[i].append(first_poly_nonoverlapping)
            nonoverlapping_regions[j].append(second_poly_nonoverlapping)

    output_polygons = []

    # Iterate over the keys in order to ensure output order matches input order
    for i in sorted(nonoverlapping_regions.keys()):
        all_polys = nonoverlapping_regions[i]
        intersection = shapely.intersection_all(all_polys)

        if isinstance(intersection, shapely.geometry.GeometryCollection):
            poly_geoms = [
                geom for geom in intersection.geoms if isinstance(geom, shapely.Polygon)
            ]
            if len(poly_geoms) != 1:
                raise NotImplementedError("Can only handle one overlapping polygon")
            intersection = poly_geoms[0]

        # There are examples of weird internal lines. This removes them.
        intersection = intersection.buffer(epsilon).buffer(-epsilon)

        output_polygons.append(intersection)

    return output_polygons
