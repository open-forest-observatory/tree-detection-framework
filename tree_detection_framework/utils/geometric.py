import cv2
import numpy as np
import shapely
from contourpy import contour_generator
from shapely import plotting
import matplotlib.pyplot as plt
import geopandas as gpd


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


def ordered_voronoi(points):
    gpd.GeoDataFrame(geometry=[points]).plot()
    voronoi = shapely.voronoi_polygons(points)

    ordered_voronoi = []
    for point in points.geoms:
        matching_polygon = list(
            filter(lambda poly: shapely.contains(poly, point), voronoi.geoms)
        )[0]
        ordered_voronoi.append(matching_polygon)
    return shapely.GeometryCollection(ordered_voronoi)


def split_overlapping_region(
    poly1: shapely.Polygon, poly2: shapely.Polygon, epsilon: float = 1e-6
):
    # Compute the intersection between the two polygons
    intersection = shapely.intersection(poly1, poly2)
    intersection = intersection.buffer(epsilon)

    # If there is no intersection, return the initial regions unchanged
    if intersection.area == 0:
        return (poly1, poly2)

    # Subtract the (slightly buffered) intersection from both input polygons
    p1_min_inter = poly1.difference(intersection)
    p2_min_inter = poly2.difference(intersection)

    # Get the boundary points, dropping the duplicated start/end point
    boundary1 = list(shapely.segmentize(p1_min_inter.exterior, 0.05).coords)[:-1]
    boundary2 = list(shapely.segmentize(p2_min_inter.exterior, 0.05).coords)[:-1]

    # Compute IDs identifying which polygon each vertex corresponds to
    vert_IDs = np.concatenate(
        [np.zeros(len(boundary1), dtype=int), np.ones(len(boundary2), dtype=int)]
    )

    # Create a multipoint with the vertices from both polygons
    all_verts = shapely.MultiPoint(boundary1 + boundary2)

    # Compute the voronoi tesselation with the polygons ordered consistently with the input verts
    voronoi = ordered_voronoi(all_verts)

    voronoi_gdf = gpd.GeoDataFrame(data={"IDs": vert_IDs}, geometry=list(voronoi.geoms))

    merged = voronoi_gdf.dissolve("IDs")
    merged["IDs"] = merged.index
    merged.plot("IDs")

    poly1_merged = merged.iloc[0, :].geometry
    poly2_merged = merged.iloc[1, :].geometry

    poly1_clipped = poly1.intersection(poly1_merged)
    poly2_clipped = poly2.intersection(poly2_merged)

    f, ax = plt.subplots()
    plotting.plot_polygon(poly1_clipped, ax=ax, color="b")
    plotting.plot_polygon(poly2_clipped, ax=ax, color="r")
    plt.show()
