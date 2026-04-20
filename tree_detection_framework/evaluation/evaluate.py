import logging
import warnings
from typing import Callable, List, Literal, Optional, Tuple, Union

import geopandas as gpd
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely
from rasterio.mask import mask
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from shapely.affinity import scale
from shapely.geometry import Point, Polygon, box

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.detection.region_detections import (
    RegionDetections,
    RegionDetectionsSet,
    reproject_detections,
)
from tree_detection_framework.utils.geospatial import get_projected_CRS
from tree_detection_framework.utils.raster import get_heights_from_chm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def polygons_to_points(
    detections: RegionDetections | RegionDetectionsSet | gpd.GeoDataFrame,
    method: Literal["centroid", "chm_max"],
    chm_path: Optional[str] = None,
    height_column: str = "height",
    crown_geometry_column: str = "crown_geometry",
    erosion_distance: float = 0.0,
) -> gpd.GeoDataFrame:
    """Convert polygon geometries to point geometries using the specified method.
    The original polygon geometries are stored in a new column, and height values
    are populated when CHM sampling is performed.

    Args:
        detections: GeoDataFrame, RegionDetections, or RegionDetectionsSet with
            polygon geometries to convert as points.
        method: Method to use to derive the representative point from each polygon.
            - "centroid": Use the centroid of each polygon. If `chm_path` is
              provided, the CHM is sampled at each centroid to populate
              `height_column`.
            - "chm_max": Requires `chm_path`. The point is set to the
              location of the highest CHM pixel within each polygon, and
              `height_column` is populated with that pixel's value. Falls back
              to centroid with a warning for polygons whose CHM coverage is
              entirely nodata.
        chm_path: Path to a canopy height model (CHM) raster. Required when
            `method="chm_max"`. Optional for `method="centroid"`.
        height_column: Name of the column to write sampled height values into.
            Raises a ValueError if this column already exists in `detections`.
            Defaults to "height".
        crown_geometry_column: Name of the column in which the original polygon
            geometries are stored after points are derived and set as the default
            geometry column. Defaults to "crown_geometry".
        erosion_distance: Distance (in CHM CRS units) to erode each polygon inward
            before finding the maximum CHM pixel. Only used when `method="chm_max"`.
            Defaults to 0.0 (no erosion).
    Returns:
        gpd.GeoDataFrame: A copy of detections with -
        - Active geometry replaced with point geometries (Point).
        - Original polygon geometries stored in `crown_geometry_column`.
        - `height_column` populated when CHM sampling is performed.
        - All other columns preserved unchanged.
    """
    # Convert input to GeoDataFrame if needed
    if isinstance(detections, RegionDetectionsSet):
        detections = detections.merge().get_data_frame()
    elif isinstance(detections, RegionDetections):
        detections = detections.get_data_frame()

    # Check if height column already exists
    if height_column in detections.columns:
        raise ValueError(
            f"Column '{height_column}' already exists in the GeoDataFrame. "
            "Rename or drop it before calling polygons_to_points to avoid "
            "overwriting existing height values."
        )

    if method == "chm_max" and chm_path is None:
        raise ValueError("chm_path must be provided when method='chm_max'.")

    # Work on a copy of the detections gdf
    result = detections.copy()
    # Store original polygon geometries in a new column before overwriting the active geometry with points
    result[crown_geometry_column] = result.geometry

    # For box polygons (only DeepForest so far), replace geometry with the largest inscribed circle or ellipse before deriving points
    first_geom = result.geometry.iloc[0]
    # check if geometry is a box polygon (i.e. it is equal to its own bounding box)
    if first_geom.equals(first_geom.envelope):
        result["geometry"] = result.geometry.apply(_inscribe_circle_or_ellipse)

    # Derive point geometries
    if method == "centroid":
        result["geometry"] = result.geometry.centroid

        if chm_path is not None:
            logging.info("Sampling CHM at centroid locations for height values.")
            coords = np.column_stack([result.geometry.x, result.geometry.y])
            result[height_column] = get_heights_from_chm(coords, result.crs, chm_path)

    elif method == "chm_max":
        logging.info("Finding tallest CHM pixel within each polygon.")
        points, heights = _chm_max_points(
            result, chm_path, erosion_distance=erosion_distance
        )
        result["geometry"] = points
        result[height_column] = heights
        result.attrs["crs"] = result.crs.to_string()

    return result


def compute_matched_ious(
    ground_truth_boxes: List[Polygon], predicted_boxes: List[Polygon]
) -> List:
    """Compute IoUs for matched pairs of ground truth and predicted boxes.
    This uses the Hungarian algorithm to find the optimal assignment.
    Args:
        ground_truth_boxes (list): List of ground truth polygons.
        predicted_boxes (list): List of predicted polygons.
    Returns:
        list: List of IoUs for matched pairs.
    """
    if not ground_truth_boxes or not predicted_boxes:
        return 0.0  # Return 0 if either list is empty

    # Create GeoDataFrames for ground truth and predicted boxes
    gt_gdf = gpd.GeoDataFrame(geometry=ground_truth_boxes)
    gt_gdf["area_gt"] = gt_gdf.geometry.area
    gt_gdf["id_gt"] = gt_gdf.index

    pred_gdf = gpd.GeoDataFrame(geometry=predicted_boxes)
    pred_gdf["area_pred"] = pred_gdf.geometry.area
    pred_gdf["id_pred"] = pred_gdf.index

    # Get the intersection between the two sets of polygons
    intersection = gpd.overlay(gt_gdf, pred_gdf, how="intersection")
    intersection["iou"] = intersection.area / (
        intersection["area_gt"] + (intersection["area_pred"] - intersection.area)
    )

    # Create a cost matrix to store IoUs
    cost_matrix = np.zeros((len(gt_gdf), len(pred_gdf)))
    cost_matrix[intersection["id_gt"], intersection["id_pred"]] = -intersection["iou"]

    # Solve optimal assignment using the Hungarian algorithm
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    # Extract IoUs of matched pairs
    matched_ious = [-cost_matrix[i, j] for i, j in zip(gt_indices, pred_indices)]
    return matched_ious


def compute_precision_recall(
    ious: List, num_gt: int, num_pd: int, threshold: float = 0.4
) -> tuple:
    """Compute precision and recall based on IoUs.
    Args:
        ious (list): List of IoUs for matched pairs.
        num_gt (int): Number of ground truth boxes.
        num_pd (int): Number of predicted boxes.
        threshold (float): IoU threshold for considering a match.
    Returns:
        tuple: Precision and recall values.
    """
    true_positives = (np.array(ious) > threshold).astype(np.uint8)
    tp = np.sum(true_positives)
    recall = tp / num_gt if num_gt > 0 else 0.0
    precision = tp / num_pd if num_pd > 0 else 0.0
    return precision, recall


def _fill_in_heights(
    df,
    coords,
    height_column=None,
    fillin_method=None,
    chm_path=None,
    bboxes=None,
):
    """Extracts or computes height arrays for a set of coordinates"""
    if height_column is not None:
        return df[height_column].values

    if fillin_method == "chm":
        logging.info("Extracting treetop heights from CHM")
        if chm_path is None:
            raise ValueError("CHM path must be provided when fillin_method is 'chm'.")
        return get_heights_from_chm(coords, df.crs, chm_path)

    elif fillin_method == "bbox":
        # TODO: Decide logic to compute height values from bounding boxes
        raise NotImplementedError()

    else:
        raise ValueError(
            "Please provide values for 'height1' and 'height2' "
            "or a 'fillin_method' to derive heights from an alternative source."
        )


def _inscribe_circle_or_ellipse(polygon: Polygon) -> Polygon:
    """Return the largest inscribed circle (square box) or ellipse (rectangular box)."""
    minx, miny, maxx, maxy = polygon.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2  # center of the box
    rx, ry = (maxx - minx) / 2, (maxy - miny) / 2  # rx = width/2, ry = height/2
    # scale() stretches a unit circle along each axis producing an ellipse that touches all sides
    return scale(Point(cx, cy).buffer(1), xfact=rx, yfact=ry)


def _chm_max_points(
    gdf: gpd.GeoDataFrame,
    chm_path: str,
    min_valid_fraction: float = 0.5,
    erosion_distance: float = 0.0,
) -> tuple[list, np.ndarray]:
    """Find the highest CHM pixel within each polygon.

    For polygons whose CHM coverage is entirely nodata, a warning is given
    and the centroid is used as a fallback point with np.nan as height.

    Args:
        gdf: GeoDataFrame with polygon geometries.
        chm_path: Path to the CHM raster.
        min_valid_fraction: Minimum fraction of valid (non-nodata) CHM pixels required
            within a polygon to compute the maximum height. Polygons below this threshold
            fall back to centroid.
        erosion_distance: Distance (in CRS units of the CHM) to erode each polygon
            inward before finding the maximum CHM pixel. Helps exclude edge pixels
            just outside the tree crown. If erosion collapses a polygon to empty
            geometry, the original polygon is used as a fallback.

    Returns:
        tuple(list, np.ndarray):
        - List of Point geometries (one per polygon), in the CRS of `gdf`.
        - np.ndarray of height values (np.nan for nodata fallbacks).
    """
    points = []

    # Initialize heights array with NaNs; will populate valid heights and leave NaN for
    # polygons with nodata coverage or other issues. This also ensures the array has the correct length.
    heights = np.full(len(gdf), np.nan, dtype=float)

    with rasterio.open(chm_path) as src:
        # Work in CHM CRS, then reproject points back to original CRS at the end
        polys_in_chm_crs = gdf if gdf.crs == src.crs else gdf.to_crs(src.crs)
        nodata = src.nodata

        for i, (_, row) in enumerate(polys_in_chm_crs.iterrows()):
            polygon = row.geometry

            if erosion_distance > 0:
                eroded = polygon.buffer(-erosion_distance)
                if not eroded.is_empty:
                    # polygon is reassigned only if eroded result is non-empty
                    # so a collapsed erosion silently falls back to original.
                    polygon = eroded

            try:
                chm_window, window_transform = mask(
                    src, [polygon], crop=True, nodata=np.nan, filled=True
                )
            except Exception as e:
                warnings.warn(
                    f"Could not mask CHM for polygon at index {i}: {e}. "
                    "Falling back to centroid."
                )
                points.append(polygon.centroid)
                continue

            data = chm_window[0]  # single band

            # Replace nodata with nan
            if nodata is not None:
                data = data.astype(float)
                data[data == nodata] = np.nan

            valid_mask = ~np.isnan(data)
            valid_fraction = np.sum(valid_mask) / data.size

            if valid_fraction == 0:
                warnings.warn(
                    f"Polygon at index {i} has entirely nodata CHM coverage. "
                    "Falling back to centroid with height=nan."
                )
                points.append(polygon.centroid)
                continue

            if valid_fraction < min_valid_fraction:
                warnings.warn(
                    f"Polygon at index {i} has only {valid_fraction:.2f} valid CHM pixels "
                    "Falling back to centroid."
                )
                points.append(polygon.centroid)
                continue

            # Find the pixel with the maximum CHM value
            flat_idx = np.nanargmax(data)
            row_idx, col_idx = np.unravel_index(flat_idx, data.shape)
            heights[i] = float(data[row_idx, col_idx])

            # Convert pixel coordinates back to spatial coordinates
            x, y = rasterio.transform.xy(window_transform, row_idx, col_idx)
            points.append(Point(x, y))

    # Reproject points back to original GDF CRS if CHM CRS differed
    if gdf.crs != polys_in_chm_crs.crs:
        pts_gdf = gpd.GeoDataFrame(geometry=points, crs=polys_in_chm_crs.crs)
        pts_gdf = pts_gdf.to_crs(gdf.crs)
        points = list(pts_gdf.geometry)

    return points, heights


def _visualize_points(coords1, coords2, matches, mode=3, buffer=5, polygons2=None):
    """Visualize matched points between two coordinate sets.
    Args:
        coords1 (np.ndarray): point coordinates from set 1
        coords2 (np.ndarray): point coordinates from set 2
        matches (List[tuple]): indices from set 1, set 2, and height/distance data for the matches
        mode (int): Ways to visualize the points
            - 1: Show only matched points
            - 2: Show all points from both sets, highlighting matches
            - 3: Same as mode 2, but crop the plot to the smaller set's bounds + buffer
        buffer (int): Used in mode 3 while cropping the smaller set's bounds
        polygons2 (gpd.GeoSeries, optional): Crown polygon geometries for set 2. If provided,
            polygon outlines are drawn underneath the scatter points.
    """
    _, ax = plt.subplots()

    # Draw crown polygons underneath points if provided
    if polygons2 is not None:
        polygons2.plot(
            ax=ax,
            facecolor="lightblue",
            edgecolor="steelblue",
            linewidth=0.5,
            alpha=0.3,
        )

    # Matched coordinates
    matched_coords1 = np.array([coords1[i1] for (i1, _, _) in matches])
    matched_coords2 = np.array([coords2[i2] for (_, i2, _) in matches])

    if mode == 1:
        ax.scatter(
            matched_coords1[:, 0],
            matched_coords1[:, 1],
            color="red",
            s=30,
            label="Set 1 (matched)",
        )
        ax.scatter(
            matched_coords2[:, 0],
            matched_coords2[:, 1],
            color="blue",
            s=30,
            label="Set 2 (matched)",
        )

    elif mode in (2, 3):
        ax.scatter(
            coords1[:, 0],
            coords1[:, 1],
            color="lightcoral",
            s=20,
            alpha=0.5,
            label="Set 1 (all)",
        )
        ax.scatter(
            coords2[:, 0],
            coords2[:, 1],
            color="lightblue",
            s=20,
            alpha=0.5,
            label="Set 2 (all)",
        )
        ax.scatter(
            matched_coords1[:, 0],
            matched_coords1[:, 1],
            color="red",
            s=30,
            label="Set 1 (matched)",
        )
        ax.scatter(
            matched_coords2[:, 0],
            matched_coords2[:, 1],
            color="blue",
            s=30,
            label="Set 2 (matched)",
        )

        if mode == 3:
            # Determine bounds from smaller set
            bounds1 = [
                coords1[:, 0].min(),
                coords1[:, 0].max(),
                coords1[:, 1].min(),
                coords1[:, 1].max(),
            ]
            bounds2 = [
                coords2[:, 0].min(),
                coords2[:, 0].max(),
                coords2[:, 1].min(),
                coords2[:, 1].max(),
            ]
            size1 = (bounds1[1] - bounds1[0]) * (bounds1[3] - bounds1[2])
            size2 = (bounds2[1] - bounds2[0]) * (bounds2[3] - bounds2[2])

            if size1 < size2:
                ref_bounds = bounds1
            else:
                ref_bounds = bounds2

            ax.set_xlim(ref_bounds[0] - buffer, ref_bounds[1] + buffer)
            ax.set_ylim(ref_bounds[2] - buffer, ref_bounds[3] + buffer)

    # Draw match lines
    for i1, i2, _ in matches:
        ax.plot(
            [coords1[i1, 0], coords2[i2, 0]],
            [coords1[i1, 1], coords2[i2, 1]],
            color="black",
            linestyle="-",
            linewidth=0.5,
            alpha=0.5,
        )

    ax.legend()
    ax.set_title("Matched Points")
    plt.show()


def match_points(
    treetop_set_1: RegionDetections | RegionDetectionsSet | gpd.GeoDataFrame,
    treetop_set_2: RegionDetections | RegionDetectionsSet | gpd.GeoDataFrame,
    height_column_1: Optional[str] = None,
    height_column_2: Optional[str] = None,
    chm_path: Optional[PATH_TYPE] = None,
    bboxes: Optional[RegionDetectionsSet] = None,
    height_threshold: Union[float, Callable[[float], float]] = lambda h: 0.5 * h,
    distance_threshold: Union[float, Callable[[float], float]] = lambda h: 0.1 * h + 1,
    fillin_method: Optional[str] = None,
    use_height_in_distance: Optional[float] = 0.0,
    vis: bool = False,
    vis_mode: int = 3,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Matches treetop detections from two datasets based on spatial proximity
    and tree height similarity.

    This function attempts to pair each treetop in `treetop_set_1` with at most
    one treetop in `treetop_set_2` such that:
      - Horizontal distance is below a maximum threshold.
      - Height difference is within an allowed range.
    Matching is greedy, preferring the closest pairs first.

    Args:
        treetop_set_1 : RegionDetections | RegionDetectionsSet | GeoDataFrame
            The reference (ground truth) treetop detections.
        treetop_set_2 : RegionDetections | RegionDetectionsSet | GeoDataFrame
            The predicted treetop detections to be matched against `treetop_set_1`.
        height_column_1 : str, optional
            Column name in `treetop_set_1` containing tree heights. If this is None and
            `fillin_method` is not provided, height will not be used.
        height_column_2 : str, optional
            Column name in `treetop_set_2` containing tree heights. If this is None and
            `fillin_method` is not provided, height will not be used.
        chm_path : PATH_TYPE, optional
            Path to a canopy height model (CHM) raster, used when `fillin_method='chm'`.
        bboxes : RegionDetectionsSet, optional
            Bounding boxes for computing height values when `fillin_method='bbox'`.
        height_threshold : float or callable
            Max allowed height difference during matching.
            - If float provided, it is considered a constant +/- tolerance in meters. Pass in
              np.inf to disable
            - If callable provided, it should accept a heights array and return the tolerance for
              each point. Default allows +/-50% the reference treetop's height
        distance_threshold : float or callable, optional
            Max allowed horizontal distance for a match in meters.
            - If float provided, it is a constant value. Pass in np.inf to disable
            - If callable is provided, it should accept a heights array and return distance
              thresholds.
        fillin_method : ['chm', 'bbox'], optional
            Method to fill in height values if they are not provided via `height1`
            and `height2`:
            - 'chm' : Sample from CHM raster at each treetop location.
            - 'bbox': Compute height from bounding boxes.
        use_height_in_distance : float, optional
            Weight to scale height difference when computing combined distance metric for sorting matches.
            If 0, height is not included in the sorting distance, but still used for validity checks. Defaults to 0.
        vis : bool, default=False
            If True, plot the matched treetop points and their connecting lines.
        vis_mode : int, default = 3
            - 1: Show only matched points
            - 2: Show all points from both sets, highlighting matches
            - 3: Same as mode 2, but crop the plot to the smaller set's bounds + buffer
    """
    if isinstance(treetop_set_1, gpd.GeoDataFrame):
        treetop_set_1 = RegionDetections(
            detection_geometries=None, data=treetop_set_1, CRS=treetop_set_1.crs
        )
    if isinstance(treetop_set_2, gpd.GeoDataFrame):
        treetop_set_2 = RegionDetections(
            detection_geometries=None, data=treetop_set_2, CRS=treetop_set_2.crs
        )
    if isinstance(treetop_set_1, RegionDetectionsSet):
        treetop_set_1 = treetop_set_1.merge()
    if isinstance(treetop_set_2, RegionDetectionsSet):
        treetop_set_2 = treetop_set_2.merge()

    # Ensure both sets have the same CRS
    if treetop_set_1.get_CRS() != treetop_set_2.get_CRS():
        # Convert set 2 detections to the CRS of set 1
        treetop_set_2 = reproject_detections(treetop_set_2, treetop_set_1.get_CRS())

    if treetop_set_1.detections.crs.is_geographic:
        lat = treetop_set_1.get_bounds()[0].bounds[1]
        lon = treetop_set_1.get_bounds()[0].bounds[0]
        projected_crs = get_projected_CRS(lat, lon)
        treetop_set_1 = reproject_detections(treetop_set_1, projected_crs)
        treetop_set_2 = reproject_detections(treetop_set_2, projected_crs)

    # Extract coordinates
    df1 = treetop_set_1.get_data_frame()
    df2 = treetop_set_2.get_data_frame()
    coords1 = shapely.get_coordinates(df1.geometry)
    coords2 = shapely.get_coordinates(df2.geometry)

    # If height values are not provided, the algorithm only uses distance to find the matches
    ignore_height = False
    if (height_column_1 is None) and (fillin_method is None):
        logging.info("Not using height values for matching points.")
        ignore_height = True

    if not ignore_height:
        # Extract height values using dataframe column or from CHM if provided
        height_vals_1 = _fill_in_heights(
            df1, coords1, height_column_1, fillin_method, chm_path, bboxes
        )
        height_vals_2 = _fill_in_heights(
            df2, coords2, height_column_2, fillin_method, chm_path, bboxes
        )

    else:
        height_vals_1, height_vals_2 = None, None

    # Compute XY distance matrix for validity checks
    distance_matrix_xy = cdist(coords1, coords2)  # (N1, N2)

    # Compute combined distance matrix for sorting if use_height_in_distance is set
    if not ignore_height:
        (
            logging.info(
                "Using height as an additional scaled dimension to compute distance"
            )
            if use_height_in_distance > 0
            else None
        )
        # Note: if `use_height_in_distance` is zero, height has no effect on the distance
        # calculation and the result is identical to pure XY distance.
        coords1_aug = np.hstack(
            [coords1, use_height_in_distance * height_vals_1.reshape(-1, 1)]
        )
        coords2_aug = np.hstack(
            [coords2, use_height_in_distance * height_vals_2.reshape(-1, 1)]
        )
        distance_matrix = cdist(coords1_aug, coords2_aug)  # combined XY + scaled height
        height_vals_1 = np.expand_dims(np.array(height_vals_1), axis=1)  # (N1, 1)
        height_vals_2 = np.expand_dims(np.array(height_vals_2), axis=0)  # (1, N2)
    else:
        distance_matrix = distance_matrix_xy  # just XY distance

    # Build valid pairs mask (based on XY distance and height difference thresholds)
    if ignore_height:
        # Only use constant distance thresholds if heights are not available
        if callable(distance_threshold):
            raise ValueError("Provide a constant value for `distance_threshold`.")
        valid_pairs_mask = distance_matrix < distance_threshold
    else:
        # Height bounds
        if callable(height_threshold):
            min_h = height_vals_1 - height_threshold(height_vals_1)
            max_h = height_vals_1 + height_threshold(height_vals_1)
        else:
            min_h = height_vals_1 - height_threshold
            max_h = height_vals_1 + height_threshold

        # Distance bounds
        if callable(distance_threshold):
            max_d = distance_threshold(height_vals_1)
        else:
            max_d = distance_threshold

        valid_pairs_mask = np.logical_and.reduce(
            [height_vals_2 > min_h, height_vals_2 < max_h, distance_matrix < max_d]
        )

    # Extract valid pair indices and sort by distance
    valid_idxs_1, valid_idxs_2 = np.where(valid_pairs_mask)
    distances = distance_matrix[valid_idxs_1, valid_idxs_2]

    # Sort matches by combined distance metric (which includes height if use_height_in_distance is set)
    sorted_idx = np.argsort(distances)
    valid_idxs_1 = valid_idxs_1[sorted_idx]
    valid_idxs_2 = valid_idxs_2[sorted_idx]
    matching_distance = distances[sorted_idx]

    # Greedy matching
    max_valid_matches = min(distance_matrix.shape)
    matched_1 = []
    matched_2 = []
    matches = []

    for i1, i2, d_h in zip(valid_idxs_1, valid_idxs_2, matching_distance):
        if i1 not in matched_1 and i2 not in matched_2:
            matches.append((i1, i2, d_h))
            matched_1.append(i1)
            matched_2.append(i2)

        if len(matched_1) == max_valid_matches:
            break

    if vis:
        polygons2 = df2["crown_geometry"] if "crown_geometry" in df2.columns else None
        _visualize_points(coords1, coords2, matches, mode=vis_mode, polygons2=polygons2)

    return matches


def assess_matches(matches: List, n_ground_truth: int, n_predictions: int):
    """
    Calculate precison, recall and F1 score to evaluate the matches
    Args:
        matches (List): List of matched pairs
        n_ground_truth (int): Number of treetops in ground truth (treetop_set_1)
        n_predictions (int): Number of treetops in predictions (treetop_set_2)

    Returns:
        precision, recall, f1_score: floats
    """
    n_matches = len(matches)
    # Precison: how many predicted treetops matched a ground truth trees
    precision = n_matches / n_predictions if n_predictions > 0 else 0
    # Recall: how many ground truth treetops got matched
    recall = n_matches / n_ground_truth if n_ground_truth > 0 else 0

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score
