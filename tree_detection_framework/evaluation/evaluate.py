from typing import Callable, List, Optional, Tuple, Union
import logging

import geopandas as gpd
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import shapely
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon

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

def _prepare_heights(df1, df2, coords1, coords2, height1 = None, height2 = None, fillin_method = None, chm_path = None):
    """Extracts or computes height arrays for both sets."""
    if height1 is not None and height2 is not None:
        h1 = df1[height1].values
        h2 = df2[height2].values

    elif fillin_method == "chm":
        logging.info("Extracting treetop heights from CHM")
        if chm_path is None:
            raise ValueError("CHM path must be provided when fillin_method is 'chm'.")
        h1 = get_heights_from_chm(coords1, df1.crs, chm_path)
        h2 = get_heights_from_chm(coords2, df2.crs, chm_path)

    elif fillin_method == "bbox":
        # TODO: Decide logic to compute height values from bounding boxes
        raise NotImplementedError()
    
    else:
        raise ValueError(
            "Please provide values for 'height1' and 'height2' "
            "or a 'fillin_method' to sample values from CHM."
        )
    
    return np.array(h1), np.array(h2)

def _vis_matches(coords1, coords2, matches):
    """Visualize matched points"""
    # TODO: Support plotting all points
    
    _, ax = plt.subplots(figsize=(6, 6))

    # Extract matched coordinates only
    matched_coords1 = np.array([coords1[i1] for (i1, _, _) in matches])
    matched_coords2 = np.array([coords2[i2] for (_, i2, _) in matches])

    # Plot only matched points
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

    # Draw lines connecting matched pairs
    lines = [[coords1[i1], coords2[i2]] for (i1, i2, _) in matches]
    lc = mc.LineCollection(lines, colors="black", linewidths=0.8)
    ax.add_collection(lc)

    ax.set_aspect("equal", adjustable="datalim")
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
    use_height_in_distance: Optional[float] = 0,
    vis: bool = False,
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
            Column name in `treetop_set_1` containing tree heights. Required unless
            `fillin_method` is provided.
        height_column_2 : str, optional
            Column name in `treetop_set_2` containing tree heights. Required unless
            `fillin_method` is provided.
        chm_path : PATH_TYPE, optional
            Path to a canopy height model (CHM) raster, used when `fillin_method='chm'`.
        bboxes : RegionDetectionsSet, optional
            Bounding boxes for computing height values when `fillin_method='bbox'`.
        height_threshold : float or callable
            Max allowed height difference during matching. 
            - if float provided, it is considered a constant +/- tolerance in meters units
            - if callable provided, it should accept a heights array and return the tolerance for each point.
              Default allows +/-50% the treetop's height
        distance_threshold : float or callable, optional
            Constant value or callable for max allowed horizontal distance in meters. If callable,
            it should accept `height1` as input and return distance thresholds.
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
        height_vals_1, height_vals_2 = _prepare_heights(df1, df2, coords1, coords2, height_column_1, height_column_2, fillin_method, chm_path)

    else:
        height_vals_1, height_vals_2 = None, None

    # Compute XY distance matrix for validity checks
    distance_matrix_xy = cdist(coords1, coords2)  # (N1, N2)

    # Compute combined distance matrix for sorting if use_height_in_distance is set
    if not ignore_height:
        logging.info("Using height as an additional scaled dimension to compute distance") if use_height_in_distance > 0 else None
        # Note: if `use_height_in_distance` is zero, height has no effect on the distance
        # calculation and the result is identical to pure XY distance.
        coords1_aug = np.hstack([coords1, use_height_in_distance * height_vals_1.reshape(-1, 1)])
        coords2_aug = np.hstack([coords2, use_height_in_distance * height_vals_2.reshape(-1, 1)])
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
        max_d = distance_threshold
        valid_pairs_mask = distance_matrix < max_d
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

    if ignore_height:
        dist_height_pairs = np.stack([distances], axis=1)
    else:
        height_diffs = np.abs(height_vals_1[valid_idxs_1, 0] - height_vals_2[0, valid_idxs_2])
        dist_height_pairs = np.stack([distances, height_diffs], axis=1)

    # Sort matches by combined distance metric (which includes height if use_height_in_distance is set)
    sorted_idx = np.argsort(distances)
    valid_idxs_1 = valid_idxs_1[sorted_idx]
    valid_idxs_2 = valid_idxs_2[sorted_idx]
    dist_height_pairs = dist_height_pairs[sorted_idx]

    # Greedy matching
    max_valid_matches = min(distance_matrix.shape)
    matched_1 = []
    matched_2 = []
    matches = []

    for i1, i2, d_h in zip(valid_idxs_1, valid_idxs_2, dist_height_pairs):
        if i1 not in matched_1 and i2 not in matched_2:
            matches.append((i1, i2, d_h))
            matched_1.append(i1)
            matched_2.append(i2)

        if len(matched_1) == max_valid_matches:
            break

    if vis:
        _vis_matches(coords1, coords2, matches)

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
