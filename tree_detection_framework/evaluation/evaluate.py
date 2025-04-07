from typing import List

import numpy as np
import geopandas as gpd
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon


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
    intersection = gpd.overlay(gt_gdf, pred_gdf, how='intersection')
    intersection['iou'] = intersection.area / (intersection['area_gt'] + (intersection['area_pred'] - intersection.area))

    # Create a cost matrix to store IoUs
    cost_matrix = np.zeros((len(gt_gdf), len(pred_gdf)))
    cost_matrix[intersection['id_gt'], intersection['id_pred']] = -intersection['iou']

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
    recall = np.sum(true_positives) / num_gt
    precision = np.sum(true_positives) / num_pd
    return precision, recall
