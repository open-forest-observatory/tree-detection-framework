import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment
from typing import List

def calculate_polygon_iou(polyA: Polygon, polyB: Polygon) -> float:
    """Compute Intersection over Union (IoU) between two polygons.
    Args:
        polyA (Polygon): First polygon
        polyB (Polygon): Second polygon
    Returns:
        float: IoU value between 0 and 1
    """
    intersection = polyA.intersection(polyB).area
    union = polyA.union(polyB).area
    return intersection / union if union > 0 else 0

def compute_matched_ious(ground_truth_boxes: List[Polygon], predicted_boxes: List[Polygon]) -> List:
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

    num_gt = len(ground_truth_boxes)
    num_pred = len(predicted_boxes)
    
    # Create IoU cost matrix (negative because Hungarian minimizes cost)
    cost_matrix = np.zeros((num_gt, num_pred))
    for i, gt in enumerate(ground_truth_boxes):
        for j, pred in enumerate(predicted_boxes):
            cost_matrix[i, j] = -calculate_polygon_iou(gt, pred)

    # Solve optimal assignment using the Hungarian algorithm
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    # Extract IoUs of matched pairs
    matched_ious = [-cost_matrix[i, j] for i, j in zip(gt_indices, pred_indices)]
    return matched_ious

def compute_precision_recall(ious: List, num_gt: int, num_pd: int, threshold: float=0.4) -> tuple:
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
    return round(precision, 3), round(recall, 3)