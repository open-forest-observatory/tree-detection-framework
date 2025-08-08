from typing import Callable, List, Optional, Tuple, Union

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


def match_points(
    treetop_set_1: RegionDetections | RegionDetectionsSet | gpd.GeoDataFrame,
    treetop_set_2: RegionDetections | RegionDetectionsSet | gpd.GeoDataFrame,
    height1: Optional[str] = None,
    height2: Optional[str] = None,
    chm_path: Optional[PATH_TYPE] = None,
    bboxes: Optional[RegionDetectionsSet] = None,
    search_height_proportion: float = 0.5,
    search_distance_fun_slope: float = 0.1,
    search_distance_fun_intercept: float = 1.0,
    height_threshold: Optional[float] = None,
    distance_threshold: Union[float, Callable[[float], float]] = None,
    fillin_method: Optional[str] = None,
    use_height_in_distance: Optional[float] = None,
    vis: bool = False,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Match two sets of treetops.
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

    # Extract height arrays
    if height1 is not None and height2 is not None:
        height1 = df1[height1].values
        height2 = df2[height2].values
    elif fillin_method == "chm":
        # Path to CHM file given. Extract height for the coordinates
        if chm_path is None:
            raise ValueError("CHM path must be provided when fillin_method is 'chm'.")
        height1 = get_heights_from_chm(coords1, df1.crs, chm_path)
        height2 = get_heights_from_chm(coords2, df2.crs, chm_path)
    elif fillin_method == "bbox":
        # TODO: Decide logic to extract height values from bounding boxes
        pass
    else:
        height1 = None
        height2 = None

    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    height1 = np.expand_dims(np.array(height1), axis=1)  # (N1, 1)
    height2 = np.expand_dims(np.array(height2), axis=0)  # (1, N2)

    distance_matrix = cdist(coords1, coords2)  # (N1, N2)

    # Height bounds setup
    if height_threshold is not None:
        # Use constant height threshold for matching (override proportion)
        min_h = height1 - height_threshold
        max_h = height1 + height_threshold
    else:
        # Flexible height bounds based on proportion
        min_h = height1 * (1 - search_height_proportion)
        max_h = height1 * (1 + search_height_proportion)

    # Distance bounds setup
    if distance_threshold is not None:
        # If a callable is given, use it on height1 to calculate distance threshold array
        if callable(distance_threshold):
            max_d = distance_threshold(height1)
        else:
            # Constant distance threshold override
            max_d = distance_threshold
    else:
        # Use flexible distance threshold as function of height
        max_d = height1 * search_distance_fun_slope + search_distance_fun_intercept

    # Compute which matches fit all three criteria
    valid_pairs_mask = np.logical_and.reduce(
        [height2 > min_h, height2 < max_h, distance_matrix < max_d]
    )

    # Extract valid pair indices and their distances
    valid_idxs_1, valid_idxs_2 = np.where(valid_pairs_mask)
    distances = distance_matrix[valid_idxs_1, valid_idxs_2]
    # Calculate absolute height differences for the valid pairs
    height_diffs = np.abs(height1[valid_idxs_1, 0] - height2[0, valid_idxs_2])
    dist_height_pairs = np.stack([distances, height_diffs], axis=1)

    # Return the indices that sort the distances array in ascending order
    sorted_idx = np.argsort(distances)
    # Reorder the valid indices in ascending order of distance
    valid_idxs_1 = valid_idxs_1[sorted_idx]
    valid_idxs_2 = valid_idxs_2[sorted_idx]
    dist_height_pairs = dist_height_pairs[sorted_idx]

    # Compute the most possible pairs, which is the min of num of set 1 and set 2 trees
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

    return matches
