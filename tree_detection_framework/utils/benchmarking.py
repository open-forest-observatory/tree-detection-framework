import logging
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
from shapely.geometry import box

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.detection.detector import Detector
from tree_detection_framework.evaluation.evaluate import (
    compute_matched_ious,
    compute_precision_recall,
)
from tree_detection_framework.postprocessing.postprocessing import single_region_NMS
from tree_detection_framework.preprocessing.preprocessing import create_image_dataloader

logging.basicConfig(level=logging.INFO)

def extract_neon_groundtruth(images_dir: PATH_TYPE, annotations_dir: PATH_TYPE) -> dict[str, dict[str, List[box]]]:
    """
    Extract ground truth bounding boxes from NEON XML annotations.
    Args:
        images_dir (PATH_TYPE): Directory containing image tiles.
        annotations_dir (PATH_TYPE): Directory containing XML annotation files.
    Returns:
        dict: A dictionary mapping image paths to a dictionary with "gt" key containing ground truth boxes.
    """
    tiles_to_predict = list(Path(images_dir).glob("*.tif"))
    mappings = {}

    for path in tiles_to_predict:
        plot_name = path.stem  # Get filename without extension
        annot_fname = Path(annotations_dir) / f"{plot_name}.xml"

        if not annot_fname.exists():
            continue

        # Load XML file
        tree = ET.parse(annot_fname)
        root = tree.getroot()

        # Extract bounding boxes
        gt_boxes = []
        for obj in root.findall(".//object"):
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                gt_boxes.append(box(xmin, ymin, xmax, ymax))

        # Add the ground truth boxes to the mappings
        mappings[str(path)] = {"gt": gt_boxes}
    return mappings

def get_neon_detections(
    images_dir: PATH_TYPE, annotations_dir: PATH_TYPE, detectors: dict[str, Detector], nms_threshold: float = None, min_confidence: float = 0.5
) -> dict[str, dict[str, List[box]]]:
    """Step 1: Get predictions using the detcetors on the NEON dataset.
    Args:
        images_dir (PATH_TYPE): Directory containing image tiles.
        annotations_dir (PATH_TYPE): Directory containing XML annotation files.
        detectors (dict[str, Detector]): Dictionary mapping detector names to Detector instances.
        nms_threshold (float, optional): Non-Maximum Suppression threshold. Default is None (no NMS applied on predictions).
        min_confidence (float, optional): Minimum confidence threshold for predictions. Default is 0.5.
        Set keys from: ["deepforest", "detectree2", "sam2"]
    Returns:
        dict: Dictionary mapping image paths to a dictionary with detector names and the corresponding output boxes.
    """
    # A dictionary mapping image paths to a dictionary with "gt" key containing ground truth boxes
    mappings = extract_neon_groundtruth(images_dir, annotations_dir)

    # Create dataloader setting image size as 420x420. NEON dataset has a standard size of 400x400.
    dataloader = create_image_dataloader(
        list(mappings.keys()),
        chip_size=420,
        chip_stride=420,
        batch_size=1,
    )

    for name, detector in detectors.items():
        # Get predictions from every detector
        logging.info(f"Running detector: {name}")
        region_detection_sets, filenames, _ = detector.predict_raw_drone_images(
            dataloader
        )

        # Add predictions to the mappings so that it looks like:
        # {"image_path_1": {"gt": gt_boxes, "detector_name_1": [boxes], ...},
        #  "image_path_2": {"gt": gt_boxes, "detector_name_1": [boxes], ...}, ...}
        for filename, rds in zip(filenames, region_detection_sets):
            if nms_threshold is not None:
                rds = single_region_NMS(rds.get_region_detections(0), threshold=nms_threshold, min_confidence=min_confidence)
            gdf = rds.get_data_frame()

            # Add the detections to the mappings dictionary
            if name == "deepforest":
                mappings[filename][name] = list(gdf.geometry)
            elif name == "detectree2":
                mappings[filename][name] = list(gdf["bbox"])
            elif name == "sam2":
                # TODO
                pass
            else:
                raise ValueError(f"Unknown detector: {name}")

    return mappings


def evaluate_detections(detections_dict: dict[str, dict[str, List[box]]]):
    """Step 2: Compute precision and recall for each detector.
    Args:
        detections_dict (dict): Dictionary mapping image paths to a dictionary
        with detector names and the corresponding output boxes. Output of get_neon_detections.
    """
    img_paths = list(detections_dict.keys())
    # Get the list of detectors, which are keys of the sub-dictionary.
    detector_names = [
        key for key in detections_dict[img_paths[0]].keys() if key != "gt"
    ]
    logging.info(f"Detectors to be evaluated: {detector_names}")
    for detector in detector_names:
        all_predictions_P = []
        all_predictions_R = []
        for img in img_paths:
            gt_boxes = detections_dict[img]["gt"]
            pred_boxes = detections_dict[img][detector]
            iou_output = compute_matched_ious(gt_boxes, pred_boxes)
            P, R = compute_precision_recall(iou_output, len(gt_boxes), len(pred_boxes))
            all_predictions_P.append(P)
            all_predictions_R.append(R)

        P = np.mean(all_predictions_P)
        R = np.mean(all_predictions_R)
        F1 = (2 * P * R) / (P + R)
        print(f"'{detector}': Precision={P}, Recall={R}, F1-Score={F1}")
