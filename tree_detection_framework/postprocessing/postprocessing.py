from typing import Optional

from lsnms import nms

from tree_detection_framework.detection.region_detections import (
    RegionDetections,
    RegionDetectionsSet,
)


def single_region_NMS(
    detections: RegionDetections,
    iou_theshold: float = 0.5,
    confidence_column: str = "score",
) -> RegionDetections:
    """Run non-max suppresion on predictions from a single region.

    Args:
        detections (RegionDetections):
            Detections from a single region to run NMS on.
        iou_threshold (float, optional):
            What intersection over union value to consider an overlapping detection. Defaults to 0.5.
        confidence_column (str, optional):
            Which column in the dataframe to use as a confidence for NMS. Defaults to "score"

    Returns:
        RegionDetections:
            NMS-suppressed set of detections
    """
    # Extract the geodataframe for the detections
    detections_df = detections.get_detections()

    # Get the axis-aligned bounds of each shape
    boxes = detections_df.bounds.to_numpy()
    # Extract the score
    confidences = detections_df[confidence_column].to_numpy()

    # Run NMS
    keep = nms(boxes, confidences, iou_threshold=iou_theshold)
    # Extract the detections that were kept
    subset_region_detections = detections.subset_detections(keep)

    return subset_region_detections


def multi_region_NMS(detections: RegionDetectionsSet) -> RegionDetections:
    """Run non-max suppresion on predictions from multiple regions.

    Args:
        detections (RegionDetectionsSet): Detections from multiple regions to run NMS on.

    Returns:
        RegionDetections:
            NMS-suppressed set of detections, merged together for the set of regions.
    """
    # This may implement more sophisticated algorithms, such as down-weighting predictions at the
    # boundaries or first performing within-tile NMS before across-tile NMS for computational reasons.
    raise NotImplementedError()
