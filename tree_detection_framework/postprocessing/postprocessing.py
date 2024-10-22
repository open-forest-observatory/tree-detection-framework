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
    detections_df = detections.get_data_frame()

    # Get the axis-aligned bounds of each shape
    boxes = detections_df.bounds.to_numpy()
    # Extract the score
    confidences = detections_df[confidence_column].to_numpy()

    # Run NMS
    keep = nms(boxes, confidences, iou_threshold=iou_theshold)
    # Extract the detections that were kept
    subset_region_detections = detections.subset_detections(keep)

    return subset_region_detections


def multi_region_NMS(
    detections: RegionDetectionsSet,
    run_per_region_NMS: bool = True,
    iou_theshold: float = 0.5,
    confidence_column: str = "score",
) -> RegionDetections:
    """Run non-max suppresion on predictions from multiple regions.

    Args:
        detections (RegionDetectionsSet):
            Detections from multiple regions to run NMS on.
        run_per_region_NMS (bool):
            Should nonmax-suppression be run on each region before the regions are merged. This may
            lead to a speedup if there is a large amount of within-region overlap. Defaults to True.
        iou_threshold (float, optional):
            What intersection over union value to consider an overlapping detection. Defaults to 0.5.
        confidence_column (str, optional):
            Which column in the dataframe to use as a confidence for NMS. Defaults to "score"

    Returns:
        RegionDetections:
            NMS-suppressed set of detections, merged together for the set of regions.
    """
    # Determine whether to run NMS individually on each region.
    if run_per_region_NMS:
        # Run NMS on each sub-region and then wrap this in a region detection set
        region_detection_set_NMS_suppressed = RegionDetectionsSet(
            [
                single_region_NMS(
                    region_detections,
                    iou_theshold=iou_theshold,
                    confidence_column=confidence_column,
                )
                for region_detections in detections.region_detections
            ]
        )
        # Merge all the detections into one RegionDetections
        merged_detections = region_detection_set_NMS_suppressed.merge()
    else:
        # Just merge the detections
        merged_detections = detections.merge()
    # Run NMS on this merged RegionDetections
    NMS_suppressed_merged_detections = single_region_NMS(
        merged_detections,
        iou_theshold=iou_theshold,
        confidence_column=confidence_column,
    )

    return NMS_suppressed_merged_detections
