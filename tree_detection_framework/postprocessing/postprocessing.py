import logging

import numpy as np
from polygone_nms import nms

from tree_detection_framework.detection.region_detections import (
    RegionDetections,
    RegionDetectionsSet,
)


def single_region_NMS(
    detections: RegionDetections,
    iou_theshold: float = 0.5,
    confidence_column: str = "score",
    min_confidence: float = 0.3,
) -> RegionDetections:
    """Run non-max suppresion on predictions from a single region.

    Args:
        detections (RegionDetections):
            Detections from a single region to run NMS on.
        iou_threshold (float, optional):
            What intersection over union value to consider an overlapping detection. Defaults to 0.5.
        confidence_column (str, optional):
            Which column in the dataframe to use as a confidence for NMS. Defaults to "score"
        min_confidence (float, optional):
            Prediction score threshold for detections to be included.

    Returns:
        RegionDetections:
            NMS-suppressed set of detections
    """
    # Extract the geodataframe for the detections
    detections_df = detections.get_data_frame()

    # Determine which detections are high enough confidence to retain
    high_conf_inds = np.where(
        (detections_df[confidence_column] >= min_confidence).to_numpy()
    )[0]

    # Filter detections based on minimum confidence score
    detections_df = detections_df.iloc[high_conf_inds]
    if detections_df.empty:
        # Return empty if no detections pass threshold
        return detections.subset_detections([])

    ## Get the polygons for each detection object
    polygons = detections_df.geometry.to_list()
    # Extract the score
    confidences = detections_df[confidence_column].to_numpy()

    # Put the data into the required format, list[(polygon, class, confidence)]
    # TODO consider adding a class, currently set to all ones
    input_data = list(zip(polygons, np.ones_like(confidences), confidences))

    # Run polygon NMS
    keep_inds = nms(
        input_data=input_data,
        distributed=None,
        nms_method="Default",
        intersection_method="IOU",
        threshold=iou_theshold,
    )

    # We only performed NMS on the high-confidence detections, but we need the indices w.r.t. the
    # original data with all detections. Sort for convenience so data is not permuted.
    keep_inds_in_original = sorted(high_conf_inds[keep_inds])
    # Extract the detections that were kept
    subset_region_detections = detections.subset_detections(keep_inds_in_original)

    return subset_region_detections


def multi_region_NMS(
    detections: RegionDetectionsSet,
    run_per_region_NMS: bool = True,
    iou_theshold: float = 0.5,
    confidence_column: str = "score",
    min_confidence: float = 0.3,
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

        min_confidence (float, optional):
            Prediction score threshold for detections to be included.
    Returns:
        RegionDetections:
            NMS-suppressed set of detections, merged together for the set of regions.
    """
    # Determine whether to run NMS individually on each region.
    if run_per_region_NMS:
        # Run NMS on each sub-region and then wrap this in a region detection set
        detections = RegionDetectionsSet(
            [
                single_region_NMS(
                    region_detections,
                    iou_theshold=iou_theshold,
                    confidence_column=confidence_column,
                    min_confidence=min_confidence,
                )
                for region_detections in detections.region_detections
            ]
        )

    # Merge the detections into a single RegionDetections
    merged_detections = detections.merge()

    # If the bounds of the individual regions were disjoint, then no NMS needs to be applied across
    # the different regions
    if detections.disjoint_bounds():
        logging.info("Bounds are disjoint, skipping across-region NMS")
        return merged_detections
    logging.info("Bound have overlap, running across-region NMS")

    # Run NMS on this merged RegionDetections
    NMS_suppressed_merged_detections = single_region_NMS(
        merged_detections,
        iou_theshold=iou_theshold,
        confidence_column=confidence_column,
        min_confidence=min_confidence,
    )

    return NMS_suppressed_merged_detections
