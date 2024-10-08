from tree_detection_framework.detection.region_detections import (
    RegionDetections, RegionDetectionsSet)


def single_region_NMS(detections: RegionDetections) -> RegionDetections:
    """Run non-max suppresion on predictions from a single region.

    Args:
        detections (RegionDetections): Detections from a single region to run NMS on.

    Returns:
        RegionDetections:
            NMS-suppressed set of detections
    """
    raise NotImplementedError()


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
