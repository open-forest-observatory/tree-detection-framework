from tree_detection_framework.detection.region_detections import (
    RegionDetections,
    RegionDetectionsSet,
)


def single_region_NMS(detections: RegionDetections):
    raise NotImplementedError()


def multi_region_NMS(detections: RegionDetectionSet):
    raise NotImplementedError()
