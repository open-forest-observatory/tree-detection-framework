---
title: Postprocessing
---

Functions for cleaning up raw detections: non-maximum suppression across and within tiles, polygon
hole suppression, tile boundary suppression, masking, and filtering by canopy height. Most of these
operate on the standardized `RegionDetections` and `RegionDetectionsSet` types.

::: tree_detection_framework.postprocessing.postprocessing
