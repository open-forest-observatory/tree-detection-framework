---
weight: 10
title: Detection
---

# Detection

The detectors themselves and the standardized types they return. All detectors inherit from the
`Detector` base class, which handles running the model over a dataloader and geospatially
referencing each tile's predictions.

- [Detector Docstrings](detector.md) — the `Detector` base class and the detectors built on it,
  including `DeepForestDetector`, `MaskRCNNDetector`, and the geometric tree top and tree crown
  detectors.

- [Models Docstrings](models.md) — the PyTorch Lightning modules wrapping the external models:
  DeepForest, Detectree2, and TCD.

- [SAM2 Detector Docstrings](SAM2_detector.md) — the Segment Anything Model 2 detector.

- [SAM3 Detector Docstrings](SAM3_detector.md) — the Segment Anything Model 3 detector.

- [Region Detections Docstrings](region_detections.md) — `RegionDetections` and
  `RegionDetectionsSet`, the standardized output types that every detector produces and every
  postprocessing function consumes.
