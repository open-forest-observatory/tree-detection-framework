---
title: "Overview"
---
# Tree Detection Framework: Standardized Tree Detection on Geospatial Data

The Tree Detection Framework (TDF) is a tool developed by the [Open Forest
Observatory](https://openforestobservatory.org/) for detecting and delineating individual trees in
remote sensing data. Land managers and research ecologists increasingly use small uncrewed aerial
systems (sUAS or "drones") and airborne lidar to survey forests, and a common first step in
analyzing that data is to identify where the individual trees are. A number of machine learning
models and geometric algorithms exist for this task, but each ships with its own input format,
output format, dependencies, and tiling assumptions. Applying one of them to a
realistic-scale orthomosaic — or comparing two of them fairly — involves a substantial amount of
one-off glue code.

TDF was developed to remove that glue code. It has three main goals:

* Enable tree detection on realistic-scale, geospatial raster data with minimal boilerplate, using
  existing (external) tree detection and segmentation models.
* Facilitate direct comparison of multiple algorithms.
* Rely on modern libraries and software best practice for a robust, performant, and modular tool.

With the exception of a geometric algorithm, this project does not itself provide tree
detection/segmentation algorithms. Instead, it provides a standardized interface for performing
training, inference, and evaluation using existing tree detection models. Support for additional
external models can be added by implementing a new `Detector` subclass.

## How it works

TDF uses the [`torchgeo`](https://torchgeo.readthedocs.io/) package to perform data loading and
standardization using standard geospatial input formats. This allows chips (tiles) to be generated
on the fly at a given size, stride, and spatial resolution, so an orthomosaic of any size can be
processed without being manually pre-tiled. Training and inference are done with modular detectors
that wrap existing models and algorithms, with preliminary support for [PyTorch
Lightning](https://lightning.ai/docs/pytorch/stable/) to minimize boilerplate. Region-level
non-maximum suppression (NMS) is done using the `PolyGoneNMS` library, which is efficient for large
images. Visualization and saving of predictions is done with
[`geopandas`](https://geopandas.org/), so results are written out as standard geospatial files
(`.gpkg` or `.geojson`).

## Supported detection algorithms

### DeepForest

* [GitHub](https://github.com/weecology/DeepForest)
* Uses RGB input data. Predicts tree crowns with rectangular bounding boxes.
* Provides a RetinaNet model trained on a large number of semi-supervised tree crown annotations
  and a smaller set of manual annotations.
* Trained using data from the US only but representing diverse regions. The model has been applied
  to data from outside the US successfully.

### Detectree2

* [GitHub](https://github.com/PatBall1/detectree2)
* Uses RGB input data. Predicts tree crowns with polygon boundaries.
* Provides a Mask R-CNN model trained on manually labeled tree crowns from four sites.
* Trained using data from tropical forests.

### Segment Anything Model 2 (SAM2)

* [GitHub](https://github.com/facebookresearch/sam2)
* Uses RGB input data. Predicts objects with polygon boundaries.
* Utilizes the Segment Anything Model (SAM 2.1 Hiera Large) checkpoint with tuned parameters for
  mask generation optimized for tree crown delineation.
* Does not rely on supervised training for tree-specific data but generalizes well due to SAM's
  zero-shot nature; however, non-tree objects are also detected and included in predictions.

### Segment Anything Model 3 (SAM3)

* [GitHub](https://github.com/facebookresearch/sam3)
* Uses RGB input data. Predicts objects with polygon boundaries.
* Utilizes the SAM3 checkpoint with the text prompt "tree" for tree crown delineation.
* Does not rely on supervised training for tree-specific data but generalizes well due to SAM's
  zero-shot nature. Unlike SAM2, non-tree objects are not usually detected or included in
  predictions.

### Geometric detector

* Implementation of the variable window filter algorithm of [Popescu and Wynne
  (2004)](https://www.ingentaconnect.com/content/asprs/pers/2004/00000070/00000005/art00003) for
  tree top detection, combined with the algorithm of [Silva et al.
  (2016)](https://www.tandfonline.com/doi/full/10.1080/07038992.2016.1196582#abstract) for crown
  segmentation.
* Uses canopy height model (CHM) input data. Predicts tree crowns with polygon boundaries.
* This is a learning-free tree detection algorithm. It is the one algorithm that is implemented
  within TDF as opposed to relying on an existing external model/algorithm.

## Software architecture

TDF is organized into modular components to make it straightforward to extend, including
integrating additional detection models. The main components are:

1. **`preprocessing.py`**<br>
   The `create_dataloader()` function accepts single or multiple orthomosaic inputs. Alternatively,
   `create_image_dataloader()` accepts a folder containing raw drone imagery. These functions tile
   the input images based on user-specified parameters such as tile size, stride, and resolution,
   and return a PyTorch-compatible dataloader for inference.
2. **`Detector` base class**<br>
   All detectors in the framework (e.g. `DeepForestDetector`, `MaskRCNNDetector`) inherit from the
   `Detector` base class. The base class defines the core logic for generating predictions and
   geospatially referencing image tiles, while model-specific detectors translate the inputs to the
   format expected by the respective model. This design allows all detectors to plug into the same
   pipeline with minimal code changes.
3. **`RegionDetections` and `RegionDetectionsSet`**<br>
   These classes standardize model outputs. A `RegionDetectionsSet` is a collection of
   `RegionDetections`, where each `RegionDetections` object represents the detections in a single
   image tile. This abstraction allows postprocessing components to operate uniformly across
   different detectors. These outputs can be saved as `.gpkg` or `.geojson` files.
4. **`postprocessing.py`**<br>
   Implements a set of postprocessing functions for cleaning up detections: non-maximum suppression
   (NMS), polygon hole suppression, tile boundary suppression, and removal of out-of-bounds
   detections. Most of these functions operate on the standardized output types
   (`RegionDetections` / `RegionDetectionsSet`).

## Where to go next

* **[Getting Started](getting_started/installation.md)** — install TDF and get example data.
* **[Examples and Applications](examples_and_applications/examples.md)** — annotated walkthroughs of
  the example notebooks.
* **[Command Line Usage](command_line_usage/index.md)** — run TDF from a terminal, with no Python
  required.
* **[API Reference](API_reference/entrypoints.md)** — generated documentation for every module.

This project is under active development. We welcome contributions and suggestions for improvement.
