# Examples

The
[`examples`](https://github.com/open-forest-observatory/tree-detection-framework/tree/main/examples)
folder contains the maintained, end-to-end workflows for TDF, in the form of Jupyter notebooks. All
of them use the Emerald Point example data described in [Data](../getting_started/data.md), and all
of them follow the same overall shape:

1. Build a dataloader that chips a large georeferenced raster on the fly.
2. Construct a detector.
3. Run `detector.predict(dataloader)`, which returns a `RegionDetectionsSet` — one
   `RegionDetections` per tile.
4. Postprocess across tiles (mainly non-maximum suppression, since chips overlap and the same tree
   is detected more than once).
5. Plot and save the result as a geospatial file.

What differs between the notebooks is the detector, the kind of input raster it consumes, and the
postprocessing that particular detector needs.

## Bounding-box and mask detection with DeepForest and Detectree2

This notebook is the best starting point. It runs a pretrained deep learning detector over an RGB
orthomosaic and produces tree crowns in geospatial coordinates.

A single `TREE_DETECTOR` constant at the top switches between the two models, which is the concrete
demonstration of the framework's central claim: the same pipeline runs either model with no other
changes. `DeepForestDetector` wraps a `DeepForestModule` (a RetinaNet that predicts axis-aligned
bounding boxes) and is moved to the GPU if one is available. `MaskRCNNDetector` wraps a
`Detectree2Module`, which predicts polygon masks and requires the downloaded
`230103_randresize_full.pth` checkpoint.

The workflow is:

* `create_dataloader` reads `data/emerald-point-ortho`, chipping it into 1024 px tiles with a 768 px
  stride at 0.05 m/pixel resolution, batched 4 at a time. The overlap between chips (1024 vs. 768)
  is deliberate — it ensures trees near a tile edge are fully visible in at least one tile.
* `visualize_dataloader` displays a few sampled tiles, so you can confirm the chip size and
  resolution are sensible for the trees in your imagery before spending time on inference.
* `detector.predict(dataloader)` produces the detections. They are plotted twice: once for the whole
  region overlaid on the orthomosaic and colored by the `score` column, and once for a single tile
  via `outputs.region_detections[0].plot(...)`, which shows what a single unmerged tile looks like.
* `multi_region_NMS` then resolves the duplicate detections created by the overlapping chips,
  suppressing detections that overlap by more than an IoU of 0.3 and dropping any with a confidence
  below 0.3. Plotting again after this step shows the effect clearly.
* `NMS_outputs.save(...)` writes a `.gpkg`.

* [examples/predict_detections.ipynb](https://github.com/open-forest-observatory/tree-detection-framework/blob/main/examples/predict_detections.ipynb)

## Zero-shot segmentation with SAM2

This notebook is deliberately near-identical to the previous one, which is the point: swapping in a
completely different class of model changes only the detector construction. `SAMV2Detector()` is
constructed with no arguments — it loads the SAM 2.1 Hiera Large checkpoint from the `checkpoints`
folder and uses TDF's tuned automatic mask generation parameters. Everything else — dataloader
creation, prediction, plotting, `multi_region_NMS`, saving — is unchanged from the DeepForest and
Detectree2 notebook.

Because SAM2 is a general-purpose segmentation model rather than a tree model, it is zero-shot: it
needs no tree-specific training but will also happily segment rocks, roads, and shadows. Those
non-tree detections are present in the output, and the next notebook shows one way to remove them.

* [examples/predict_detections_sam2.ipynb](https://github.com/open-forest-observatory/tree-detection-framework/blob/main/examples/predict_detections_sam2.ipynb)

## Text-prompted segmentation with SAM3, filtered by canopy height

SAM3 is prompted with the text "tree", so unlike SAM2 it mostly returns trees rather than every
object in the scene. `SAM3Detector` takes a `huggingface_token` (see
[Installation](../getting_started/installation.md#optional-sam3)) and a `confidence_threshold`,
which is applied inside the detector rather than downstream. Larger chips are used here (2000 px
with a 1200 px stride) than in the other notebooks.

This notebook also demonstrates using a second raster to clean up the results. After
`multi_region_NMS`, it calls `plot_ortho_chm_overlay` to display the canopy height model on top of
the orthomosaic, then `filter_by_chm(NMS_outputs, CHM_FILE_PATH, min_height=10)` to discard
detections whose height in the CHM is below 10 m. This removes ground-level objects that look
tree-like in RGB but are obviously not trees once height is taken into account.

!!! note
    The notebook saves `NMS_outputs` rather than `CHM_filtered_detections`, so the file on disk is
    the unfiltered version. Change the argument to `save` if you want the height-filtered result.

* [examples/predict_detections_sam3.ipynb](https://github.com/open-forest-observatory/tree-detection-framework/blob/main/examples/predict_detections_sam3.ipynb)

## Two-stage geometric detection from a canopy height model

This is the one algorithm implemented within TDF rather than wrapped from an external project, and
it is the only workflow that uses a canopy height model as its primary input instead of RGB. It
requires no learned weights at all. It is a two-stage pipeline, and the notebook shows why the
framework's dataloader abstraction matters: the output of stage one becomes an input to the
dataloader for stage two.

**Stage 1 — tree tops.** A standard `create_dataloader` chips the CHM (512 px chips, 400 px stride,
0.2 m/pixel). `GeometricTreeTopDetector` implements a variable window filter: it Gaussian-blurs the
CHM by `blur_sigma` and finds local maxima using a search window whose radius is a function of
canopy height, given by the coefficients `a=0`, `b=0.0325`, `c=0.25`. Taller trees get a larger
window, which is what stops a single wide crown from being split into several detections. The
result is a set of points.

Duplicates from overlapping chips are handled here by `remove_detections_from_tile_overlap` rather
than NMS — for point detections, simply discarding whatever falls in a tile's overlap region is
both cheaper and cleaner than suppressing by IoU. The tree tops are saved to their own file at this
stage.

**Stage 2 — crowns.** `create_intersection_dataloader` combines the CHM raster with the tree top
points from stage one, so each tile now carries both the height data and its seed points.
`GeometricTreeCrownDetector(approach="watershed")` grows a crown outward from each seed. Because
crowns are polygons that can straddle tile boundaries, this stage does use `multi_region_NMS`, with
`intersection_method="IOS"` (intersection over smaller) rather than IoU, which is the appropriate
choice when one crown may be substantially contained within another.

The notebook then calls `treecrown_detections.get_data_frame()` to show that the treetop unique IDs
survive into the crown detections, so each crown can be traced back to the tree top that seeded it.
Finally `single_region_hole_suppression` fills interior holes in the crown polygons — the plot
before and after makes the difference visible — and the crowns are saved.

* [examples/sequential_geometric_detector.ipynb](https://github.com/open-forest-observatory/tree-detection-framework/blob/main/examples/sequential_geometric_detector.ipynb)

The same two-stage workflow is available as a command line script; see
[`detect_geometric_two_stage`](../command_line_usage/detect_geometric_two_stage.md).

