# `detect_in_raw_images`

Detect trees in a folder of raw (non-georeferenced) drone images, rather than in a stitched
orthomosaic. One detection file is written per input image.

Because raw images have no geospatial reference, the detections are in pixel coordinates relative to
each image, not in a map projection. Each image is tiled, run through the model, cleaned up, and
saved separately.

```
python -m tree_detection_framework.entrypoints.detect_in_raw_images \
    <image_dir> <out_dir> <model_key> [options]
```

## Example

```
python -m tree_detection_framework.entrypoints.detect_in_raw_images \
    data/raw_images \
    data/raw_image_detections \
    deepforest \
    --chip_size 2200 \
    --chip_stride 2000 \
    --batch_size 4
```

## Required arguments

These are positional: give them in this order, with no flag in front.

| Position | Argument | Type | Description |
| --- | --- | --- | --- |
| 1 | `image_dir` | path | Directory containing raw images, or nested subdirectories containing them. All images are read recursively. Must exist. |
| 2 | `out_dir` | path | Directory to save detection results into. Created if it does not exist. Any subdirectory structure inside `image_dir` is mirrored here. |
| 3 | `model_key` | choice | Which model to use. One of `deepforest`, `detectree2`, `sam2`. |

## Optional arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--detectree-checkpoints` | path | none | Path to the Detectree2 checkpoint file, e.g. `230103_randresize_full.pth`. **Required when `model_key` is `detectree2`** — the command exits with an error if it is missing, and also if the path is not a real file. Ignored for the other models. |
| `--chip_size` | integer | `2200` | Size in pixels of each square chip an image is cut into. |
| `--chip_stride` | integer | `2000` | Distance in pixels between the start of one chip and the next. |
| `--batch_size` | integer | `4` | Number of chips fed to the model at once. |

!!! note "These three flags use underscores, not dashes"
    Unlike the other entrypoints, this script spells them `--chip_size`, `--chip_stride`, and
    `--batch_size`. `--detectree-checkpoints` does use a dash.

## Behavior

* A GPU is used automatically if one is available; otherwise the CPU is used. The device chosen is
  printed at startup.
* After prediction, detections extending past the bounds of the image are removed, and overlapping
  detections within each image are suppressed with non-maximum suppression. Both steps are automatic
  and have no flags.

## Output

**One GeoPackage per input image**, written under `out_dir` (the second positional argument), which
is created if it does not exist. There is no single combined file — if you need one, merge them
afterwards.

| | |
| --- | --- |
| Path | `<out_dir>/<same relative path as the image>/<image name>.gpkg` |
| Naming | The image's extension is replaced with `.gpkg`, so `DJI_0123.JPG` produces `DJI_0123.gpkg`. |
| Structure | The subdirectory layout of `image_dir` is mirrored. An image at `image_dir/flight1/north/DJI_0123.JPG` produces `out_dir/flight1/north/DJI_0123.gpkg`. Intermediate directories are created as needed. |

Each file is a single-layer GeoPackage with one row per detection: a `geometry` column (bounding
box for DeepForest, polygon for Detectree2 and SAM2), a `score` column, and any detector-specific
columns — SAM2 adds `predicted_iou` and `stability_score`.

!!! warning "These coordinates are not geospatial"
    Raw drone images are not georeferenced, so the geometries are in **pixel coordinates relative to
    each image**, and the files carry no CRS. They cannot be overlaid on a map or combined across
    images without first being projected. For detections in real-world coordinates, run
    [`generate_predictions`](generate_predictions.md) against an orthomosaic instead.

The output path is printed on completion. To render these detections on top of the source images,
see [`visualize_detections`](visualize_detections.md), which is designed to consume this output —
it pairs files by name, which is why the naming convention above matters.
