# `detect_trees`

Run any of the supported detectors through a single, uniform command and write the result to a fixed
filename inside an output directory.

This is a pipeline step rather than a command meant to be typed by hand. It exists so an automated
workflow can call one command regardless of which detector was selected, including the geometric
one, and know exactly where the output will be. Internally it dispatches to
[`detect_geometric_two_stage`](detect_geometric_two_stage.md) for the geometric detector and to
[`generate_predictions`](generate_predictions.md) for everything else. If you are running detection
interactively, use those commands directly instead.

Its output is designed to be read by [`postprocess`](postprocess.md), which is the intended next
step.

```
python -m tree_detection_framework.entrypoints.detect_trees \
    --detector <NAME> \
    --detection-params-json <JSON> \
    --detector-dir <PATH> \
    --preprocessed-local-files <JSON>
```

## Example

```
python -m tree_detection_framework.entrypoints.detect_trees \
    --detector deepforest \
    --detection-params-json '{"chip_size": 1024, "chip_stride": 768, "resolution": 0.05, "batch_size": 4}' \
    --detector-dir runs/deepforest \
    --preprocessed-local-files '{"ortho": "data/emerald-point-ortho/ortho.tif", "chm": "data/emerald-point-chm/chm.tif"}'
```

## Required arguments

All four are required.

| Argument | Type | Description |
| --- | --- | --- |
| `--detector` | choice | Which detector to run. One of `geometric`, `deepforest`, `detectree2`, `sam2`, `sam3`, `tcd`. |
| `--detection-params-json` | JSON string | Detection parameters. See below. |
| `--detector-dir` | path | Directory for detector outputs. `raw_detections.gpkg` is written here; the directory is created if needed. |
| `--preprocessed-local-files` | JSON string | Paths to the input rasters, as a JSON object with the keys `ortho` and `chm`. Both keys must be present, even for a detector that only uses one of them. |

## `--detection-params-json`

A JSON object. Which keys are read depends on the detector.

### For `--detector geometric`

The CHM from `--preprocessed-local-files` is used as input.

| Key | Default | Description |
| --- | --- | --- |
| `chip_size` | `2000` | Chip size in pixels. |
| `chip_stride` | `1900` | Chip stride in pixels. |
| `resolution` | `0.2` | Meters per pixel the CHM is resampled to. |
| `raster_blur_sigma` | none | Gaussian smoothing sigma in meters applied to the CHM. |
| `edge_suppression_meters` | none | Suppress trees whose tops fall within this distance of the raster edge. |

!!! note
    Only tree tops are produced in this mode — `raw_detections.gpkg` contains the tree tops, and no
    crown save path is passed through, so stage two is skipped. Use
    [`detect_geometric_two_stage`](detect_geometric_two_stage.md) directly if you need crowns.

### For the deep learning detectors

The orthomosaic from `--preprocessed-local-files` is used as input.

| Key | Required | Description |
| --- | --- | --- |
| `chip_size` | yes | Chip size. Unlike the geometric branch, this has no default and must be supplied. |
| `chip_stride` | yes | Chip stride. Also has no default here. |
| `resolution` | no | Meters per pixel. Falls back to the resolution of the input raster. |
| `batch_size` | no | Batch size. |
| `detectree2_weights_path` | no | Path to Detectree2 weights. Defaults to `checkpoints/230103_randresize_full.pth`. |
| `sam2_checkpoint_path` | no | Path to the SAM2 checkpoint. |
| `sam3_checkpoint_path` | no | Path to a local SAM3 checkpoint. Defaults to `checkpoints/sam3.pt`. |
| `sam3_huggingface_token` | no | HuggingFace token for downloading SAM3 weights. |

**Any other key** in the object is forwarded to the detector's constructor as a keyword argument —
this is the equivalent of `--detector-kwargs` on
[`generate_predictions`](generate_predictions.md#advanced). Values that look numeric are converted to
`int` or `float` automatically, so `{"points_per_side": "32"}` and `{"points_per_side": 32}` behave
the same.

## Output

`<detector-dir>/raw_detections.gpkg`. For the deep learning detectors, non-maximum suppression is
deliberately **not** run at this stage, so the file keeps its per-tile structure and still contains
duplicate detections from overlapping chips. Cleaning that up is the job of
[`postprocess`](postprocess.md).
