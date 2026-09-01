# `generate_predictions`

Generate tree detections for a georeferenced raster dataset using one of the supported deep learning
detectors. This is the main detection command. It creates a dataloader from the tiled raster,
runs the selected model over every tile, optionally suppresses overlapping detections, and
optionally plots and saves the result.

For the canopy height model (geometric) detector, use
[`detect_geometric_two_stage`](detect_geometric_two_stage.md) instead.

```
python -m tree_detection_framework.entrypoints.generate_predictions \
    --raster-folder-path <PATH> \
    --chip-size <NUMBER> \
    --tree-detection-model <MODEL> \
    [other options]
```

## Example

```
python -m tree_detection_framework.entrypoints.generate_predictions \
    --raster-folder-path data/emerald-point-ortho \
    --chip-size 1024 \
    --chip-stride 768 \
    --resolution 0.05 \
    --tree-detection-model deepforest \
    --batch-size 4 \
    --run-nms \
    --iou-threshold 0.3 \
    --min-confidence 0.3 \
    --predictions-save-path data/emerald-point-predictions.gpkg
```

## Required arguments

| Argument | Type | Description |
| --- | --- | --- |
| `--raster-folder-path` | path | Path to a raster file, or to a folder of raster files, to run detection on. Typically an orthomosaic. |
| `--chip-size` | number | Size of each square chip the raster is cut into. Interpreted as pixels unless `--use-units-meters` is given. |
| `--tree-detection-model` | choice | Which model to use. One of `deepforest`, `detectree2`, `sam2`, `sam3`, `tcd`. Anything else causes an error listing the valid options. |

## Tiling options

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--chip-stride` | number | none | Distance between the start of one chip and the next. A stride smaller than the chip size produces overlapping chips, which avoids cutting trees at tile edges. Do not set this together with `--chip-overlap-percentage`. |
| `--chip-overlap-percentage` | number | none | Overlap between neighboring chips, from 0 to 100. An alternative way of expressing the stride. Do not set this together with `--chip-stride`. |
| `--use-units-meters` | flag | off | Interpret `--chip-size` and `--chip-stride` in meters on the ground rather than in pixels. |
| `--resolution` | number | resolution of the input | Ground sample distance in meters per pixel that the raster is resampled to before detection. If unset, the native resolution of the first raster read is used. Matching this to the resolution the model was trained at matters a great deal for detection quality. |
| `--region-of-interest` | path | none | Restrict processing to a spatial sub-region. Expects a path to a geospatial vector file describing the region. |
| `--output-CRS` | string | CRS of the input | Coordinate reference system for the output detections, e.g. `EPSG:32610`. If unset, the CRS of the first tile is used. |
| `--batch-size` | integer | `1` | Number of chips fed to the model at once. Larger values are faster on a GPU but use more memory. |

## Output options

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--predictions-save-path` | path | none | Where to write the detections, e.g. `detections.gpkg`. If omitted, nothing is saved. The file format is inferred from the extension; `.gpkg` and `.geojson` both work. |
| `--view-predictions-plot` | flag | off | Open a plot window showing the detections and region bounds. The script waits until you close the window. Not useful on a headless machine. |

## Postprocessing options

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--run-nms` | flag | off | Run non-maximum suppression across all tiles, merging them into a single set of detections. Recommended whenever chips overlap, since the same tree is otherwise detected once per chip that contains it. |
| `--iou-threshold` | number | `0.3` | Two detections overlapping by more than this intersection-over-union are treated as the same object, and the lower-scoring one is suppressed. Only applies when `--run-nms` is given. |
| `--min-confidence` | number | `0.3` | Detections scoring below this are discarded. Only applies when `--run-nms` is given — except for SAM3, where this value is also passed to the detector as its confidence threshold regardless. |

!!! note "How the output file is written depends on `--run-nms`"
    With `--run-nms`, the tiles are merged and a single flat layer of detections is written. Without
    it, the per-tile structure is preserved in the output file instead. See [Output](#output) below.

## Model checkpoint options

Only the option matching your chosen `--tree-detection-model` is used.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--detectree2-weights-path` | path | `checkpoints/230103_randresize_full.pth` | Path to the Detectree2 pretrained weights `.pth` file. See [Installation](../getting_started/installation.md#optional-detectree2) for how to download it. |
| `--sam2-checkpoint-path` | path | `checkpoints/sam2.1_hiera_large.pt` | Path to the SAM2 checkpoint file. |
| `--sam3-checkpoint-path` | path | none | Path to a local SAM3 checkpoint file. If not set, the weights are downloaded from HuggingFace, which requires `--sam3-huggingface-token`. |
| `--sam3-huggingface-token` | string | none | HuggingFace access token used to download the SAM3 weights. Only used when `--sam3-checkpoint-path` is not set. See [Installation](../getting_started/installation.md#optional-sam3) for how to create one. |

## Advanced

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--detector-kwargs` | JSON string | `{}` | Named arguments passed straight through to the detector's constructor, for tuning options that have no dedicated flag. Wrap in single quotes so the inner double quotes survive the shell. |

The available keys depend on the detector:

* **SAM2** (`SAMV2Detector`): `score_metric` (`"predicted_iou"`), `points_per_side` (`64`),
  `points_per_batch` (`128`), `pred_iou_thresh` (`0.5`), `stability_score_thresh` (`0.7`),
  `stability_score_offset` (`0.7`), `crop_n_layers` (`1`), `box_nms_thresh` (`1`),
  `crop_n_points_downscale_factor` (`2`), `min_mask_region_area` (`25.0`), `use_m2m` (`true`),
  `model_cfg`.
* **SAM3** (`SAM3Detector`): `text_prompt` (`"tree"`), `bpe_path`.
* **DeepForest**, **Detectree2**, **TCD**: these constructors take only the model module, so there
  is little to override here.

Example:

```
--detector-kwargs '{"points_per_side": 32, "pred_iou_thresh": 0.6}'
```

## Output

One file, written to the path you give as `--predictions-save-path`. **If you omit that argument,
nothing is saved** — the detections are computed and discarded (useful only with
`--view-predictions-plot`). Parent directories are created if needed. The format is inferred from
the extension: use `.gpkg` for GeoPackage or `.geojson` for GeoJSON.

Everything is written in the CRS of the input raster, unless `--output-CRS` overrides it.

**The structure of the file depends on `--run-nms`**, and the two forms are not interchangeable.

### With `--run-nms`

A single flat layer: overlapping tiles are merged, duplicates suppressed, and low-confidence
detections dropped. This is the form you want for analysis or for loading into QGIS.

| Column | Description |
| --- | --- |
| `geometry` | Bounding box (DeepForest) or polygon (Detectree2, SAM2, SAM3, TCD) for each tree. |
| `score` | Confidence score. All rows satisfy `score >= --min-confidence`. |
| `unique_ID` | Identifier assigned during the merge. |
| `region_ID` | Zero-based index of the tile the detection came from. |

Detector-specific columns are carried through as well: SAM2 adds `predicted_iou` and
`stability_score`, and DeepForest adds `labels`.

### Without `--run-nms`

A GeoPackage with **two layers**, preserving the tile structure:

| Layer | Contents |
| --- | --- |
| `detections` | All detections from all tiles, each tagged with a `region_ID` column identifying its tile. Still contains the same tree detected once per overlapping chip. |
| `bounds` | The footprint of each tile, one row per tile with a matching `region_ID`. Tiles that produced no detections still appear here. |

This is the form [`postprocess`](postprocess.md) expects to read back, since reconstructing the
per-tile set is what allows boundary-aware cleanup. Note that `.geojson` cannot hold multiple
layers, so use `.gpkg` in this mode.

`--view-predictions-plot` writes nothing to disk; it opens a window and blocks until you close it.
