# `detect_geometric_two_stage`

Detect trees geometrically from a canopy height model (CHM) and save the detected tree tops, and
optionally the tree crowns. This uses the learning-free algorithm implemented within TDF, so no
model weights or GPU are needed.

It runs the same two-stage workflow as the
[sequential geometric detector notebook](../examples_and_applications/examples.md#two-stage-geometric-detection-from-a-canopy-height-model):
stage one finds tree tops with a variable window filter, and stage two grows crowns outward from
those tree tops. Stage two runs only if `--tree-crowns-save-path` is given.

```
python -m tree_detection_framework.entrypoints.detect_geometric_two_stage \
    <CHM_file> <tree_tops_save_path> [options]
```

## Example

```
python -m tree_detection_framework.entrypoints.detect_geometric_two_stage \
    data/emerald-point-chm/chm.tif \
    data/tree_tops.gpkg \
    --tree-crowns-save-path data/tree_crowns.gpkg \
    --resolution 0.2 \
    --raster-blur-sigma 0.5 \
    --edge-suppression-meters 5
```

## Required arguments

These are positional: give them in this order, with no flag in front.

| Position | Argument | Type | Description |
| --- | --- | --- | --- |
| 1 | `CHM_file` | path | Path to the canopy height model raster to detect trees from. |
| 2 | `tree_tops_save_path` | path | Where to save the detected tree tops. Parent directories are created if needed. |

## Optional arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--tree-crowns-save-path` | path | none | Where to save the detected tree crowns. **If this is not given, stage two is skipped entirely and no crowns are detected** — only tree tops are produced. |
| `--chip-size` | integer | `2000` | Size in pixels of each square chip the CHM is cut into. |
| `--chip-stride` | integer | `1900` | Distance in pixels between the start of one chip and the next. The default leaves a 100 px overlap between neighboring chips. |
| `--resolution` | number | `0.2` | Ground sample distance in meters per pixel that the CHM is resampled to. |
| `--raster-blur-sigma` | number | none (no smoothing) | Standard deviation, in meters, of a 2D Gaussian smoothing kernel applied to the CHM before finding local maxima. Smoothing suppresses spurious tree tops caused by noise or branch-level detail in the CHM. Values around 0.5 m are a reasonable starting point. |
| `--edge-suppression-meters` | number | none | Discard any tree whose top falls within this distance of the edge of the valid raster area. Trees at the raster boundary are often only partly visible ("shoulders" of trees mostly outside the raster) and detect poorly. When crowns are also being produced, crowns belonging to suppressed tree tops are dropped too. |
| `--tree-top-detector-kwargs` | JSON string | `{}` | Extra named arguments for the tree top detector. See below. |
| `--crown-segmentation-kwargs` | JSON string | `{}` | Extra named arguments for the crown segmentation step. See below. |

## Tuning the tree top detector

Passed via `--tree-top-detector-kwargs`, as a quoted JSON object.

The variable window filter searches for local maxima in the CHM using a window whose radius depends
on the height at that location, computed as `a·h² + b·h + c`. A larger window at greater heights is
what prevents a single tall, wide crown from being split into several detections.

| Key | Default | Description |
| --- | --- | --- |
| `a` | `0` | Coefficient for the quadratic term in the window radius calculation. |
| `b` | `0.0325` | Coefficient for the linear term in the window radius calculation. |
| `c` | `0.25` | Constant term in the window radius calculation. |
| `min_ht` | `5` | Minimum height, in the units of the CHM, for a pixel to be considered part of a tree. |
| `filter_shape` | `"circle"` | Shape of the local maximum search window. One of `"circle"`, `"square"`, `"none"`. |
| `confidence_feature` | `"distance"` | Which feature to use as the confidence score for each detection. |

!!! note
    `--raster-blur-sigma` sets the detector's `blur_sigma`, so do not also pass `blur_sigma` here.

Example:

```
--tree-top-detector-kwargs '{"min_ht": 3, "b": 0.04}'
```

## Tuning crown segmentation

Passed via `--crown-segmentation-kwargs`, as a quoted JSON object. Only relevant when
`--tree-crowns-save-path` is given.

| Key | Default | Description |
| --- | --- | --- |
| `approach` | `"watershed"` | Crown delineation algorithm. `"watershed"` or `"silva"`. |
| `radius_factor` | `0.6` | Factor determining the crown radius. Used by the `"silva"` approach only. |
| `threshold_factor` | `0.3` | Factor determining the threshold for the binary crown mask. Used by the `"silva"` approach only. |
| `tree_height_column` | `"height"` | Name of the column holding tree height in the tree top detections. |
| `min_height` | `5` | Minimum height for a crown to be produced. |
| `simplify_tolerance` | `2.0` | Tolerance for simplifying the crown polygon boundary. Larger values give smoother, lower-vertex-count polygons. |
| `contour_backend` | `"cv2"` | Which backend to use to trace the crown boundary. |

!!! note
    `confidence_feature` is set to `"distance"` by this entrypoint so that crowns whose tree top sits
    closer to the center of its tile are preferred when duplicates are resolved. Overriding it here
    will change that behavior.

## Output

Between one and two files, written to the paths you gave as arguments. Nothing else is written, and
no directory structure is created beyond the parent folders of those two paths (which are created
if they do not exist).

| File | Path | When |
| --- | --- | --- |
| Tree tops | the second positional argument, `tree_tops_save_path` | always |
| Tree crowns | the value of `--tree-crowns-save-path` | only if that flag is given |

The file format is inferred from the extension you use, via GeoPandas' `to_file`. Use `.gpkg` for
GeoPackage or `.geojson` for GeoJSON. Both files are written in the CRS of the input CHM.

**Tree tops** — one **point** per detected tree, with these columns:

| Column | Description |
| --- | --- |
| `geometry` | Point at the detected treetop location. |
| `height` | Height of the tree at that point, in the units of the CHM (normally meters). |
| `score` | Confidence score. By default this is the `"distance"` feature, meaning distance from the tile edge, so detections nearer the middle of their tile score higher. |
| `unique_ID` | Identifier assigned when the per-tile detections are merged. This is the key that the crowns file refers back to. |
| `region_ID` | Zero-based index of the tile the detection came from. |

If `--edge-suppression-meters` was given, tree tops within that distance of the valid raster edge
have already been removed from this file.

**Tree crowns** — one **polygon** per tree top, with these columns:

| Column | Description |
| --- | --- |
| `geometry` | Polygon boundary of the crown. |
| `height` | Height of the tree top that seeded this crown. |
| `score` | Confidence score, `"distance"` by default. |
| `treetop_unique_ID` | The `unique_ID` of the tree top that seeded this crown. |

Exactly one crown is kept per tree top — where several tiles produced a crown for the same tree,
the one whose tree top sat closest to the center of its tile wins. Because `treetop_unique_ID`
matches `unique_ID` in the tree tops file, the two can be joined on that key to get height and
crown geometry together. If `--edge-suppression-meters` was given, crowns belonging to suppressed
tree tops are dropped from this file too, so the two files stay consistent.
