# `postprocess`

Apply a named chain of postprocessing steps to raw tree detections, then filter by minimum tree
height and write the final result.

This is the second half of a pipeline and is meant to be run after
[`detect_trees`](detect_trees.md). It reads `raw_detections.gpkg` from a detector directory and
writes `detections.gpkg` beside it.

```
python -m tree_detection_framework.entrypoints.postprocess \
    --detector-dir <PATH> \
    --preprocessed-local-files <JSON> \
    --min-tree-height <NUMBER> \
    --postprocessing-config-file <PATH> \
    [--postprocessing-id <ID>]
```

## Example

```
python -m tree_detection_framework.entrypoints.postprocess \
    --detector-dir runs/deepforest \
    --preprocessed-local-files '{"ortho": "data/emerald-point-ortho/ortho.tif", "chm": "data/emerald-point-chm/chm.tif"}' \
    --min-tree-height 5 \
    --postprocessing-config-file postprocessing_config.yaml \
    --postprocessing-id deepforest_default
```

## Required arguments

| Argument | Type | Description |
| --- | --- | --- |
| `--detector-dir` | path | Directory containing `raw_detections.gpkg`. `detections.gpkg` is written here. |
| `--preprocessed-local-files` | JSON string | Paths to the input rasters, as a JSON object. Only the `chm` key is used by this script, for height filtering and for converting crown polygons to points. |
| `--min-tree-height` | number | Minimum tree height in meters. Detections shorter than this are dropped. Applied last, to every detection, in both modes below. |
| `--postprocessing-config-file` | path | Path to the postprocessing config YAML that defines the named chains. This file is not shipped in the repository — you write it yourself, in the format shown below. |

## Optional arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--postprocessing-id` | string | `""` (empty) | Which named chain in the config file to apply. See the two modes below. |

## The two modes

**Empty `--postprocessing-id` (the geometric detector).** The raw detections are read as a plain
table and no chain is run — only the `--min-tree-height` filter is applied. The geometric detector
already produces clean points with a `height` column, so there is nothing to suppress.

**A named `--postprocessing-id` (the deep learning detectors).** The raw detections are read back
with their per-tile structure intact, the named chain is looked up in the config file and applied
step by step, and then the resulting crown polygons are converted to points using the `chm_max`
method — each crown becomes the point of maximum canopy height within it, which also assigns the
`height` used by the final filter. If the ID is not found in the config, the command fails and lists
the IDs that are available.

## Config file format

The config is a YAML mapping from an ID to a list of steps. Each step names a function from
`tree_detection_framework.postprocessing.postprocessing` and gives its arguments:

```yaml
deepforest_default:
  - name: suppress_tile_boundary_with_NMS
    args:
      threshold: 0.3
  - name: multi_region_NMS
    args:
      threshold: 0.3
      min_confidence: 0.3
  - name: filter_by_chm
    args:
      min_height: 5
```

Any function in the [postprocessing module](../API_reference/postprocessing.md) can be named here;
an unknown name causes an error. The `chm_path` argument for `filter_by_chm` is filled in
automatically from `--preprocessed-local-files`, so do not specify it in the YAML.

## Output

A single file, at a path derived from `--detector-dir` rather than one you pass in:

```
<the value you gave for --detector-dir>/detections.gpkg
```

So `--detector-dir runs/deepforest` writes `runs/deepforest/detections.gpkg`, alongside the
`raw_detections.gpkg` that [`detect_trees`](detect_trees.md) left there. The input file is not
modified or deleted, so you can re-run postprocessing with different settings without re-running
detection. The filename is hardcoded; the directory is created if needed.

The output is a **single-layer GeoPackage with one point per tree** — not the two-layer tiled
structure of the input. The per-tile layout is consumed and collapsed during postprocessing:

| Column | Description |
| --- | --- |
| `geometry` | A **point**, not a polygon. For the deep learning detectors each crown polygon is reduced to the location of maximum canopy height within it, using the CHM. |
| `height` | Tree height in meters, sampled from the CHM. Every row satisfies `height >= --min-tree-height`. |
| `score` | Confidence score carried through from the detector. |

Other columns from the raw detections are carried through as well; the intermediate
`crown_geometry` column is dropped, so the crown outlines do **not** survive into this file. If you
need the polygons, keep `raw_detections.gpkg`.

Progress is printed as the run proceeds — each postprocessor as it is applied with its arguments,
the count of raw detections loaded, the before-and-after counts for the height filter, and the final
path written.
