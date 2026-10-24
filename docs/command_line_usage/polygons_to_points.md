# `polygons_to_points`

Convert tree crown polygons to one point per tree, optionally sampling tree heights from a canopy
height model (CHM).

```
python -m tree_detection_framework.entrypoints.polygons_to_points \
    --detections-path <PATH> \
    --output-path <PATH> \
    --method {centroid,chm_max} \
    [--chm-path <PATH>] \
    [--height-column <NAME>] \
    [--erosion-distance <NUMBER>]
```

## Example

```
python -m tree_detection_framework.entrypoints.polygons_to_points \
    --detections-path detections.gpkg \
    --output-path tree_tops.gpkg \
    --method chm_max \
    --chm-path data/emerald-point-chm/chm.tif
```

## Required arguments

| Argument | Type | Description |
| --- | --- | --- |
| `--detections-path` | path | Vector file of polygon detections, such as the output of [`generate_predictions`](generate_predictions.md). |
| `--output-path` | path | Where to write the point detections. The parent directory is created if needed. |
| `--method` | `centroid` or `chm_max` | How to derive each point. `centroid` uses the centre of the polygon. `chm_max` uses the location of the highest CHM pixel within the polygon, and also records that pixel's value as the height. |

## Optional arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--chm-path` | path | none | Path to the CHM raster. Required for `chm_max`. With `centroid`, if given, the CHM is sampled at each centroid to fill the height column. |
| `--height-column` | string | `height` | Name of the column to write heights into. The command fails if the detections already have a column with this name. |
| `--erosion-distance` | number | `0.0` | Distance, in the CHM's CRS units, to shrink each polygon inward before searching for the highest pixel. This excludes edge pixels that fall just outside the crown. Only used with `chm_max`. |

## Output

A single-layer file with one point per input polygon, with all the other columns of the input
carried through. When a CHM is used, the height column holds the sampled height. For `chm_max`,
polygons whose CHM coverage is entirely nodata fall back to their centroid with a height of NaN.

The original crown polygons are not saved in the output.
