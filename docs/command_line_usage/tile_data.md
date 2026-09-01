# `tile_data`

Create a dataloader from raster data and inspect it: display a sample of the chips it produces,
and/or write the chips out to disk. No detection is performed.

This is a preprocessing check. Use it to confirm that your chip size, stride, and resolution give
tiles at a sensible scale for the trees in your imagery before spending GPU time on a detection run,
or to export chips for use elsewhere.

```
python -m tree_detection_framework.entrypoints.tile_data \
    --raster-folder-path <PATH> --chip-size <NUMBER> [options]
```

## Example

Show three sample tiles:

```
python -m tree_detection_framework.entrypoints.tile_data \
    --raster-folder-path data/emerald-point-ortho \
    --chip-size 1024 \
    --chip-stride 768 \
    --resolution 0.05 \
    --visualize-n-tiles 3
```

Save 50 randomly sampled tiles to disk:

```
python -m tree_detection_framework.entrypoints.tile_data \
    --raster-folder-path data/emerald-point-ortho \
    --chip-size 1024 \
    --chip-stride 768 \
    --resolution 0.05 \
    --save-folder data/tiles \
    --save-n-tiles 50 \
    --random-sample
```

## Required arguments

| Argument | Type | Description |
| --- | --- | --- |
| `--raster-folder-path` | path | Path to a raster file, or to a folder of raster files. |
| `--chip-size` | number | Size of each square chip. Interpreted as pixels unless `--use-units-meters` is given. |

## Tiling options

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--chip-stride` | number | none | Distance between the start of one chip and the next. Do not set this together with `--chip-overlap-percentage`. |
| `--chip-overlap-percentage` | number | none | Overlap between neighboring chips, from 0 to 100. Do not set this together with `--chip-stride`. |
| `--use-units-meters` | flag | off | Interpret `--chip-size` and `--chip-stride` in meters on the ground rather than in pixels. |
| `--resolution` | number | resolution of the input | Ground sample distance in meters per pixel that the raster is resampled to. If unset, the native resolution of the first raster read is used. |
| `--output-CRS` | string | CRS of the input | Coordinate reference system for the output data, e.g. `EPSG:32610`. If unset, the CRS of the first tile is used. |
| `--region-of-interest` | — | none | Restrict the dataloader to a spatial sub-region. See the caveat below before using this. |
| `--batch-size` | integer | see caveat below | Number of images loaded in a batch. |

## Label options

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--vector-label-folder-path` | path | none | Folder of geospatial vector files to use as labels. If unset, the dataloader is unlabeled. |
| `--vector-label-attribute` | string | `treeID` | Which attribute to read from the vector data, such as the class or instance ID. Only meaningful together with `--vector-label-folder-path`. |

## Visualization options

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--visualize-n-tiles` | integer | none | Display this many randomly sampled tiles. If unset, nothing is displayed. Requires a display; not useful on a headless machine. |

## Saving options

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--save-folder` | path | none | Folder to write tiles into. Created if it does not exist. If unset, nothing is saved. |
| `--save-n-tiles` | integer | all tiles | How many tiles to save. If unset, every tile in the dataloader is written, which can be a very large number of files for a big raster. |
| `--random-sample` | flag | off | When `--save-n-tiles` is set, sample the tiles randomly from across the raster instead of taking the first N from the beginning. Has no effect unless `--save-n-tiles` is given. |

!!! warning "Two rough edges in this script"
    * **`--region-of-interest`** is documented in the script's own `--help` as taking
      `minx,miny,maxx,maxy`, but the value is forwarded as a plain string to a parameter that
      expects a path to a vector file or a geometry object. A comma-separated bounding box will not
      be interpreted as one. Restrict the region in Python, or by passing a vector file, until this
      is fixed.
    * **`--batch-size`** has no argparse default, so leaving it out passes `None` through rather than
      falling back to the documented default of `1`. Pass `--batch-size 1` explicitly.
