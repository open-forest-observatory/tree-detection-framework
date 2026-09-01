# Data

## Example data

The public example data can be downloaded [here](https://ucdavis.box.com/v/tdf-example-data). It
should be extracted into the `data` folder at the top level of this project. All of the notebooks in
the [`examples`](https://github.com/open-forest-observatory/tree-detection-framework/tree/main/examples)
folder read from this location, via the `DATA_FOLDER` constant in
[`tree_detection_framework.constants`](../API_reference/constants.md).

The example data covers the Emerald Point site and contains two rasters that the notebooks expect
at these paths:

| Path | Description |
| --- | --- |
| `data/emerald-point-ortho/ortho.tif` | RGB orthomosaic. Input to the DeepForest, Detectree2, SAM2, and SAM3 detectors. |
| `data/emerald-point-chm/chm.tif` | Canopy height model. Input to the geometric detector, and used to filter detections by height. |

## Using your own data

TDF accepts standard geospatial and image formats. There are two entry points for data, depending
on what you have:

**Georeferenced rasters (orthomosaics, CHMs).** Use
[`create_dataloader`](../API_reference/preprocessing/preprocessing.md), which accepts either a
single raster file or a folder of raster files. Because tiling is done on the fly by `torchgeo`, the
raster does not need to be pre-tiled; you specify `chip_size`, `chip_stride` (or
`chip_overlap_percentage`), and `resolution` and the data is resampled and chipped as it is read.
The coordinate reference system of the output is taken from the input unless you set `output_CRS`,
and you can restrict processing to a sub-area with `region_of_interest`. Rasters must be
georeferenced for the resulting detections to be placed in geospatial coordinates.

**Raw drone images.** Use
[`create_image_dataloader`](../API_reference/preprocessing/preprocessing.md), which reads all images
recursively from a directory. These images are not georeferenced, so detections are produced in
pixel coordinates relative to each image rather than in a geospatial CRS. This is the path used by
the [`detect_in_raw_images`](../command_line_usage/detect_in_raw_images.md) entrypoint.

**Vector labels.** For training or evaluation, `create_dataloader` also accepts
`vector_label_folder_path`, a folder of geospatial vector files (e.g. `.gpkg`, `.geojson`) that will
be rasterized into per-tile labels. `vector_label_attribute` selects which attribute to read from
the vector data, such as a class or instance ID.

## Benchmark datasets

The repository also includes work-in-progress benchmarking notebooks in the `sandbox/evaluation`
folder. These require additional datasets.

### NEON

Download the NEON dataset files and save the annotations and RGB folders under a new directory in
the `data` folder:

```
wget -O annotations.zip "https://zenodo.org/records/5914554/files/annotations.zip?download=1"
unzip annotations.zip
wget -O evaluation.zip "https://zenodo.org/records/5914554/files/evaluation.zip?download=1"
unzip -j evaluation.zip "evaluation/RGB/*" -d RGB
rm annotations.zip
rm evaluation.zip
```

Then follow the steps in `sandbox/evaluation/neon_benchmark.ipynb` for the DeepForest and Detectree2
detectors, and `sandbox/evaluation/sam2_neon_benchmark.ipynb` for SAM2.

### Detectree2

There are two ways to get this dataset:

* Download the site-specific `.tif` (orthomosaic) and `.gpkg` (ground truth polygons) files from
  <https://zenodo.org/records/8136161>, then follow the steps in the [Detectree2 tiling
  notebook](https://github.com/PatBall1/detectree2/blob/master/notebooks/colab/tilingJB.ipynb) to do
  the tiling.
* Or download our pre-tiled dataset from
  <https://ucdavis.box.com/s/thjmaane9d38opw1bhnyxrsrtt90j37m>.

Add the tiled dataset folder to the `data` folder in this repo, then see
`sandbox/evaluation/dtree2_benchmark.ipynb` and `sandbox/evaluation/sam2_dtree2_benchmark.ipynb`.

!!! note
    Code in `sandbox` is work-in-progress or one-off, and is not guaranteed to be current or
    generalizable. The `examples` folder is the maintained set of workflows.
