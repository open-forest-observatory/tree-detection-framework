# tree-detection-framework
This project has three main goals:
* Enable tree detection on realistic-scale raster data with minimal boilerplate
* Facilitate direct comparison of multiple algorithms
* Rely on modern libraries and software best practice for a robust, performant, and modular tool

We use the `torchgeo` package to perform data loading and standardization using standard geospatial input formats. This library allows us to generate chips on the fly of a given size, stride, and spatial resolution. Training and inference is done with modular detectors that can be based on existing models and algorithms. We have preliminary support for using `PyTorch Lightning` to minimize boilerplate around model training and prediction. Region-level nonmax-suppression (NMS) is done using the `PolyGoneNMS` library which is efficient for large images. Visualization and saving of the predictions is done using `geopandas`, a common library for geospatial data.

This project is under active development by the [Open Forest Observatory](https://openforestobservatory.org/). We welcome contributions and suggestions for improvement.

## Other resources
There are a variety of projects for tree detection that you may find useful. This list is incomplete, so feel free to suggest additions.

### DeepForest
- [Github](https://github.com/weecology/DeepForest)
- Implements various preprocessing, postprocessing, visualization, and evaluation tasks.
- Used for RGB data with rectangular bounding box predictions.
- Provides a RetinaNet model trained on a large number of semi-supervised tree crown annotations and a smaller set of manual annotations.
- Training data is from the US only but represents diverse regions the model has been applied on data from outside the US successfully.
- Supports model fine-tuning with optional support for species/type classification
- Implemented in this framework.

### DetectTree2
- [Github](https://github.com/PatBall1/detectree2)
- Implements various preprocessing, postprocessing, visualization, and evaluation tasks.
- Used for RGB data with polygon boundaries.
- Provides a Mask R-CNN model train on a manually labeled tree crowns from four sites.
- Trained using data from tropical forests.
- Planned support within this framework

## Install
Some of the dependencies are managed by a tool called [Poetry](https://python-poetry.org/). I've found
easiest to install this using the "official installer" option as follows. Note that this should be run
in the base conda environment or with no environment active.
```
curl -sSL https://install.python-poetry.org | python3 -
```
Now create and activate a conda environment for the dependencies of this project.
```
conda create -n tree-detection-framework python=3.10 -y
conda activate tree-detection-framework
```

Now, from the root directory of the project, run the following command. Note that on Jetstream2, you
may need to run this in a graphical session and respond to a keyring popup menu.
```
poetry install
```
Finally, choose to either install the Detectron2 or SAM2 detection framework.

**Detectron2:** 
The Detectron2 library is not compatible with `poetry` so must be installed directly with pip
```
# https://detectron2.readthedocs.io/en/latest/tutorials/install.html#build-detectron2-from-source
pip install git+https://github.com/facebookresearch/detectron2.git
```

**SAM2:** 
Clone the SAM2 repository and install the necessary config files
```
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
```
And download the associated checkpoints
```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```
And move into this repo
```
mv checkpoints ../tree-detection-framework 
```


## Use
The module code is in the `tree_detection_framework` folder. Once installed using the `poetry`
command above, this code can be imported into scripts or notebooks under the name
 `tree_detection_framework` the same as you would for any other library.

## Examples
To begin with, you can access example geospatial data
[here](https://ucdavis.box.com/v/tdf-example-data), which should be downloaded and placed in the `data` folder at the top level of this project. Our goal is to have high-quality,
up-to-date examples in the `examples` folder. We also have work-in-progress or one-off code in
`sandbox`, which still may provide some insight but is not guaranteed to be current or generalizable.
Finally, the `tree_detection_framework/entrypoints` folder has command line scripts that can be run
to complete tasks.

## Evaluation and benchmark with NEON
Download the NEON dataset files and save the annotations and RGB folders under a new directory in the `data` folder.
```
wget -O annotations.zip "https://zenodo.org/records/5914554/files/annotations.zip?download=1"
unzip annotations.zip
wget -O evaluation.zip "https://zenodo.org/records/5914554/files/evaluation.zip?download=1"
unzip -j evaluation.zip "evaluation/RGB/*" -d RGB
rm annotations.zip
rm evaluation.zip
```
Follow the steps in `tree-detection-framework/sandbox/evaluation/neon_benchmark.ipynb` for detectors `DeepForest` & `Detectree2`, and `tree-detection-framework/sandbox/evaluation/sam2_neon_benchmark.ipynb` to use `SAM2`.

## Evaluation and benchmark with Detectree2 datasets
1. Download the site-specific .tif (for orthomosaic) and .gpkg (for ground truth polygons) files from https://zenodo.org/records/8136161.
2. Follow steps in https://github.com/PatBall1/detectree2/blob/master/notebooks/colab/tilingJB.ipynb to do the the tiling.
3. Add the tiled dataset to a new directory in the `data` folder.
4. For benchmark and evaluation see steps in `tree-detection-framework/sandbox/evaluation/dtree2_benchmark.ipynb` and `tree-detection-framework/sandbox/evaluation/sam2_dtree2_benchmark.ipynb` 