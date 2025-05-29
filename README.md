# tree-detection-framework
This project has three main goals:
* Enable tree detection on realistic-scale, geospatial raster data with minimal boilerplate, using existing (external) tree detection/segmentation models
* Facilitate direct comparison of multiple algorithms
* Rely on modern libraries and software best practice for a robust, performant, and modular tool

This project does not, itself, provide tree detection/segmentation algorithms (with the exception of a geometric algorithm). Instead, it provides a standardized interface for performing training, inference, and evaluation using existing tree detection models and algorithms. The project currently supports the external computer vision models DeepForest, Dectree2, and SAM2, as well as a geometric canopy height model segmentor implemented within TDF. Support for other external models can be added by implementing a new `Detector` class.

We use the `torchgeo` package to perform data loading and standardization using standard geospatial input formats. This library allows us to generate chips on the fly of a given size, stride, and spatial resolution. Training and inference is done with modular detectors that can be based on existing models and algorithms. We have preliminary support for using `PyTorch Lightning` to minimize boilerplate around model training and prediction. Region-level nonmax-suppression (NMS) is done using the `PolyGoneNMS` library which is efficient for large images. Visualization and saving of the predictions is done using `geopandas`, a common library for geospatial data.

This project is under active development by the [Open Forest Observatory](https://openforestobservatory.org/). We welcome contributions and suggestions for improvement.

## Tree detection models supported
TDF currently supports the following tree detection/segmentation algorithms.

### DeepForest
- [Github](https://github.com/weecology/DeepForest)
- Uses RGB input data. Predicts tree crowns with rectangular bounding boxes.
- Provides a RetinaNet model trained on a large number of semi-supervised tree crown annotations and a smaller set of manual annotations.
- Trained using data from the US only but representing diverse regions. The model has been applied on data from outside the US successfully.

### Detectree2
- [Github](https://github.com/PatBall1/detectree2)
- Uses RGB input data. Predicts tree crowns with polygon boundaries.
- Provides a Mask R-CNN model trained on manually labeled tree crowns from four sites.
- Trained using data from tropical forests.

### Segment Anything Model 2 (SAM2)
- [Github](https://github.com/facebookresearch/sam2)
- Uses RGB input data. Predicts objects with polygon boundaries.
- Utilizes the Segment Anything Model (SAM 2.1 Hiera Large) checkpoint with tuned parameters for mask generation optimized for tree crown delineation.
- Does not rely on supervised training for tree-specific data but generalizes well due to SAM's zero-shot nature; however, non-tree objects are also detected and included in predictions.

### Geometric Detector
- Implementation of the variable window filter algorithm of [Popescu and Wynne
  (2004)](https://www.ingentaconnect.com/content/asprs/pers/2004/00000070/00000005/art00003) for
  tree top detection, combined with the algorithm of [Silva et al.
  (2016)](https://www.tandfonline.com/doi/full/10.1080/07038992.2016.1196582#abstract) for crown
  segmentation.
- Uses canopy height model (CHM) input data. Predicts tree crowns with polygon boundaries.
- This is a learning-free tree detection algorithm. It is the one algorithm that is implemented within TDF as opposed to relying on an existing external model/algorithm.

## Software Architecture
The `tree-detection-framework` is organized into modular components to facilitate extension and integration of different detection models. The main components are:

1. **`preprocessing.py`**<br>
   The `create_dataloader()` method accepts single/multiple orthomosaic inputs. Alternatively,
   `create_image_datalaoder()` accepts a folder containing raw drone imagery. The methods tile the
   input images based on user-specified parameters such as tile size, stride, and resolution and
   return a PyTorch-compatible dataloader for inference.
2. **`Detector` Base Class**<br>
   All detectors in the framework (e.g., DeepForestDetector, Detectree2Detector) inherit from the
   `Detector` base class. The base class defines the core logic for generating predictions and
   geospatially referencing image tiles, while model-specific detectors translate the inputs to the
   format expected by the respective model. This design allows all detectors to plug into the same
   pipeline with minimal code changes.
3. **`RegionDetectionsSet` and `RegionDetections`**<br>
   These classes standardize model outputs. A `RegionDetectionsSet` is a collection of `RegionDetections`, where each `RegionDetections` object represents the detections in a single image tile. This abstraction allows postprocessing components to operate uniformly across different detectors. These outputs can be saved out as `.gpkg` or `.geojson` files.
4. **`postprocessing.py`**<br>
   Impelments a set of postprocessing functions for cleaning the detections by Non-Maximum Suppression(NMS), polygon hole suppression, tile boundary suppression, and removing out of bounds detections. Most of these methods operate on standardized output types (`RegionDetections` / `RegionDetectionsSet`).

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
Download the detectree2 checkpoint weights.
```
cd checkpoints
mkdir detectree2
cd detectree2
wget https://zenodo.org/records/10522461/files/230103_randresize_full.pth
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
1. Download the dataset. There are two ways to get the dataset:
    Download the site-specific .tif (for orthomosaic) and .gpkg (for ground truth polygons) files from https://zenodo.org/records/8136161. Then, follow steps in https://github.com/PatBall1/detectree2/blob/master/notebooks/colab/tilingJB.ipynb to do the the tiling.
    (OR)
    Download our pre-tiled dataset from https://ucdavis.box.com/s/thjmaane9d38opw1bhnyxrsrtt90j37m 
3. Add the tiled dataset folder to the `data` folder in this repo.
4. For benchmark and evaluation see steps in `tree-detection-framework/sandbox/evaluation/dtree2_benchmark.ipynb` and `tree-detection-framework/sandbox/evaluation/sam2_dtree2_benchmark.ipynb` 
