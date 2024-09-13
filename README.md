# tree-detection-framework

## Install
First create and activate a conda environment
```
conda create -n tree-detection-framework python=3.10 -y
conda activate tree-detection-framework
```
Some of the dependencies are managed by a tool called [Poetry](https://python-poetry.org/). I've found
easiest to install this using the "official installer" option as follows
```
curl -sSL https://install.python-poetry.org | python3 -
```
Now, from the root directory of the project, run the following command. Note that on Jetstream2, you
may need to run this in a graphical session and respond to a keyring popup menu.
```
poetry install
```
Finally, the Detectron2 library is not compatible with `poetry` so must be installed directly with pip
```
# https://detectron2.readthedocs.io/en/latest/tutorials/install.html#build-detectron2-from-source
pip install git+https://github.com/facebookresearch/detectron2.git
```

## Structure
The module code will be developed in the `tree_detection_framework`. Initial prototyping of
 functionality can be done in the `sandbox` folder.