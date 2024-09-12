# tree-detection-framework

## Install
```
conda create -n tree-detection-framework python=3.10 -y
conda activate tree-detection-framework
# Install the dependencies that are hosted through pypi
poetry install
# Install detectron2 which cannot be installed easily
# https://detectron2.readthedocs.io/en/latest/tutorials/install.html#build-detectron2-from-source
pip install git+https://github.com/facebookresearch/detectron2.git
```
