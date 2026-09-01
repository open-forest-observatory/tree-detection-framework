# Installation

TDF is installed from source. The core dependencies are managed with
[Poetry](https://python-poetry.org/); the optional detection backends (Detectron2 for Detectree2,
SAM2, and SAM3) are installed separately because they are not compatible with Poetry's resolver.

If you would rather not install anything locally, see [Docker](#docker) below.

## Core installation

Install Poetry. The "official installer" option is easiest. Note that this should be run in the
base conda environment, or with no environment active.

```
curl -sSL https://install.python-poetry.org | python3 -
```

Create and activate a conda environment for the dependencies of this project:

```
conda create -n tree-detection-framework python=3.10 -y
conda activate tree-detection-framework
```

Now, from the root directory of the project, run the following. Note that on Jetstream2, you may
need to run this in a graphical session and respond to a keyring popup menu.

```
poetry install
```

This is enough to use the DeepForest detector and the geometric detector. The remaining sections
are only needed for the detector you intend to use.

## Optional: Detectree2

The Detectron2 library is not compatible with `poetry`, so it must be installed directly with pip.
See the [Detectron2 install
docs](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#build-detectron2-from-source)
for more detail.

```
pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git
```

Download the Detectree2 checkpoint weights:

```
cd checkpoints
mkdir detectree2
cd detectree2
wget https://zenodo.org/records/10522461/files/230103_randresize_full.pth
```

## Optional: SAM2

Clone the SAM2 repository and install the necessary config files:

```
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
```

Download the associated checkpoints:

```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

And move them into this repo:

```
mv checkpoints ../tree-detection-framework
```

## Optional: SAM3

Clone the SAM3 repository and install the dependencies. This can be done from any directory.

```
git clone https://github.com/facebookresearch/sam3.git && cd sam3

pip install -e .
```

!!! note
    `numpy` may be downgraded during installation due to dependency constraints. This is expected
    and does not impact other functionality in TDF.

Next, install the missing `decord` package:

```
pip install decord
```

To download the model checkpoint, you will need to log in to HuggingFace and request model access
from the [SAM3 repo](https://huggingface.co/facebook/sam3). Once access has been granted to your
account (usually within a couple of minutes), follow these steps:

* Go to <https://huggingface.co/settings/tokens>
* Select "Create new token"
* Select token type "Read"
* Enter a token name and create the token. Copy and store the token somewhere — you will not see it
  again.

The token is then passed to the SAM3 detector, either as the `huggingface_token` argument in Python
(see `examples/predict_detections_sam3.ipynb`) or as `--sam3-huggingface-token` on the command
line.

## Docker

An alternative to a local installation is the pre-built Docker image. It bundles the dependencies
for every supported detector — including Detectron2, SAM2, and SAM3, which are the fiddliest parts
of the local install — and ships with the Detectree2, SAM2, and TCD weights already downloaded.

The image is published to the GitHub Container Registry and can be pulled with:

```
docker pull ghcr.io/open-forest-observatory/tree-detection-framework:latest
```

The `latest` tag tracks the `main` branch. All available tags are listed on the
[packages](https://github.com/orgs/open-forest-observatory/packages?repo_name=tree-detection-framework)
tab of the repository.

Two detectors still need something at runtime:

* **DeepForest** downloads its own weights from GitHub the first time it runs, so the container
  needs network access.
* **SAM3** weights are gated behind a HuggingFace access request, so they must be mounted into the
  container or downloaded with a token.

See [Running with Docker](../command_line_usage/running_with_docker.md) for how to mount your data,
get results back out, and run each entrypoint from the image.

## Building the documentation

The documentation toolchain is deliberately kept out of `pyproject.toml` and `poetry.lock`, since
MkDocs is a build tool for this site rather than a dependency of the library. Keeping it out means
the lock file never churns for a documentation change, and the Docker image and CI test runs stay
free of it. Install it with pip, into the same environment created above:

```
pip install mkdocs-material "mkdocstrings[python]" mkdocs-awesome-pages-plugin \
            mkdocs-nav-weight mkdocs-git-revision-date-localized-plugin \
            mkdocs-git-committers-plugin-2
```

The same list is installed by the `gh-pages` GitHub Actions workflow that deploys this site, which
is the canonical copy of it.

!!! note
    The API Reference is generated by importing the code, so building the docs requires the full
    project environment created above — the packages listed here are not sufficient on their own.

To preview the site locally, with live reload:

```
mkdocs serve
```

To build the static site into the `site/` folder:

```
mkdocs build
```

## Use

The module code is in the `tree_detection_framework` folder. Once installed with the `poetry`
command above, this code can be imported into scripts or notebooks under the name
`tree_detection_framework`, the same as you would for any other library.
