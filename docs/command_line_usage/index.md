---
weight: -10
title: Overview
---

# Command Line Usage

TDF can be run entirely from a terminal, without writing any Python. This section documents every
command line script in the
[`tree_detection_framework/entrypoints`](https://github.com/open-forest-observatory/tree-detection-framework/tree/main/tree_detection_framework/entrypoints)
folder: what each command does, every flag it accepts, what each flag does, and which ones are
required.

## How to run a script

The entrypoints are not installed as standalone commands. Each one is run as a Python module with
the `-m` flag, from any directory, with the conda environment created during
[installation](../getting_started/installation.md) activated:

```
conda activate tree-detection-framework
python -m tree_detection_framework.entrypoints.<script_name> [arguments]
```

For example:

```
python -m tree_detection_framework.entrypoints.generate_predictions \
    --raster-folder-path data/emerald-point-ortho \
    --chip-size 1024 \
    --tree-detection-model deepforest \
    --predictions-save-path detections.gpkg \
    --run-nms
```

Every script accepts `--help`, which prints the same information from the script itself:

```
python -m tree_detection_framework.entrypoints.generate_predictions --help
```

If you would rather not install anything, all of these commands can be run from the pre-built
Docker image instead. See [Running with Docker](running_with_docker.md).

## Reading these pages

Each argument is listed with:

* **Required** — the command will fail without it.
* **Optional** — the command runs without it, using the default shown.

Two conventions appear throughout:

* **Positional arguments** are given in order, with no `--flag` in front of them. For example
  `python -m ... detect_in_raw_images /path/to/images /path/to/output deepforest`.
* **Flags** (arguments declared with `action="store_true"`, such as `--use-units-meters`) take no
  value. Including the flag turns the option on; leaving it out leaves it off.

A few arguments expect a **JSON string**. Because a command line cannot carry a dictionary, these
are passed as quoted JSON, for example:

```
--detector-kwargs '{"points_per_side": 32, "pred_iou_thresh": 0.6}'
```

Wrap the whole thing in single quotes so that the double quotes inside the JSON survive the shell.

## The scripts

### Running a detection

| Script | Use it when |
| --- | --- |
| [`generate_predictions`](generate_predictions.md) | You have an orthomosaic (or other georeferenced raster) and want tree detections from DeepForest, Detectree2, SAM2, SAM3, or TCD. This is the main detection command. |
| [`detect_geometric_two_stage`](detect_geometric_two_stage.md) | You have a canopy height model and want the learning-free geometric detector, which produces tree tops and optionally tree crowns. |
| [`detect_in_raw_images`](detect_in_raw_images.md) | You have a folder of raw (non-georeferenced) drone images rather than an orthomosaic. |

### Inspecting inputs and outputs

| Script | Use it when |
| --- | --- |
| [`tile_data`](tile_data.md) | You want to check how your raster will be chipped, or export the chips to disk, before running a detector. |
| [`visualize_detections`](visualize_detections.md) | You want to render detections on top of raw images as PNG files. |

### Pipeline steps

These two scripts are designed to be called in sequence by an automated pipeline rather than typed
by hand. They pass structured information as JSON strings and use fixed filenames
(`raw_detections.gpkg`, `detections.gpkg`) inside a shared output directory.

| Script | Use it when |
| --- | --- |
| [`detect_trees`](detect_trees.md) | You are driving detection from a pipeline and want one command that dispatches to any detector, including the geometric one. |
| [`postprocess`](postprocess.md) | You want to apply a named, YAML-configured chain of postprocessing steps to the output of `detect_trees`. |
