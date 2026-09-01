# Running with Docker

Every command in this section can be run from the pre-built Docker image instead of a local
installation. This avoids installing Poetry, conda, Detectron2, SAM2, and SAM3 on your own machine,
and it is the quickest way to run a detector on a new system.

```
docker pull ghcr.io/open-forest-observatory/tree-detection-framework:latest
```

## The general form

The image sets `WORKDIR /app` and `PYTHONPATH=/app`, with the TDF source at
`/app/tree_detection_framework`. So the `python -m ...` commands documented on the other pages in
this section work unchanged — you just prefix them with `docker run` and append them to the image
name:

```
docker run --rm --gpus all \
    -v /path/on/your/machine:/data \
    ghcr.io/open-forest-observatory/tree-detection-framework:latest \
    python -m tree_detection_framework.entrypoints.<script_name> [arguments]
```

Anything after the image name replaces the image's default command, so any entrypoint can be
selected this way.

### What each Docker flag does

| Flag | Required | What it does |
| --- | --- | --- |
| `--rm` | no | Delete the container when the command finishes. Without it, every run leaves a stopped container behind. |
| `--gpus all` | no | Make the host's NVIDIA GPUs visible inside the container. Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host. Omit it to run on CPU, which works but is much slower for the deep learning detectors. |
| `-v HOST:CONTAINER` | **yes** | Mount a host directory into the container. Without at least one of these, the container cannot see your data and anything it writes disappears when it exits. |
| `--user $(id -u):$(id -g)` | no | Run as your own user so output files are owned by you. See [File ownership](#file-ownership) below. |

## Getting data in and results out

The container has its own filesystem. Your orthomosaics are not in it, and files it writes are gone
when it exits unless they land in a mounted directory.

The simplest approach is one mount for everything, using paths inside the container in the TDF
arguments:

```
docker run --rm --gpus all \
    -v "$PWD/data:/data" \
    ghcr.io/open-forest-observatory/tree-detection-framework:latest \
    python -m tree_detection_framework.entrypoints.generate_predictions \
        --raster-folder-path /data/emerald-point-ortho \
        --chip-size 1024 \
        --chip-stride 768 \
        --resolution 0.05 \
        --tree-detection-model deepforest \
        --run-nms \
        --predictions-save-path /data/detections.gpkg
```

Note that `--raster-folder-path` and `--predictions-save-path` are `/data/...`, the paths *inside*
the container, not the host paths. This is the most common mistake: passing a host path that does
not exist in the container gives a confusing "file not found" for a file you can plainly see.

To keep inputs safe, mount them read-only and give outputs their own writable mount:

```
docker run --rm --gpus all \
    -v "$PWD/data:/data:ro" \
    -v "$PWD/outputs:/outputs" \
    ghcr.io/open-forest-observatory/tree-detection-framework:latest \
    python -m tree_detection_framework.entrypoints.generate_predictions \
        --raster-folder-path /data/emerald-point-ortho \
        --chip-size 1024 \
        --tree-detection-model deepforest \
        --predictions-save-path /outputs/detections.gpkg
```

### File ownership

By default the container runs as root, so files it writes into a mounted directory are owned by
root on the host and you may not be able to delete or edit them without `sudo`. Passing your own
user and group ID avoids this:

```
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -v "$PWD/outputs:/outputs" \
    ...
```

## Model weights

The image bundles the dependencies for every supported detector. Weights for most of them are
already baked in at `/app/checkpoints`, which is exactly where TDF's default checkpoint paths look,
so nothing extra is needed:

| Detector | Bundled in the image? | Notes |
| --- | --- | --- |
| `deepforest` | downloaded on first use | The DeepForest package fetches its own weights from GitHub the first time it runs, so the container needs network access. They are re-downloaded on every fresh container unless you mount a cache. |
| `detectree2` | yes | `/app/checkpoints/230103_randresize_full.pth` |
| `sam2` | yes | `/app/checkpoints/sam2.1_hiera_large.pt` |
| `tcd` | yes | Cached in the image's HuggingFace cache. |
| `geometric` | not applicable | Learning-free; no weights at all. |
| `sam3` | **no** | Must be supplied at runtime. See below. |

SAM3 weights are gated behind a HuggingFace access request and so are not included. Either mount a
checkpoint you already have:

```
docker run --rm --gpus all \
    -v "$PWD/data:/data" -v "$PWD/checkpoints:/checkpoints" \
    ghcr.io/open-forest-observatory/tree-detection-framework:latest \
    python -m tree_detection_framework.entrypoints.generate_predictions \
        --raster-folder-path /data/ortho.tif --chip-size 2000 \
        --tree-detection-model sam3 \
        --sam3-checkpoint-path /checkpoints/sam3.pt \
        --predictions-save-path /data/sam3_detections.gpkg
```

or let it download them, passing a token via an environment variable so the token does not appear in
your shell history or in `docker ps` output:

```
docker run --rm --gpus all -e HF_TOKEN \
    -v "$PWD/data:/data" \
    ghcr.io/open-forest-observatory/tree-detection-framework:latest \
    python -m tree_detection_framework.entrypoints.generate_predictions \
        --raster-folder-path /data/ortho.tif --chip-size 2000 \
        --tree-detection-model sam3 \
        --sam3-huggingface-token "$HF_TOKEN" \
        --predictions-save-path /data/sam3_detections.gpkg
```

With `-e HF_TOKEN` (no value), Docker passes through the variable of that name from your shell.

## More examples

Geometric detection from a canopy height model, which needs no GPU:

```
docker run --rm -v "$PWD/data:/data" \
    ghcr.io/open-forest-observatory/tree-detection-framework:latest \
    python -m tree_detection_framework.entrypoints.detect_geometric_two_stage \
        /data/emerald-point-chm/chm.tif \
        /data/tree_tops.gpkg \
        --tree-crowns-save-path /data/tree_crowns.gpkg \
        --resolution 0.2 \
        --raster-blur-sigma 0.5
```

Export tiles to inspect how a raster will be chipped:

```
docker run --rm -v "$PWD/data:/data" \
    ghcr.io/open-forest-observatory/tree-detection-framework:latest \
    python -m tree_detection_framework.entrypoints.tile_data \
        --raster-folder-path /data/emerald-point-ortho \
        --chip-size 1024 --chip-stride 768 --resolution 0.05 \
        --save-folder /data/tiles --save-n-tiles 20 --random-sample
```

Read a script's own help text:

```
docker run --rm ghcr.io/open-forest-observatory/tree-detection-framework:latest \
    python -m tree_detection_framework.entrypoints.generate_predictions --help
```

Open a shell in the container to poke around:

```
docker run --rm -it -v "$PWD/data:/data" \
    ghcr.io/open-forest-observatory/tree-detection-framework:latest bash
```

## Things that do not work in a container

* **Anything that opens a plot window.** `--view-predictions-plot` on
  [`generate_predictions`](generate_predictions.md) and `--visualize-n-tiles` on
  [`tile_data`](tile_data.md) both need a display the container does not have. Write results to a
  file and view them on the host instead, or use
  [`visualize_detections`](visualize_detections.md), which saves PNGs rather than opening a window.
* **Host paths in TDF arguments.** Every path you pass to a TDF script must be a path inside the
  container, matching the right-hand side of a `-v` mount.

## The default command

The image declares a default command:

```
CMD python /app/tree_detection_framework/entrypoints/generate_predictions.py
```

Running the image with no arguments therefore invokes `generate_predictions` with no arguments,
which exits with an error about missing required arguments. This is expected — always append the
command you actually want, as shown above.
