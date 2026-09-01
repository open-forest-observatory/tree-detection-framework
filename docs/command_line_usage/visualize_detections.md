# `visualize_detections`

Render detections on top of their source images and save the result as PNG files. This is a
visual-check tool: it samples a subset of the detection files, plots each one over its image, and
writes a figure per image.

It is designed to consume the output of [`detect_in_raw_images`](detect_in_raw_images.md), which
writes one `.gpkg` per image named after that image.

```
python -m tree_detection_framework.entrypoints.visualize_detections \
    <image_dir> <detection_dir> <out_dir> [options]
```

## Example

```
python -m tree_detection_framework.entrypoints.visualize_detections \
    data/raw_images \
    data/raw_image_detections \
    data/detection_figures \
    --n-images 10 \
    --step 5 \
    --image-ext JPG
```

## Required arguments

These are positional: give them in this order, with no flag in front.

| Position | Argument | Type | Description |
| --- | --- | --- | --- |
| 1 | `image_dir` | path | Directory containing the raw images. Must exist and be non-empty. |
| 2 | `detection_dir` | path | Directory containing the detection `.gpkg` files. Must exist and be non-empty. |
| 3 | `out_dir` | path | Directory to save the visualizations into. Created if it does not exist. |

## Optional arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--n-images` | integer | `5` | How many images to visualize. |
| `--step` | integer | `1` | Stride through the sorted list of detection files. `--step 5` visualizes every fifth file, which is useful for sampling across a whole flight rather than looking at consecutive, near-identical frames. |
| `--no-centroid` | flag | off | Do not draw the centroid of each detection. By default centroids are shown. |
| `--image-ext` | string | `JPG` | File extension of the source images, without the dot, e.g. `JPG` or `tif`. This is case-sensitive and must match the actual files on disk. |

## Matching images to detections

Detection files are paired to images by filename stem: `DJI_0123.gpkg` is paired with
`DJI_0123.<image-ext>`. Before doing any work, the script checks that every `.gpkg` in
`detection_dir` has a matching image, and exits with an error naming the offending file if one does
not. If you see that error, the usual cause is a wrong `--image-ext`.

## Output

One PNG per visualized image, named after the detection file's stem, written at 300 dpi into
`out_dir`.
