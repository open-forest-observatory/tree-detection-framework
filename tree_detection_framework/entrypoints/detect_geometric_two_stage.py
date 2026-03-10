import argparse
import json
from pathlib import Path
from typing import Optional
from math import ceil

import kornia.augmentation as K

from tree_detection_framework.detection.detector import (
    GeometricTreeCrownDetector,
    GeometricTreeTopDetector,
)
from tree_detection_framework.postprocessing.postprocessing import (
    multi_region_NMS,
    remove_edge_detections,
)
from tree_detection_framework.preprocessing.preprocessing import (
    create_dataloader,
    create_intersection_dataloader,
)
from tree_detection_framework.preprocessing.utils import KorniaTransformWrapper

CHIP_SIZE = 2000
CHIP_STRIDE = 1900
RESOLUTION = 0.2


def detect_trees_two_stage(
    CHM_file: Path,
    tree_tops_save_path: Path,
    tree_crowns_save_path: Path,
    chip_size: int = CHIP_SIZE,
    chip_stride: int = CHIP_STRIDE,
    resolution: float = RESOLUTION,
    raster_blur_sigma: Optional[float] = None,
    tree_top_detector_kwargs: dict = {},
    crown_segmentation_kwargs: dict = {},
):
    """Detect trees geometrically and save the detected tree tops and tree crowns.

    Args:
        CHM_file (Path):
            Path to a CHM file to detect trees from
        tree_tops_save_path (Path):
            Where to save the detected tree tops.
        tree_crowns_save_path (Path):
            Where to save the detected tree crowns.
        chip_size (int, optional):
            The size of the chip in pixels. Defaults to CHIP_SIZE.
        chip_stride (int, optional):
            The stride of the sliding chip window in pixels. Defaults to CHIP_STRIDE.
        resolution (float, optional):
            The spatial resolution that the CHM is resampled to. Defaults to OUTPUT_RESOLUTION.
        raster_blur_sigma (float, optional):
            The standard deviation of the 2D gaussian kernel, in meters. Defaults to None, no smoothing.
        tree_top_detector_kwargs (dict, optional):
            Keyword arguments to pass to the tree top detector. Defaults to {}.
        crown_segmentation_kwargs (dict, optional):
            Keyword arguments to pass to the crown segmentation approach. Defaults to {}.
    """
    # Stage 1: Create a dataloader for the raster data and detect the tree-tops

    if raster_blur_sigma is not None and raster_blur_sigma != 0:
        # Add a blurring operation to avoid spurious tree top detections
        # Compute the kernel sigma in pixels
        kernel_sigma_pixels = raster_blur_sigma / resolution
        # Set the kernel size at over two sigmas, which captures the vast majority of the probability density
        kernel_size = 2 * ceil(kernel_sigma_pixels) + 1

        # Create a gaussian blur operation. The kernel sigma is normally random, but we set the upper and lower
        # values to be identical. The probability is 1.0 so it's always applied. The wrapper ensures that
        # the shape of singleton batches is maintained.
        raster_transforms = KorniaTransformWrapper(
            K.AugmentationSequential(
                K.RandomGaussianBlur(
                    kernel_size=kernel_size,
                    sigma=(kernel_sigma_pixels, kernel_sigma_pixels),
                    p=1.0,
                ),
            )
        )
    else:
        raster_transforms = None

    # TODO, consider a larger window for tree detection to reduce boundary artificts, while still
    # keeping the watershed step fast. Counterpoint: in large rasters, the NMS step becomes extremely
    # expensive and the bottleneck, so additional duplicated regions may further slow that step.
    dataloader = create_dataloader(
        raster_folder_path=CHM_file,
        chip_size=chip_size,
        chip_stride=chip_stride,
        resolution=resolution,
        raster_transforms=raster_transforms,
    )

    # Create the detector for variable window maximum detection
    treetop_detector = GeometricTreeTopDetector(
        confidence_feature="distance", **tree_top_detector_kwargs
    )

    # Generate tree top predictions
    treetop_detections = treetop_detector.predict(dataloader)

    ## Remove the tree tops that were generated in the edges of tiles. This is an alternative to NMS.

    # Compute the suppresion distance so that each tile only contributes detections from its "core"
    # area. If there is only one tile, no suppression is needed.

    # TODO, a better alternative would be to suppress only overlapping regions of tiles,
    # such as the approach started here: https://github.com/open-forest-observatory/tree-detection-framework/pull/130
    suppression_distance = (
        0 if len(dataloader) == 1 else (chip_size - chip_stride) * resolution / 2
    )
    # Remove suppressed detections
    treetop_detections = remove_edge_detections(
        treetop_detections,
        suppression_distance=suppression_distance,
    )

    # Save the tree tops
    tree_tops_save_path.parent.mkdir(parents=True, exist_ok=True)
    treetop_detections.save(tree_tops_save_path)

    # Stage 2: Combine raster and vector data (from the tree-top detector) to create a new dataloader
    raster_vector_dataloader = create_intersection_dataloader(
        raster_data=CHM_file,
        vector_data=treetop_detections,
        chip_size=chip_size,
        chip_stride=chip_stride,
        resolution=resolution,
    )

    # Create the crown detector, which is seeded by the tree top points detected in the last step
    # The score metric is how far from the edge the detection is, which prioritizes central detections
    treecrown_detector = GeometricTreeCrownDetector(
        confidence_feature="distance", **crown_segmentation_kwargs
    )

    # Predict the crowns
    treecrown_detections = treecrown_detector.predict(raster_vector_dataloader)
    # Suppress overlapping crown predictions. This step can be slow.
    treecrown_detections = multi_region_NMS(
        treecrown_detections,
        confidence_column="score",
        intersection_method="IOS",
        run_per_region_NMS=False,
    )
    # Save the crowns
    tree_crowns_save_path.parent.mkdir(parents=True, exist_ok=True)
    treecrown_detections.save(tree_crowns_save_path)


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Detect trees geometrically and save the detected tree tops and tree crowns."
    )
    parser.add_argument(
        "CHM_file",
        type=Path,
        help="Path to a CHM file to detect trees from",
    )
    parser.add_argument(
        "tree_tops_save_path",
        type=Path,
        help="Where to save the detected tree tops.",
    )
    parser.add_argument(
        "tree_crowns_save_path",
        type=Path,
        help="Where to save the detected tree crowns.",
    )
    parser.add_argument(
        "--chip-size",
        type=int,
        default=CHIP_SIZE,
        help=f"The size of the chip in pixels. Defaults to {CHIP_SIZE}.",
    )
    parser.add_argument(
        "--chip-stride",
        type=int,
        default=CHIP_STRIDE,
        help=f"The stride of the sliding chip window in pixels. Defaults to {CHIP_STRIDE}.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=RESOLUTION,
        help=f"The spatial resolution that the CHM is resampled to. Defaults to {RESOLUTION}.",
    )
    parser.add_argument(
        "--raster-blur-sigma",
        type=float,
        help=f"The sigma in meters for a 2D Gaussian smoothing kernel. If unset, no smoothing occurs.",
    )
    parser.add_argument(
        "--tree-top-detector-kwargs",
        type=str,
        default="{}",
        help="A JSON string of keyword arguments to pass to the tree top detector.",
    )
    parser.add_argument(
        "--crown-segmentation-kwargs",
        type=str,
        default="{}",
        help="A JSON string of keyword arguments to pass to the crown segmentation approach.",
    )
    args = parser.parse_args()

    # You can't parse a dictionary with argparse, so we use json to convert the string
    args.tree_top_detector_kwargs = json.loads(args.tree_top_detector_kwargs)
    args.crown_segmentation_kwargs = json.loads(args.crown_segmentation_kwargs)

    return args


if __name__ == "__main__":
    args = parse_args()
    detect_trees_two_stage(
        CHM_file=args.CHM_file,
        tree_tops_save_path=args.tree_tops_save_path,
        tree_crowns_save_path=args.tree_crowns_save_path,
        chip_size=args.chip_size,
        chip_stride=args.chip_stride,
        resolution=args.resolution,
        raster_blur_sigma=args.raster_blur_sigma,
        tree_top_detector_kwargs=args.tree_top_detector_kwargs,
        crown_segmentation_kwargs=args.crown_segmentation_kwargs,
    )
