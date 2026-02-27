from pathlib import Path
import argparse

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
        output_resolution (float, optional):
            The spatial resolution that the CHM is resampled to. Defaults to OUTPUT_RESOLUTION.
    """
    # Stage 1: Create a dataloader for the raster data and detect the tree-tops
    dataloader = create_dataloader(
        raster_folder_path=CHM_file,
        chip_size=chip_size,
        chip_stride=chip_stride,
        resolution=resolution,
    )

    # Create the detector for variable window maximum detection
    treetop_detector = GeometricTreeTopDetector(
        a=0, b=0.0325, c=0.25, confidence_feature="distance"
    )

    # Generate tree top predictions
    treetop_detections = treetop_detector.predict(dataloader)

    ## Remove the tree tops that were generated in the edges of tiles. This is an alternative to NMS.

    # Compute the suppresion distance so that each tile only contributes detections from its "core"
    # area. If there is only one tile, no suppression is needed.
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
    treecrown_detector = GeometricTreeCrownDetector(confidence_feature="distance")

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
        "--chip_size",
        type=int,
        default=CHIP_SIZE,
        help=f"The size of the chip in pixels. Defaults to {CHIP_SIZE}.",
    )
    parser.add_argument(
        "--chip_stride",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detect_trees_two_stage(
        CHM_file=args.CHM_file,
        tree_tops_save_path=args.tree_tops_save_path,
        tree_crowns_save_path=args.tree_crowns_save_path,
        chip_size=args.chip_size,
        chip_stride=args.chip_stride,
        resolution=args.resolution,
    )
