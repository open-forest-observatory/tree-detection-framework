import argparse
import json
from pathlib import Path
from typing import Optional

from tree_detection_framework.detection.detector import (
    GeometricTreeCrownDetector,
    GeometricTreeTopDetector,
)
from tree_detection_framework.postprocessing.postprocessing import (
    multi_region_NMS,
    remove_detections_from_tile_overlap,
)
from tree_detection_framework.preprocessing.preprocessing import (
    create_dataloader,
    create_intersection_dataloader,
)
from tree_detection_framework.utils.raster import get_valid_raster_region

CHIP_SIZE = 2000
CHIP_STRIDE = 1900
RESOLUTION = 0.2


def detect_trees_two_stage(
    CHM_file: Path,
    tree_tops_save_path: Path,
    tree_crowns_save_path: Optional[Path] = None,
    chip_size: int = CHIP_SIZE,
    chip_stride: int = CHIP_STRIDE,
    resolution: float = RESOLUTION,
    raster_blur_sigma: Optional[float] = None,
    edge_suppression_meters: Optional[float] = None,
    tree_top_detector_kwargs: dict = {},
    crown_segmentation_kwargs: dict = {},
):
    """Detect trees geometrically and save the detected tree tops and tree crowns.

    Args:
        CHM_file (Path):
            Path to a CHM file to detect trees from
        tree_tops_save_path (Path):
            Where to save the detected tree tops.
        tree_crowns_save_path (Path, optional):
            Where to save the detected tree crowns. If not provided, no crowns will be detected. Defaults to None.
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
    # TODO, consider a larger window for tree detection to reduce boundary artificts, while still
    # keeping the watershed step fast. Counterpoint: in large rasters, the NMS step becomes extremely
    # expensive and the bottleneck, so additional duplicated regions may further slow that step.
    dataloader = create_dataloader(
        raster_folder_path=CHM_file,
        chip_size=chip_size,
        chip_stride=chip_stride,
        resolution=resolution,
    )

    # Create the detector for variable window maximum detection
    treetop_detector = GeometricTreeTopDetector(
        confidence_feature="distance",
        blur_sigma=raster_blur_sigma,
        **tree_top_detector_kwargs,
    )

    # Generate tree top predictions
    treetop_detections = treetop_detector.predict(dataloader)

    # Remove the tree tops that were generated in the edges of tiles. This is an alternative to NMS.
    treetop_detections = remove_detections_from_tile_overlap(treetop_detections)

    # Detect tree crowns if requested
    if tree_crowns_save_path is not None:
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
            confidence_feature="distance",
            **crown_segmentation_kwargs,
        )

        # Predict the crowns
        treecrown_detections = treecrown_detector.predict(raster_vector_dataloader)
        treecrown_detections_gdf = treecrown_detections.get_data_frame(merge=True)
        treecrown_detections_gdf = treecrown_detections_gdf.sort_values(
            "score", ascending=False
        )
        treecrown_detections_gdf = treecrown_detections_gdf.groupby(
            "treetop_unique_ID"
        ).first()

    # Convert to the geodataframe representation
    treetop_detections_gdf = treetop_detections.get_data_frame(merge=True)

    # If requested, suppress the trees detected at the boundary of the raster. These can often be
    # low-quality and represent the "shoulder" of trees which are partly outside the raster.
    if edge_suppression_meters is not None and edge_suppression_meters != 0:
        # Determine the extent of the valid raster
        valid_raster_region = get_valid_raster_region(CHM_file)
        # Construct the new valid region eroded from the edge by the requested amount
        valid_raster_region.geometry = valid_raster_region.buffer(
            -edge_suppression_meters
        )
        # Make sure the CRS matches
        valid_raster_region.to_crs(treetop_detections_gdf.crs, inplace=True)

        # Subset tree tops to this valid region
        treetop_detections_gdf = treetop_detections_gdf[
            treetop_detections_gdf.within(valid_raster_region.geometry.values[0])
        ]

    # Save the tree tops
    tree_tops_save_path.parent.mkdir(parents=True, exist_ok=True)
    treetop_detections_gdf.to_file(tree_tops_save_path)

    if tree_crowns_save_path is not None:
        # Drop the crowns corresponding to trees detected at the edge by ensuring crowns correspond
        # to a tree top which was kept
        if edge_suppression_meters is not None and edge_suppression_meters != 0:
            treecrown_detections_gdf = treecrown_detections_gdf[
                treecrown_detections_gdf.treetop_unique_ID.isin(
                    treetop_detections_gdf.unique_ID
                )
            ]
        # Save the crowns
        tree_crowns_save_path.parent.mkdir(parents=True, exist_ok=True)
        treecrown_detections_gdf.to_file(tree_crowns_save_path)


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
        "--tree-crowns-save-path",
        type=Path,
        help="Where to save the detected tree crowns. If not provided, no crowns will be detected.",
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
        "--edge-suppression-meters",
        type=float,
        help="Suppress all trees with treetops within this distance of the edge of the raster.",
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
        edge_suppression_meters=args.edge_suppression_meters,
        tree_top_detector_kwargs=args.tree_top_detector_kwargs,
        crown_segmentation_kwargs=args.crown_segmentation_kwargs,
    )
