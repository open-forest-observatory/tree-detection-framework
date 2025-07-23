import argparse
import logging
import warnings
from pathlib import Path

import torch
from tqdm import tqdm

from tree_detection_framework.detection.detector import (
    DeepForestDetector,
    Detectree2Detector,
)
from tree_detection_framework.detection.models import DeepForestModule, Detectree2Module
from tree_detection_framework.detection.SAM2_detector import SAMV2Detector
from tree_detection_framework.postprocessing.postprocessing import (
    multi_region_NMS,
    remove_edge_detections,
    remove_out_of_bounds_detections,
)
from tree_detection_framework.preprocessing.preprocessing import create_image_dataloader

MODEL_KEYS = ["detectree2", "deepforest", "sam2"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect trees in raw images using a specified model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image_dir", type=Path, help="Directory containing raw images.")
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Directory to save detection results. Will be created if it does"
        " not already exist.",
    )
    parser.add_argument(
        "model_key", type=str, choices=MODEL_KEYS, help=f"Model to use: {MODEL_KEYS}"
    )
    parser.add_argument(
        "--detectree-checkpoints",
        type=Path,
        help="Path to detectree checkpoints, see repo README."
        " E.g. 230103_randresize_full.pth. Only used if model_key == detectree2",
    )
    parser.add_argument(
        "--chip_size", type=int, default=2200, help="Chip size for tiling images"
    )
    parser.add_argument(
        "--chip_stride",
        type=int,
        default=2000,
        help="Chip stride for tiling images",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for inference"
    )
    parser.add_argument(
        "--downsample-factor",
        type=float,
        default=1.0,
        help="Downsample factor for input images (default: 1.0, no downsampling).",
    )
    parser.add_argument(
        "--filter-edges",
        action="store_true",
        help="If set, filter detections at the edges of tiles.",
    )
    args = parser.parse_args()

    assert args.image_dir.is_dir(), f"image_dir {args.image_dir} is not a directory"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.model_key == "detectree2":
        assert (
            args.detectree_checkpoints is not None
        ), "--detectree-checkpoints required for detectree2 model"
        assert args.detectree_checkpoints.is_file()
    return args


def main(
    image_dir: Path,
    out_dir: Path,
    model_key: str,
    detectree_checkpoints: Path,
    chip_size: int,
    chip_stride: int,
    batch_size: int,
    downsample_factor: float,
    filter_edges: bool,
) -> None:
    """
    Detect trees in raw images using a specified model and save detection results.

    Args:
        image_dir (Path): Directory containing raw images.
        out_dir (Path): Directory to save detection results. Will be created if it does not already exist.
        model_key (str): Model to use for detection.
        detectree_checkpoints (Path or None): Path to detectree2 checkpoints. Required if model_key is 'detectree2'.
        chip_size (int): Chip size for tiling images.
        chip_stride (int): Chip stride for tiling images.
        batch_size (int): Batch size for inference.
        downsample_factor (float): Downsample factor for input images. 1.0 means no downsampling;
            values >1.0 reduce image size by this factor. Default is 1.0.
        filter_edges (bool): If True, filter detections at the edges of tiles.

    Raises:
        ValueError: If an unknown model_key is provided.
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Detecting on device: {device}")

    # Create dataloader for raw images
    dataloader = create_image_dataloader(
        images_dir=image_dir,
        chip_size=chip_size,
        chip_stride=chip_stride,
        batch_size=batch_size,
        downsample_factor=downsample_factor,
    )
    print("Dataloader created")

    # Instantiate the model
    if model_key == "deepforest":
        model_params = {"backbone": "retinanet", "num_classes": 1}
        module = DeepForestModule(model_params)
        module.to(device)
        detector = DeepForestDetector(module)
    elif model_key == "detectree2":
        model_params = {"update_model": str(detectree_checkpoints)}
        module = Detectree2Module(model_params)
        detector = Detectree2Detector(module)
    elif model_key == "sam2":
        detector = SAMV2Detector(device=device)
        warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'")
        logging.getLogger().setLevel(logging.WARNING)
    else:
        raise ValueError(f"Unknown model key: {model_key}")
    print("Model loaded")

    # Use predict_raw_drone_images to get per-image results
    detection_sets, files, bounds = detector.predict_raw_drone_images(dataloader)

    # For each image, remove detections that extend past the image bounds
    filtered_sets = remove_out_of_bounds_detections(
        detection_sets, bounds, verbose=True
    )
    # If requested, remove detections that get too close to the tile boundaries
    if filter_edges:
        filtered_sets = [
            remove_edge_detections(
                rds,
                suppression_distance=chip_size / 20,
                verbose=True,
            )
            for rds in tqdm(filtered_sets, desc="Removing edge detections")
        ]
    # For each image, suppress overlapping detections. Note that this collapses the
    # list of RegionDetectionsSet into a list of RegionDetections
    filtered_sets = [
        multi_region_NMS(rd) for rd in tqdm(filtered_sets, desc="Multi region NMS")
    ]

    # Save results for each image
    for rds, file in zip(filtered_sets, files):
        out_path = out_dir / (Path(file).stem + ".gpkg")
        rds.save(out_path)
    print(f"Detection regions saved to {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        model_key=args.model_key,
        detectree_checkpoints=args.detectree_checkpoints,
        chip_size=args.chip_size,
        chip_stride=args.chip_stride,
        batch_size=args.batch_size,
        downsample_factor=args.downsample_factor,
        filter_edges=args.filter_edges,
    )
