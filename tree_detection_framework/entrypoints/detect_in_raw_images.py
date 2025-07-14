import argparse
from pathlib import Path

import torch

from tree_detection_framework.detection.detector import (
    DeepForestDetector,
    Detectree2Detector,
)
from tree_detection_framework.detection.models import DeepForestModule, Detectree2Module
from tree_detection_framework.detection.SAM2_detector import SAMV2Detector
from tree_detection_framework.postprocessing.postprocessing import (
    multi_region_NMS,
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

    Raises:
        ValueError: If an unknown model_key is provided.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Detecting on device: {device}")

    # Create dataloader for raw images
    dataloader = create_image_dataloader(
        images_dir=image_dir,
        chip_size=chip_size,
        chip_stride=chip_stride,
        batch_size=batch_size,
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
    else:
        raise ValueError(f"Unknown model key: {model_key}")
    print("Model loaded")

    # Use predict_raw_drone_images to get per-image results
    detection_sets, files, bounds = detector.predict_raw_drone_images(dataloader)

    # For each image, remove detections that extend past the image bounds
    filtered_sets = remove_out_of_bounds_detections(detection_sets, bounds)
    # For each image, suppress overlapping detections
    filtered_sets = [multi_region_NMS(rds) for rds in filtered_sets]

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
    )
