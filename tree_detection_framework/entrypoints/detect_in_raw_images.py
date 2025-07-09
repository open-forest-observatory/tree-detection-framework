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


def main():
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
        help="Path to detectree checkpoints, see repo README. E.g."
        " 230103_randresize_full.pth. Only used if model_key == detectree",
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

    assert args.image_dir.is_dir()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Detecting on device: {device}")

    # Create dataloader for raw images
    dataloader = create_image_dataloader(
        images_dir=args.image_dir,
        chip_size=args.chip_size,
        chip_stride=args.chip_stride,
        batch_size=args.batch_size,
    )
    print("Dataloader created")

    # Instantiate the model
    if args.model_key == "deepforest":
        model_params = {"backbone": "retinanet", "num_classes": 1}
        module = DeepForestModule(model_params)
        module.to(device)
        detector = DeepForestDetector(module)
    elif args.model_key == "detectree2":
        assert args.detectree_checkpoints.is_file()
        model_params = {"update_model": str(args.detectree_checkpoints)}
        module = Detectree2Module(model_params)
        detector = Detectree2Detector(module)
    elif args.model_key == "sam2":
        detector = SAMV2Detector(device=device)
    else:
        raise ValueError(f"Unknown model key: {args.model_key}")
    print("Model loaded")

    # Use predict_raw_drone_images to get per-image results
    detection_sets, files, bounds = detector.predict_raw_drone_images(dataloader)

    # Filter detections
    filtered_sets = remove_out_of_bounds_detections(detection_sets, bounds)
    filtered_sets = [multi_region_NMS(rds) for rds in filtered_sets]

    # Save results for each image
    for rds, file in zip(filtered_sets, files):
        out_path = args.out_dir / (Path(file).stem + ".gpkg")
        rds.save(out_path)
    print(f"Detection regions saved to {args.out_dir}")


if __name__ == "__main__":
    main()
