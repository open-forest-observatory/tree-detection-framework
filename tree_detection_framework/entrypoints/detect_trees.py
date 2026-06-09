import argparse
import json
from pathlib import Path

from tree_detection_framework.constants import CHECKPOINTS_FOLDER
from tree_detection_framework.entrypoints.detect_geometric_two_stage import (
    CHIP_SIZE,
    CHIP_STRIDE,
    RESOLUTION,
    detect_trees_two_stage,
)
from tree_detection_framework.entrypoints.generate_predictions import (
    generate_predictions,
)


def detect_trees(
    detector: str,
    detection_params: dict,
    detector_dir: Path,
    preprocessed_local_files: dict,
):
    """
    Detect trees using the specified detector and save raw detections.

    Args:
        detector: Detector name: geometric, deepforest, detectree2, sam2, sam3, or tcd.
        detection_params: Detection parameters (chip_size, chip_stride, resolution, batch_size, detectree2_weights_path, sam3_checkpoint_path, etc.).
        detector_dir: Directory for detector outputs. raw_detections.gpkg is written here.
        preprocessed_local_files: Preprocessed local file paths (should have keys "ortho" and "chm").
    """
    ortho_path = preprocessed_local_files["ortho"]
    chm_path = preprocessed_local_files["chm"]
    output_path = detector_dir / "raw_detections.gpkg"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if detector == "geometric":
        detect_trees_two_stage(
            CHM_file=Path(chm_path),
            tree_tops_save_path=output_path,  # add tree_crowns_save_path if you want to save tree crowns too
            chip_size=int(detection_params.get("chip_size", CHIP_SIZE)),
            chip_stride=int(detection_params.get("chip_stride", CHIP_STRIDE)),
            resolution=float(detection_params.get("resolution", RESOLUTION)),
        )

    else:
        # CV-based detectors: deepforest, detectree2, sam2, sam3, tcd
        generate_predictions(
            raster_folder_path=ortho_path,
            chip_size=float(detection_params["chip_size"]),
            chip_stride=float(detection_params["chip_stride"]),
            tree_detection_model=detector,
            resolution=(
                float(detection_params["resolution"])
                if detection_params.get("resolution")
                else None
            ),
            predictions_save_path=str(output_path),
            run_nms=False,
            batch_size=(
                int(detection_params["batch_size"])
                if detection_params.get("batch_size")
                else None
            ),
            detectree2_weights_path=detection_params.get(
                "detectree2_weights_path",
                Path(CHECKPOINTS_FOLDER, "230103_randresize_full.pth"),
            ),
            sam3_checkpoint_path=detection_params.get(
                "sam3_checkpoint_path",
                Path(CHECKPOINTS_FOLDER, "sam3.pt"),
            ),
        )

    print(f"[detect] Detections saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect trees using the specified detector and save raw detections."
    )
    parser.add_argument(
        "--detector",
        required=True,
        choices=["geometric", "deepforest", "detectree2", "sam2", "sam3", "tcd"],
        help="Detector name: geometric, deepforest, detectree2, sam2, sam3, or tcd.",
    )
    parser.add_argument(
        "--detection-params-json",
        required=True,
        help="JSON string of detection parameters (chip_size, chip_stride, resolution, batch_size, etc.).",
    )
    parser.add_argument(
        "--detector-dir",
        required=True,
        type=Path,
        help="Directory for detector outputs. raw_detections.gpkg is written here.",
    )
    parser.add_argument(
        "--preprocessed-local-files",
        required=True,
        help="JSON string of preprocessed local file paths (ortho, chm, etc.).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detect_trees(
        detector=args.detector,
        detection_params=json.loads(args.detection_params_json),
        detector_dir=args.detector_dir,
        preprocessed_local_files=json.loads(args.preprocessed_local_files),
    )
