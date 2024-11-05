import argparse
from typing import Optional

import pyproj

from tree_detection_framework.constants import BOUNDARY_TYPE, PATH_TYPE
from tree_detection_framework.detection.detector import DeepForestDetector
from tree_detection_framework.detection.models import DeepForestModule
from tree_detection_framework.preprocessing.preprocessing import create_dataloader


def generate_predictions(
    raster_folder_path: PATH_TYPE,
    chip_size: float,
    tree_detection_model: str,
    chip_stride: Optional[float] = None,
    chip_overlap_percentage: float = None,
    use_units_meters: bool = False,
    region_of_interest: Optional[BOUNDARY_TYPE] = None,
    output_resolution: Optional[float] = None,
    output_CRS: Optional[pyproj.CRS] = None,
    save_folder: Optional[PATH_TYPE] = None,
    view_predictions_plot: bool = False,
    batch_size: int = 1,
):

    # Create the dataloader by passing folder path to raster data.
    dataloader = create_dataloader(
        raster_folder_path=raster_folder_path,
        chip_size=chip_size,
        chip_stride=chip_stride,
        chip_overlap_percentage=chip_overlap_percentage,
        use_units_meters=use_units_meters,
        region_of_interest=region_of_interest,
        output_resolution=output_resolution,
        output_CRS=output_CRS,
        batch_size=batch_size,
    )

    # Setup the specified tree detection model
    if tree_detection_model == "deepforest":

        # Setup the parameters dictionary
        param_dict = {
            "backbone": "retinanet",
            "num_classes": 1,
        }

        df_module = DeepForestModule(param_dict)
        lightning_detector = DeepForestDetector(df_module)

    outputs = lightning_detector.predict(dataloader)

    if save_folder:
        outputs.save(save_folder)

    if view_predictions_plot is True:
        outputs.plot()


def parse_args() -> argparse.Namespace:
    description = (
        "This script generates tree detections for a given raster dataset. First, it creates a dataloader "
        + "with the tiled data and provides the images as input to the selected tree detection model. "
    )
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--raster-folder-path", required=True)
    parser.add_argument("--chip-size", type=float, required=True)
    parser.add_argument("--tree-detection-model", type=str, required=True)
    parser.add_argument("--chip-stride", type=float)
    parser.add_argument("--chip-overlap-percentage", type=float)
    parser.add_argument("--use-units-meters", action="store_true")
    parser.add_argument("--region-of-interest")
    parser.add_argument("--output-resolution", type=float)
    parser.add_argument("--output-CRS")
    parser.add_argument("--save-folder")
    parser.add_argument("--view-predictions-plot", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    generate_predictions(**args.__dict__)
