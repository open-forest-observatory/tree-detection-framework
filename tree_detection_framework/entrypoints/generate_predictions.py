import argparse
import logging
from typing import Optional

import pyproj
import torch

from tree_detection_framework.constants import BOUNDARY_TYPE, PATH_TYPE
from tree_detection_framework.detection.detector import (
    DeepForestDetector,
    Detectree2Detector,
)
from tree_detection_framework.detection.models import DeepForestModule, Detectree2Module
from tree_detection_framework.postprocessing.postprocessing import multi_region_NMS
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
    predictions_save_path: Optional[PATH_TYPE] = None,
    view_predictions_plot: bool = False,
    run_nms: bool = True,
    iou_threshold: Optional[float] = 0.3,
    min_confidence: Optional[float] = 0.3,
    batch_size: int = 1,
):
    """
    Entrypoint script to generate tree detections for a raster dataset input. Supports visualizing and saving predictions.

    Args:
        raster_folder_path (PATH_TYPE): Path to the folder or raster files.
        chip_size (float):
            Dimension of the chip. May be pixels or meters, based on `use_units_meters`.
        chip_stride (Optional[float], optional):
            Stride of the chip. May be pixels or meters, based on `use_units_meters`. If used,
            `chip_overlap_percentage` should not be set. Defaults to None.
        tree_detection_model (str):
            Selected model for detecting trees.
        chip_overlap_percentage (Optional[float], optional):
            Percent overlap of the chip from 0-100. If used, `chip_stride` should not be set.
            Defaults to None.
        use_units_meters (bool, optional):
            Use units of meters rather than pixels when interpreting the `chip_size` and `chip_stride`.
            Defaults to False.
        region_of_interest (Optional[BOUNDARY_TYPE], optional):
            Only data from this spatial region will be included in the dataloader. Defaults to None.
        output_resolution (Optional[float], optional):
            Spatial resolution the data in meters/pixel. If un-set, will be the resolution of the
            first raster data that is read. Defaults to None.
        output_CRS: (Optional[pyproj.CRS], optional):
            The coordinate reference system to use for the output data. If un-set, will be the CRS
            of the first tile found. Defaults to None.
        predictions_save_path (Optional[PATH_TYPE], optional):
            Path to a geofile to save the prediction outputs.
        view_predictions_plot (bool, optional):
            Set to True if visualization of the detected regions is needed. Defaults to False.
        run_nms: (bool, optional):
            Set to True if non-max suppresion needs to be run on predictions from multiple regions.
        iou_threshold (float, optional):
            What intersection over union value to consider an overlapping detection. Defaults to 0.5.
        min_confidence (float, optional):
            Prediction score threshold for detections to be included.
        batch_size (int, optional):
            Number of images to load in a batch. Defaults to 1.
    """

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
        # Move the module to the GPU if available
        df_module.to(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        lightning_detector = DeepForestDetector(df_module)

    elif tree_detection_model == "detectree2":

        # Load detectree2 pretrained weights
        # TODO: download pretrained weights when called, instead of providing a local path
        trained_model = (
            "/ofo-share/repos-amritha/detectree2-code/230103_randresize_full.pth"
        )
        param_dict = {"update_model": trained_model}

        dtree2_module = Detectree2Module(param_dict)
        lightning_detector = Detectree2Detector(dtree2_module)

    else:
        raise ValueError(
            """Please enter a valid tree detection model. Currently supported models are: 
                1. deepforest
                2. detectree2"""
        )

    # Get predictions by invoking the tree_detection_model
    logging.info("Getting tree detections")
    outputs = lightning_detector.predict(dataloader)

    if run_nms is True:
        logging.info("Running non-max suppression")
        # Run non-max suppression on the detected regions
        outputs = multi_region_NMS(
            outputs, iou_theshold=iou_threshold, min_confidence=min_confidence
        )

    if predictions_save_path:
        # Save predictions to disk
        outputs.save(predictions_save_path)

    if view_predictions_plot is True:
        logging.info("View plot. Kill the plot window to exit.")
        # Plot the detections and the bounds of the region
        outputs.plot()


def parse_args() -> argparse.Namespace:
    description = (
        "This script generates tree detections for a given raster image. First, it creates a dataloader "
        + "with the tiled raster dataset and provides the images as input to the selected tree detection model. "
        + "All of the arguments are passed to "
        + "tree_detection_framework.entrypoints.generate_predictions "
        + "which has the following documentation:\n\n"
        + generate_predictions.__doc__
    )
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--raster-folder-path", required=True)
    parser.add_argument("--chip-size", type=float, required=True)
    parser.add_argument("--tree-detection-model", type=str, required=True)
    parser.add_argument("--chip-stride", type=float)
    parser.add_argument("--chip-overlap-percentage", type=float)
    parser.add_argument("--use-units-meters", action="store_true")
    parser.add_argument("--region-of-interest")
    parser.add_argument("--output-resolution", type=float)
    parser.add_argument("--output-CRS")
    parser.add_argument("--predictions-save-path")
    parser.add_argument("--view-predictions-plot", action="store_true")
    parser.add_argument("--run-nms", action="store_true")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=1)

    try:
        args = parser.parse_args()

    except SystemExit as e:
        print("\nError: Missing required arguments.")
        parser.print_help()
        raise e

    return args


if __name__ == "__main__":
    args = parse_args()
    generate_predictions(**args.__dict__)
