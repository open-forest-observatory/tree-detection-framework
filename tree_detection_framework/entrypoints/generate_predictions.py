import argparse
from typing import Optional

import pyproj

from tree_detection_framework.constants import ARRAY_TYPE, BOUNDARY_TYPE, PATH_TYPE
from tree_detection_framework.detection.detector import DeepForestDetector
from tree_detection_framework.detection.models import DeepForestModule

from tree_detection_framework.preprocessing.preprocessing import create_dataloader

def generate_predictions(
    raster_folder_path: PATH_TYPE,
    chip_size: float,
    chip_stride: Optional[float] = None,
    chip_overlap_percentage: float = None,
    use_units_meters: bool = False,
    region_of_interest: Optional[BOUNDARY_TYPE] = None,
    output_resolution: Optional[float] = None,
    output_CRS: Optional[pyproj.CRS] = None,
    tree_detection_model: Optional[str] = None,
    save_folder: Optional[PATH_TYPE] = None,
    view_predictions_plot: bool = False,
    batch_size: int = 1,
):

    # Create the dataloader by passing folder path to raster data and optionally a path to the vector data folder.
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

    if tree_detection_model == 'deepforest':

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

    
    