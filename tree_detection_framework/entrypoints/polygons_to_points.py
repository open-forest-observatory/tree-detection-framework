import argparse
from pathlib import Path
from typing import Optional

import geopandas as gpd

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.evaluation.evaluate import (
    polygons_to_points as convert_polygons_to_points,
)


def polygons_to_points(
    detections_path: PATH_TYPE,
    output_path: PATH_TYPE,
    method: str,
    chm_path: Optional[PATH_TYPE] = None,
    height_column: str = "height",
    erosion_distance: float = 0.0,
):
    """
    Convert tree crown polygons to one representative point per tree and save the result.

    Args:
        detections_path (PATH_TYPE): Path to a vector file of polygon detections.
        output_path (PATH_TYPE): Path to write the point detections to.
        method (str): Options: "centroid" or "chm_max". Method used to derive each point. "centroid"
            uses the polygon centroid, and samples the CHM there if `chm_path` is given. "chm_max"
            uses the location of the highest CHM pixel within the polygon, and requires `chm_path`.
        chm_path (Optional[PATH_TYPE], optional): Path to a canopy height model raster. Required
            for "chm_max". Defaults to None.
        height_column (str, optional): Name of the column to write sampled heights into. Must not
            already exist in the detections. Defaults to "height".
        erosion_distance (float, optional): Distance (in CHM CRS units) to erode each polygon
            inward before finding the maximum CHM pixel. Only used for "chm_max". Defaults to 0.0.
    """
    detections = gpd.read_file(detections_path)
    print(f"[polygons_to_points] Loaded {len(detections)} detections.")

    result = convert_polygons_to_points(
        detections,
        method=method,
        chm_path=None if chm_path is None else str(chm_path),
        height_column=height_column,
        erosion_distance=erosion_distance,
    )
    # The crown polygons are a second geometry column, which can't be written to the same layer
    result = result.drop(columns="crown_geometry")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_file(output_path)
    print(f"[polygons_to_points] Saved {len(result)} points to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert tree crown polygons to points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--detections-path",
        required=True,
        type=Path,
        help="Path to a vector file of polygon detections.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=Path,
        help="Path to write the point detections to.",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["centroid", "chm_max"],
        help="Method used to derive each point from its polygon.",
    )
    parser.add_argument(
        "--chm-path",
        type=Path,
        help="Path to a canopy height model raster. Required for chm_max.",
    )
    parser.add_argument(
        "--height-column",
        default="height",
        help="Name of the column to write sampled heights into.",
    )
    parser.add_argument(
        "--erosion-distance",
        type=float,
        default=0.0,
        help="Distance (in CHM CRS units) to erode each polygon inward before finding the "
        "maximum CHM pixel. Only used for chm_max.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    polygons_to_points(**vars(args))
