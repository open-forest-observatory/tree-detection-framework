import argparse
import json
from pathlib import Path

import geopandas as gpd
import yaml

import tree_detection_framework.postprocessing.postprocessing as postprocessors
from tree_detection_framework.detection.region_detections import RegionDetectionsSet
from tree_detection_framework.evaluation.evaluate import polygons_to_points


def postprocess(
    postprocessing_id: str,
    chm_path: str,
    min_tree_height: float,
    postproc_config_path: Path,
    detector_dir: Path | None = None,
    raw_detections_path: Path | None = None,
    output_path: Path | None = None,
):
    """
    Apply postprocessing chain to raw tree detections and save final detections as detections.gpkg
    in detector_dir. This script is meant to be run after detect_trees.py.

    Args:
        postprocessing_id: Postprocessing config ID that corresponds to a chain of postprocessing steps
            defined in `postproc_config_path`. An empty string for geometric detector would apply only height filtering.
        chm_path: Path to the CHM raster (used for height filtering and polygon-to-point conversion).
        min_tree_height: Minimum tree height (meters) used to filter detections.
        postproc_config_path: Path to postprocessing_config.yaml.
        detector_dir: Directory containing raw_detections.gpkg. detections.gpkg is written here.
            Used to derive raw_detections_path and output_path if they are not provided.
        raw_detections_path: Path to raw_detections.gpkg. If not provided, derived from detector_dir.
        output_path: Path to write detections.gpkg. If not provided, derived from detector_dir.
    """
    if raw_detections_path is None or output_path is None:
        if detector_dir is None:
            raise ValueError("detector_dir must be provided if raw_detections_path or output_path are not specified.")
    # detect_trees.py writes raw detections to detector_dir / "raw_detections.gpkg", so default to that if not provided 
    raw_detections_path = raw_detections_path or detector_dir / "raw_detections.gpkg"
    output_path = output_path or detector_dir / "detections.gpkg"

    # Geometric: skip chain, just height filter
    if not postprocessing_id:
        detections = gpd.read_file(raw_detections_path)
        print(f"[postprocess] Loaded {len(detections)} raw detections.")

    # CV detectors: full postprocessor chain
    else:
        result = RegionDetectionsSet.from_tiled_file(raw_detections_path)
        print(f"[postprocess] Loaded {len(result.get_data_frame())} raw detections.")

        with open(postproc_config_path) as f:
            full_config = yaml.safe_load(f)

        if postprocessing_id not in full_config:
            raise ValueError(
                f"postprocessing_id '{postprocessing_id}' not found in {postproc_config_path}. "
                f"Available keys: {list(full_config.keys())}"
            )

        steps = full_config[postprocessing_id]

        for step in steps:
            fn_name = step["name"]
            kwargs = step.get("args", {})

            fn = getattr(postprocessors, fn_name, None)
            if fn is None:
                raise ValueError(
                    f"Unknown postprocessor '{fn_name}'. "
                    f"Not found in tree_detection_framework.postprocessing.postprocessing."
                )

            # filter_by_chm needs chm_path which is not in the yaml
            if fn_name == "filter_by_chm":
                kwargs["chm_path"] = chm_path

            print(f"[postprocess] Applying {fn_name} with args {kwargs}")
            result = fn(result, **kwargs)

        # Convert polygons to points, sampling CHM max within each crown
        print(f"[postprocess] Converting polygons to points via chm_max")
        result = polygons_to_points(result, method="chm_max", chm_path=chm_path)
        detections = result.drop(columns="crown_geometry")

    # Apply minimum height filter and save final detections
    n_before = len(detections)
    detections = detections[detections["height"] >= min_tree_height]
    print(f"[postprocess] Height filter: {n_before} -> {len(detections)} trees")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detections.to_file(output_path)
    print(f"[postprocess] Saved {len(detections)} detections to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply postprocessing chain to raw tree detections."
    )
    parser.add_argument(
        "--postprocessing-id",
        default="",
        help="Postprocessing config ID. Empty string means geometric (height filter only).",
    )
    parser.add_argument(
        "--detector-dir",
        required=True,
        type=Path,
        help="Directory containing raw_detections.gpkg. detections.gpkg is written here.",
    )
    parser.add_argument(
        "--preprocessed-local-files",
        required=True,
        help="JSON string of preprocessed local file paths (used to get chm path).",
    )
    parser.add_argument(
        "--min-tree-height",
        required=True,
        type=float,
        help="Minimum tree height (meters) used to filter detections.",
    )
    parser.add_argument(
        "--postprocessing-config-file",
        required=True,
        help="Path to postprocessing_config.yaml.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    local_files = json.loads(args.preprocessed_local_files)
    postprocess(
        postprocessing_id=args.postprocessing_id,
        chm_path=local_files["chm"],
        min_tree_height=args.min_tree_height,
        postproc_config_path=Path(args.postprocessing_config_file),
        detector_dir=args.detector_dir,
    )
