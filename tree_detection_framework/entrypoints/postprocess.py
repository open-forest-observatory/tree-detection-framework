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
    detector_dir: Path,
    chm_path: str,
    min_tree_height: float,
    postproc_config_path: Path,
):
    raw_detections_path = detector_dir / "raw_detections.gpkg"
    output_path         = detector_dir / "detections.gpkg"

    # Geometric: skip chain, just height filter
    if not postprocessing_id:
        detections = gpd.read_file(raw_detections_path)
        print(f"[postprocess] Loaded {len(detections)} raw detections.")
        n_before   = len(detections)
        detections = detections[detections["height"] >= min_tree_height]
        print(f"[postprocess] Height filter: {n_before} -> {len(detections)} trees")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        detections.to_file(output_path)
        print(f"[postprocess] Saved {len(detections)} detections to {output_path}")

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
            kwargs  = step.get("args", {})

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

        n_before = len(result)
        result   = result[result["height"] >= min_tree_height]
        print(f"[postprocess] Height filter: {n_before} -> {len(result)} trees")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = result.drop(columns="crown_geometry")
        result.to_file(output_path)
        print(f"[postprocess] Saved {len(result)} detections to {output_path}")


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
        detector_dir=args.detector_dir,
        chm_path=local_files["chm"],
        min_tree_height=args.min_tree_height,
        postproc_config_path=Path(args.postprocessing_config_file),
    )
