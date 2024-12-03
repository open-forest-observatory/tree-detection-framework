import logging
from typing import Optional

import numpy as np
import pyproj
from polygone_nms import nms
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon

from tree_detection_framework.detection.region_detections import (
    RegionDetections,
    RegionDetectionsSet,
)


def single_region_NMS(
    detections: RegionDetections,
    iou_theshold: float = 0.5,
    confidence_column: str = "score",
    min_confidence: float = 0.3,
) -> RegionDetections:
    """Run non-max suppresion on predictions from a single region.

    Args:
        detections (RegionDetections):
            Detections from a single region to run NMS on.
        iou_threshold (float, optional):
            What intersection over union value to consider an overlapping detection. Defaults to 0.5.
        confidence_column (str, optional):
            Which column in the dataframe to use as a confidence for NMS. Defaults to "score"
        min_confidence (float, optional):
            Prediction score threshold for detections to be included.

    Returns:
        RegionDetections:
            NMS-suppressed set of detections
    """
    # Extract the geodataframe for the detections
    detections_df = detections.get_data_frame()

    # Determine which detections are high enough confidence to retain
    high_conf_inds = np.where(
        (detections_df[confidence_column] >= min_confidence).to_numpy()
    )[0]

    # Filter detections based on minimum confidence score
    detections_df = detections_df.iloc[high_conf_inds]
    if detections_df.empty:
        # Return empty if no detections pass threshold
        return detections.subset_detections([])

    ## Get the polygons for each detection object
    polygons = detections_df.geometry.to_list()
    # Extract the score
    confidences = detections_df[confidence_column].to_numpy()

    # Put the data into the required format, list[(polygon, class, confidence)]
    # TODO consider adding a class, currently set to all ones
    input_data = list(zip(polygons, np.ones_like(confidences), confidences))

    # Run polygon NMS
    keep_inds = nms(
        input_data=input_data,
        distributed=None,
        nms_method="Default",
        intersection_method="IOU",
        threshold=iou_theshold,
    )

    # We only performed NMS on the high-confidence detections, but we need the indices w.r.t. the
    # original data with all detections. Sort for convenience so data is not permuted.
    keep_inds_in_original = sorted(high_conf_inds[keep_inds])
    # Extract the detections that were kept
    subset_region_detections = detections.subset_detections(keep_inds_in_original)

    return subset_region_detections


def multi_region_NMS(
    detections: RegionDetectionsSet,
    run_per_region_NMS: bool = True,
    iou_theshold: float = 0.5,
    confidence_column: str = "score",
    min_confidence: float = 0.3,
) -> RegionDetections:
    """Run non-max suppresion on predictions from multiple regions.

    Args:
        detections (RegionDetectionsSet):
            Detections from multiple regions to run NMS on.
        run_per_region_NMS (bool):
            Should nonmax-suppression be run on each region before the regions are merged. This may
            lead to a speedup if there is a large amount of within-region overlap. Defaults to True.
        iou_threshold (float, optional):
            What intersection over union value to consider an overlapping detection. Defaults to 0.5.
        confidence_column (str, optional):
            Which column in the dataframe to use as a confidence for NMS. Defaults to "score"

        min_confidence (float, optional):
            Prediction score threshold for detections to be included.
    Returns:
        RegionDetections:
            NMS-suppressed set of detections, merged together for the set of regions.
    """
    # Determine whether to run NMS individually on each region.
    if run_per_region_NMS:
        # Run NMS on each sub-region and then wrap this in a region detection set
        detections = RegionDetectionsSet(
            [
                single_region_NMS(
                    region_detections,
                    iou_theshold=iou_theshold,
                    confidence_column=confidence_column,
                    min_confidence=min_confidence,
                )
                for region_detections in detections.region_detections
            ]
        )

    # Merge the detections into a single RegionDetections
    merged_detections = detections.merge()

    # If the bounds of the individual regions were disjoint, then no NMS needs to be applied across
    # the different regions
    if detections.disjoint_bounds():
        logging.info("Bounds are disjoint, skipping across-region NMS")
        return merged_detections
    logging.info("Bound have overlap, running across-region NMS")

    # Run NMS on this merged RegionDetections
    NMS_suppressed_merged_detections = single_region_NMS(
        merged_detections,
        iou_theshold=iou_theshold,
        confidence_column=confidence_column,
        min_confidence=min_confidence,
    )

    return NMS_suppressed_merged_detections

def postprocess_detections(
        detections: RegionDetectionsSet,
        crs: Optional[pyproj.CRS] = None,
        tolerance: Optional[float] = 0.2,
        min_area_threshold: Optional[float] = 20.0
) -> RegionDetections:
    
    """Apply postprocessing techniques that include: 
    1. Get a union of polygons that have been split across tiles
    2. Simplify the edges of polygons by `tolerance` value
    3. Remove holes within the polygons that are smaller than `min_area` value
    Merges regions into a single RegionDetections. 

    Args:
        detections(RegionDetectionsSet):
            Detections from multiple regions to postprocess.
        crs (Optional[pyproj.CRS], optional):
            What CRS to use. Defaults to None.
        tolerance (Optional[float], optional):
            A value that controls the simplification of the detection polygons.
            The higher this value, the smaller the number of vertices in the resulting geometry.
        min_area_threshold (Optional[float], optional):
            Holes within polygons having an area lesser than this value get removed.

    Returns:
        RegionDetections:
            Postprocessed set of detections, merged together for the set of regions.
    """
    # Get the detections as a merged GeoDataFrame
    all_detections_gdf = detections.get_data_frame(merge=True)

    # Compute the union of the set of polyogns. This step removes any vertical lines caused by the tile edges 
    # and combines a single polygon that might have been split into multiple. Also removes any overlaps.
    union_detections = unary_union(all_detections_gdf.geometry)

    # Simplify the polygons by tolerance value and extract only Polygons and MultiPolygons 
    # since `union_detections` can have Point objects as well
    filtered_geoms = [
        geom.simplify(tolerance) for geom in list(union_detections.geoms)
        if isinstance(geom, (Polygon, MultiPolygon))
    ]

    # To remove small holes within polygons
    new_polygons = []
    for polygon in filtered_geoms:
        list_interiors = []

        for interior in polygon.interiors:
            p = Polygon(interior)
            # If the area of the hole is greater than the threshold, include it in the final output    
            if p.area > min_area_threshold:
                list_interiors.append(interior)

        # Create a new polygon for the same that does not have the smaller holes
        new_polygon = Polygon(polygon.exterior.coords, holes=list_interiors)
        new_polygons.append(new_polygon)

    # Create a RegionDetections for the merged and postprocessed detections 
    postprocessed_detections = RegionDetections(new_polygons, CRS=crs, input_in_pixels=False)

    return postprocessed_detections
