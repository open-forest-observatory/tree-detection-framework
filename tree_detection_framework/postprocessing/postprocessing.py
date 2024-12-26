import logging
from typing import Optional

import numpy as np
import pyproj
from polygone_nms import nms
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from tree_detection_framework.detection.region_detections import (
    RegionDetections,
    RegionDetectionsSet,
)


def single_region_NMS(
    detections: RegionDetections,
    iou_theshold: float = 0.5,
    confidence_column: str = "score",
    min_confidence: float = 0.3,
    intersection_method: str = "IOU",
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
    # Get rows that are both high confidence and not empty
    not_empty_mask = ~detections_df.geometry.is_empty
    high_conf_not_empty_inds = np.where(
        (
            (detections_df[confidence_column] >= min_confidence) & not_empty_mask
        ).to_numpy()
    )[0]

    # Filter detections based on minimum confidence score
    detections_df = detections_df.iloc[high_conf_not_empty_inds]
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
        intersection_method=intersection_method,
        threshold=iou_theshold,
    )

    # We only performed NMS on the high-confidence detections, but we need the indices w.r.t. the
    # original data with all detections. Sort for convenience so data is not permuted.
    keep_inds_in_original = sorted(high_conf_not_empty_inds[keep_inds])
    # Extract the detections that were kept
    subset_region_detections = detections.subset_detections(keep_inds_in_original)

    return subset_region_detections


def multi_region_NMS(
    detections: RegionDetectionsSet,
    run_per_region_NMS: bool = True,
    iou_theshold: float = 0.5,
    confidence_column: str = "score",
    min_confidence: float = 0.3,
    intersection_method: str = "IOU",
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
                    intersection_method=intersection_method,
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


def polygon_hole_suppression(polygon: Polygon, min_area_threshold: float = 20.0):
    """To remove holes within a polygon

    Args:
        polygon(shapely.Polygon):
            A shapely polygon object
        min_area_threshold(float):
            Remove holes within the polygons that have area smaller than this value

    Returns:
        shapely.Polygon:
            The equivalent polygon created after suppressing the holes
    """
    list_interiors = []
    # Iterate through interiors list which includes the holes
    for interior in polygon.interiors:
        interior_polygon = Polygon(interior)
        # If area of the hole is greater than the threshold, include it in the final output
        if interior_polygon.area > min_area_threshold:
            list_interiors.append(interior)

    # Return a new polygon with holes suppressed
    return Polygon(polygon.exterior.coords, holes=list_interiors)


def single_region_hole_suppression(
    detections: RegionDetections, min_area_threshold: float = 20.0
):
    """Suppress polygon holes in a RegionDetections object.

    Args:
        detections (RegionDetections):
            Detections from a single region that needs suppression of polygon holes.
        min_area_threshold(float):
            Remove holes within the polygons that have area smaller than this value.

    Returns:
        RegionDetections:
            Detections after suppressing polygon holes.
    """
    detections_df = detections.get_data_frame()
    modified_geometries = []

    for tree_crown in detections_df.geometry.to_list():
        # If tree_crown is a Polygon, directly do polygon hole suppression
        if isinstance(tree_crown, Polygon):
            clean_tree_crown = polygon_hole_suppression(tree_crown, min_area_threshold)
        # If it is a MultiPolygon, do polygon hole suppression for each polygon within it
        elif isinstance(tree_crown, MultiPolygon):
            clean_polygons = []
            for polygon in tree_crown.geoms:
                clean_polygon = polygon_hole_suppression(polygon, min_area_threshold)
                clean_polygons.append(clean_polygon)
            # Create a new MultiPolygon with the suppressed polygons
            clean_tree_crown = MultiPolygon(clean_polygons)
        # For any other cases, create an empty polygon (just to be safe)
        else:
            clean_tree_crown = Polygon()

        # Add the cleaned polygon/multipolygon to a list
        modified_geometries.append(clean_tree_crown)

    # Set this list as the geometry column in the dataframe
    detections_df.geometry = modified_geometries
    # Return a new RegionDetections object created using the updated dataframe
    # TODO: Handle cases where the data is in pixels with no transform to geospatial
    return RegionDetections(
        detection_geometries=None,
        data=detections_df,
        input_in_pixels=False,
        CRS=detections.get_CRS(),
    )


def multi_region_hole_suppression(
    detections: RegionDetectionsSet, min_area_threshold: float = 20.0
):
    """Suppress polygon holes in a RegionDetectionsSet object.

    Args:
        detections (RegionDetectionsSet):
            Set of detections from a multiple regions that need suppression of polygon holes.
        min_area_threshold(float):
            Remove holes within the polygons that have area smaller than this value.

    Returns:
        RegionDetectionsSet:
            Set of detections after suppressing polygon holes.
    """
    # Perform single_region_hole_suppression for every region within the RegionDetectionsSet
    return RegionDetectionsSet(
        [
            single_region_hole_suppression(region_detections, min_area_threshold)
            for region_detections in detections.region_detections
        ]
    )


def merge_and_postprocess_detections(
    detections: RegionDetectionsSet,
    tolerance: Optional[float] = 0.2,
    min_area_threshold: Optional[float] = 20.0,
) -> RegionDetections:
    """Apply postprocessing techniques that include:
    1. Get a union of polygons that have been split across tiles
    2. Simplify the edges of polygons by `tolerance` value
    3. Remove holes within the polygons that are smaller than `min_area_theshold` value
    Merges regions into a single RegionDetections.

    Args:
        detections(RegionDetectionsSet):
            Detections from multiple regions to postprocess.
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

    # Apply a small negative buffer to shrink polygons slightly
    buffered_geoms = [geom.buffer(-0.001) for geom in all_detections_gdf.geometry]

    # Compute the union of the set of polyogns. This step removes any vertical lines caused by the tile edges
    # and combines a single polygon that might have been split into multiple. Also removes any overlaps.
    union_detections = unary_union(buffered_geoms)

    # Simplify the polygons by tolerance value and extract only Polygons and MultiPolygons
    # since `union_detections` can have Point objects as well
    filtered_geoms = [
        geom.simplify(tolerance)
        for geom in list(union_detections.geoms)
        if isinstance(geom, (Polygon, MultiPolygon))
    ]

    # To remove small holes within polygons
    new_polygons = []
    for polygon in filtered_geoms:
        new_polygon = polygon_hole_suppression(polygon, min_area_threshold)
        new_polygons.append(new_polygon)

    # Create a RegionDetections for the merged and postprocessed detections
    # TODO: Handle cases when input is in pixels
    postprocessed_detections = RegionDetections(
        new_polygons, input_in_pixels=False, CRS=all_detections_gdf.crs
    )

    return postprocessed_detections
