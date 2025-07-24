import logging
from pathlib import Path
from typing import List, Optional, Union

from affine import Affine
import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame
from PIL import Image
from polygone_nms import nms
import rasterstats
from shapely import box
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.detection.region_detections import (
    RegionDetections,
    RegionDetectionsSet,
)


def single_region_NMS(
    detections: RegionDetections,
    threshold: float = 0.5,
    confidence_column: str = "score",
    min_confidence: float = 0.3,
    intersection_method: str = "IOU",
) -> RegionDetections:
    """Run non-max suppresion on predictions from a single region.

    Args:
        detections (RegionDetections):
            Detections from a single region to run NMS on.
        threshold (float, optional):
            The threshold for the NMS(intersection) method. Defaults to 0.5.
        confidence_column (str, optional):
            Which column in the dataframe to use as a confidence for NMS. Defaults to "score"
        min_confidence (float, optional):
            Prediction score threshold for detections to be included.
        intersection_method (str, optional):
            The method to compute intersections, one of ("IOU", "IOS", "Dice", "IOT"). Defaults to "IOU".

    Returns:
        RegionDetections:
            NMS-suppressed set of detections
    """
    # Extract the geodataframe for the detections
    detections_df = detections.get_data_frame()

    # Buffer by zero to fix any issues with invalid geometries
    detections_df["geometry"] = detections_df.geometry.buffer(0)

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
        threshold=threshold,
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
    threshold: float = 0.5,
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
        threshold (float, optional):
            The threshold for the NMS(intersection) method. Defaults to 0.5.
        confidence_column (str, optional):
            Which column in the dataframe to use as a confidence for NMS. Defaults to "score"

        min_confidence (float, optional):
            Prediction score threshold for detections to be included.
        intersection_method (str, optional):
            The method to compute intersections, one of ("IOU", "IOS", "Dice", "IOT"). Defaults to "IOU".
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
                    threshold=threshold,
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
        threshold=threshold,
        confidence_column=confidence_column,
        min_confidence=min_confidence,
        intersection_method=intersection_method,
    )

    return NMS_suppressed_merged_detections


def update_gdf_to_centroid(gdf: GeoDataFrame) -> GeoDataFrame:
    """Return the geodataframe with the geometry updated to be the centroid Point

    Args:
        gdf (GeoDataFrame): A dataframe to convert

    Returns:
        GeoDataFrame: the input geodataframe with the geometry updated to be the centroid Point
    """
    # Copy the data to avoid updating the orignal
    copied_gdf = gdf.copy()
    # Set the geometry column to the centroid
    copied_gdf.geometry = gdf.centroid

    return copied_gdf


def NMS_on_points(
    detections: Union[RegionDetections, RegionDetectionsSet],
    threshold_distance: float,
) -> RegionDetections:
    """
    Run non-max suppression on point data, suppressing points within a thereshold distance of
    higher-score points

    Args:
        detections (Union[RegionDetections, RegionDetectionsSet]):
            The input data to run NMS on. Can either be a RegionDetections or RegionDetectionsSet
            object.
        threshold_distance (float):
            Detections within this distance of each other are suppressed

    Returns:
        RegionDetections:
            The NMS-suppressed detections with all the original attributes
    """

    # Define a function to buffer each point out by half the suppression distance
    def buffer_out(gdf: GeoDataFrame):
        copied_gdf = gdf.copy()
        # Buffer by half the distance because both objects will be buffered
        copied_gdf.geometry = copied_gdf.buffer(threshold_distance / 2)
        return copied_gdf

    # Buffer the points out to form circles of the desired radius
    # TODO Consider error checking that this is actually point-typed data
    buffered_points = detections.apply_function_to_detections(buffer_out, inplace=False)

    # Determine whether the single- or multi-region NMS function based on the type of the input data
    NMS_func = (
        multi_region_NMS
        if isinstance(detections, RegionDetectionsSet)
        else single_region_NMS
    )

    # Run NMS with the appropriate function. The theshold is set very low because we want any nonzero
    # overlap between the buffered circles to to count as an intersection, since that means the two
    # original centers are less than the threshold distance
    NMS_suppressed = NMS_func(buffered_points, threshold=1e-10, min_confidence=0)

    # We ran NMS on the buffered version of the data. Now, convert it back to a point representation.
    # Note: this whole process should run even if the input data was not points, but the result will
    # be points either way
    points = NMS_suppressed.apply_function_to_detections(update_gdf_to_centroid)

    return points


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
    3. Remove holes within the polygons that are smaller than `min_area_threshold` value
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
        new_polygons, CRS=all_detections_gdf.crs
    )

    return postprocessed_detections


def suppress_tile_boundary_with_NMS(
    predictions: RegionDetectionsSet,
    iou_threshold: float = 0.5,
    ios_threshold: float = 0.5,
    min_confidence: float = 0.3,
) -> RegionDetections:
    """
    Used as a post-processing step with the `GeometricDetector` class to suppress detections that are split across tiles.
    This is done by applying NMS twice, first using IOU and then using IOS.

    Args:
        predictions (RegionDetectionsSet):
            Detections from multiple regions.
        iou_threshold (float, optional):
            The threshold for the NMS method that uses IoU metric. Defaults to 0.5.
        ios_threshold (float, optional):
            The threshold for the NMS method that uses IoS metric. Defaults to 0.5.
        min_confidence (float, optional):
            Prediction score threshold for detections to be included.

    Returns:
        RegionDetections:
            NMS postprocessed set of detections, merged together.
    """

    iou_nms = multi_region_NMS(
        predictions,
        intersection_method="IOU",
        threshold=iou_threshold,
        min_confidence=min_confidence,
    )

    iou_ios_nms = single_region_NMS(
        iou_nms,
        intersection_method="IOS",
        threshold=ios_threshold,
        min_confidence=min_confidence,
    )

    return iou_ios_nms


def remove_out_of_bounds_detections(
    region_detection_sets: List[RegionDetectionsSet], image_bounds: List
) -> List[RegionDetectionsSet]:
    """
    Filters out detections that are outside the bounds of a defined region.
    Used as a post-processing step after `predict_raw_drone_images()`.

    Args:
        region_detection_sets (List[RegionDetectionSet]):
            Each elemet is a RegionDetectionsSet derived from a specific drone image.
            Length is the number of raw drone images given to  the dataloader.
        image_bounds (List[bounding_box]):
            Each element is a `bounding_box` object derived from the dataloader.
            Length: number of regions in a set * number of RegionDetectionSet objects

    Returns:
        List of RegiondetectionSet objects with out-of-bounds predictions filtered out.
    """

    region_idx = 0  # To index elements in true_bounds
    list_of_filtered_region_sets = []

    for rds in region_detection_sets:

        # Find the number of regions in every set
        num_of_regions_in_a_set = len(rds.get_data_frame())
        # To save filtered regions in a particular set
        list_of_filtered_regions = []

        # Get the region image bounds for the RegionDetectionSet
        region_image_bounds = image_bounds[region_idx]

        for idx in range(num_of_regions_in_a_set):

            # Get RegionsDetections object from the set
            rd = rds.get_region_detections(idx)
            rd_gdf = rd.get_data_frame()

            # Create a polygon of size equal to the image dimensions
            region_set_polygon = box(
                region_image_bounds.minx,
                region_image_bounds.maxy,
                region_image_bounds.maxx,
                region_image_bounds.miny,
            )

            # TODO: Instead of removing detections partially extending out of the boundary,
            # try cropping it using gpd.clip()
            # Remove detections that extend beyond the image bounds
            within_bounds_indices = rd_gdf.within(region_set_polygon)
            within_bounds_indices_true = within_bounds_indices[
                within_bounds_indices
            ].index

            # Subset the RegionDetections object keeping only the valid indices calculated before
            filtered_rd = rd.subset_detections(within_bounds_indices_true)
            list_of_filtered_regions.append(filtered_rd)

        list_of_filtered_region_sets.append(
            RegionDetectionsSet(list_of_filtered_regions)
        )

        # Update region_idx to point to the image dims of the next rds
        region_idx += num_of_regions_in_a_set

    return list_of_filtered_region_sets


def remove_edge_detections(
    detections: RegionDetectionsSet,
    suppression_distance: float,
    retain_method: str = "within",
) -> RegionDetectionsSet:
    """Remove detections at the edges of tiles

    Args:
        detections (RegionDetectionsSet):
            The detections to suppress
        suppression_distance (float):
            Remove detections within this distance of the boundary
        retain_method (str, optional):
            How to deal with edge detections. "within" keeps detections that are fully within the
            interior region, "intersects" keeps any that touch the interior region, and "clip" clips
            detections on the boundary so only the portion within the interior is retained. Defaults
            to "within".

    Returns:
        RegionDetectionsSet: Updated detections
    """
    updated_rd_list = []
    for rd in detections.region_detections:
        # Compute the buffered bounds for this region
        buffered_bounds = rd.get_bounds().buffer(-suppression_distance)

        def update_detections(detections: gpd.GeoDataFrame):
            if retain_method == "within":
                # Keep detections that are fully within the interior region
                updated_detections = detections[
                    detections.within(buffered_bounds.geometry[0])
                ]
            elif retain_method == "intersects":
                # Keep detections that intersect the interior region
                updated_detections = detections[
                    detections.intersects(buffered_bounds.geometry[0])
                ]
            elif retain_method == "clip":
                # Clip detections to the interior region, entirely dropping empty geometries
                updated_detections = detections.clip(buffered_bounds)
            else:
                raise ValueError(
                    f"retain_method was {retain_method}, but should be 'within', 'intersects', or 'clip'"
                )

            return updated_detections

        # Apply the function to the detections DataFrame to suppress detections in the edge region
        updated_rd = rd.apply_function_to_detections(update_detections)
        # Record the result
        updated_rd_list.append(updated_rd)

    # Create a new RDS
    updated_rds = RegionDetectionsSet(updated_rd_list)
    return updated_rds


def remove_masked_detections(
    region_detection_sets: List[RegionDetectionsSet],
    image_root: PATH_TYPE,
    image_paths: List[PATH_TYPE],
    valid_classes: List[int],
    mask_root: PATH_TYPE,
    mask_extension: str = ".png",
    threshold: float = 0.4,
) -> List[RegionDetectionsSet]:
    """
    Filters out detections that marked as invalid in the given mask images.
    One example use case is that mask images could be rastered where the
    certain pixels are marked as "ground" vs. "above ground". Then detections
    that mostly overlap with the "ground" mask could be removed.

    Args:
        region_detection_sets (List[RegionDetectionSet]):
            Each element is a RegionDetectionsSet derived from a specific drone image.
            Length is the number of raw drone images given to  the dataloader.
            image_root (PATH_TYPE)
            image_paths (List[PATH_TYPE])
            valid_classes (List[int])
            mask_root (PATH_TYPE)
            mask_extension (str)
                Defaults to ".png".
            threshold (float)
                Defaults to 0.4.

    Returns:
        List of RegiondetectionSet objects with masked predictions filtered out.
    """

    filtered_sets = []

    # Iterate over [0] the path to a specific image and [2] the
    # RegionDetectionsSet of detections in that image
    for im_path, rds in zip(image_paths, region_detection_sets):

        # Assuming the mask path relative to the mask root matches the
        # image path relative to the image root, open the mask file
        subpath = Path(im_path).relative_to(image_root)
        mask_path = (mask_root / subpath).with_suffix(mask_extension)

        # Calculate a mask which is 1 where the data is valid, a.k.a.
        # in the parts of the image we think could contain good detections.
        # Open the image as grayscale
        mask_img = Image.open(mask_path).convert("L")
        # Make the mask integer so that we can do math on the detections.
        # For example, if a detection is 50+% valid, the mean value within
        # the polygon will be 0.5+ because the mask is in numbers.
        mask = np.isin(mask_img, valid_classes).astype(int)

        # Define a transformation from the image space ([0, 0] at the top left,
        # y increases going down) to rasterstats ([0, 0] at the bottom left,
        # y increases going up)
        # transform = Affine.identity()
        transform = Affine.translation(0, mask.shape[0]) * Affine.scale(1, -1)

        # To save filtered region detections in a particular set
        filtered_regions = []

        # Iterate over the region detections, which is equivalent to iterating
        # over the chips that detections were calculated in
        for idx in range(len(rds.region_detections)):
            rd = rds.get_region_detections(idx)
            gdf = rd.get_data_frame()

            # Get the mean value of each detection polygon, as a list of
            # [{"mean": <value>}, ...]
            stats = rasterstats.zonal_stats(
                gdf,
                mask.astype(int),
                stats=["mean"],
                affine=transform,
            )

            # Get indices that have a greater fraction of good pixels than the
            # threshold requires.
            good_indices = [i for i, stat in enumerate(stats) if stat["mean"] > threshold]

            # Subset the RegionDetections object keeping only the valid indices
            filtered_rd = rd.subset_detections(good_indices)
            filtered_regions.append(filtered_rd)

        filtered_sets.append(RegionDetectionsSet(filtered_regions))

    return filtered_sets