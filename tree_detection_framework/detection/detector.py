import os
import logging
from abc import abstractmethod
from itertools import groupby
from typing import Any, DefaultDict, Dict, Iterator, List, Optional, Tuple, Union

import geopandas as gpd
import lightning
import numpy as np
import pyproj
import rasterio
import shapely
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from rasterio.features import rasterize, shapes
from scipy.ndimage import maximum_filter
from shapely import affinity
from shapely.geometry import (
    GeometryCollection,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
    shape,
)
from shapely.geometry.base import BaseGeometry
from skimage.filters import gaussian
from skimage.segmentation import watershed
from torch.utils.data import DataLoader
from torchgeo.datasets import IntersectionDataset
from tqdm import tqdm

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.detection.models import DeepForestModule
from tree_detection_framework.detection.region_detections import (
    RegionDetections,
    RegionDetectionsSet,
)
from tree_detection_framework.preprocessing.derived_geodatasets import (
    CustomDataModule,
    CustomRasterDataset,
    bounding_box,
)
from tree_detection_framework.utils.detection import calculate_scores
from tree_detection_framework.utils.geometric import mask_to_shapely

try:
    import detectron2.data.transforms as T
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import MetadataCatalog
    from detectron2.modeling import build_model

    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    logging.warning("detectron2 not found. MaskRCNNDetector will be disabled.")

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import warnings

warnings.filterwarnings("ignore", message=".*nodata.*shadowing.*")


class Detector:
    def __init__(self, postprocessors=None):
        """
        Base class for all detectors.
        Args:
            postprocessors (list, optional):
                List of postprocessing functions applied sequentially to the predictions.
                Each element must be a callable that takes a single RegionDetections{Set} as input and returns a modified
                RegionDetections{Set}. This can be a lambda or a named function.
                If `detector.predict()` is called, first postprocessing step must take a RegionDetectionsSet as input.
                If `detector.predict_as_generator()` is called, all postprocessing steps must take a RegionDetections as input.

                Example:
                postprocessors = [
                    lambda r: suppress_tile_boundary_with_NMS(
                        r, iou_threshold=0.5, ios_threshold=0.5, min_confidence=0.3
                    ),
                    lambda r: single_region_NMS(
                        r, confidence_column="score", threshold=0.6, min_confidence=0.3
                    ),
                ]
        """
        self.postprocessors = postprocessors or []

    @abstractmethod
    def setup(self):
        """Any setup tasks that should be performed once when the Detector is instantiated"""
        # This should not be implemented here unless there are setup tasks that are shared by every detector
        raise NotImplementedError()

    def predict_as_generator(
        self,
        inference_dataloader: DataLoader,
        postprocess_region_detections: bool = False,
        **kwargs,
    ) -> Iterator[RegionDetections]:
        """
        A generator that yields a RegionDetections object for each image in the dataloader. Note
        that the dataloader may have batched data but predictions will be returned individually.

        Args:
            inference_dataloader (DataLoader): Dataloader to generate predictions for
            postprocess_region_detections (bool, optional): Set as True if `postprocessors` is intended for RegionDetections.
        """
        # Store the dataset resolution so that derived detector classes can access it if needed
        # Resolution is only available for raster datasets, and is only used for geometric tree
        # identification (not usable on other types of datasets anyway)
        if isinstance(
            inference_dataloader.dataset, (CustomRasterDataset, IntersectionDataset)
        ):
            self.data_resolution = inference_dataloader.dataset.res
        else:
            self.data_resolution = None

        # Iterate over each batch in the dataloader
        for batch in tqdm(
            inference_dataloader, desc="Performing prediction on batches"
        ):

            # This is the expensive step, generate the predictions using predict_batch from the
            # derived class. The additional arguments are also passed to this method with kwargs
            batch_preds_pixel_geometries, batch_preds_data = self.predict_batch(
                batch, **kwargs
            )

            # If the prediction doesn't generate any data, set it to a list of None for
            # compatability with downstream steps
            if batch_preds_data is None:
                batch_preds_data = [None] * len(batch_preds_pixel_geometries)

            # Convert the detections to geospatail
            batch_preds_geospatial_geometries = self.convert_detections_to_geospatial(
                batch_preds_pixel_geometries, batch
            )

            # Attributes required to construct the RegionDetection object
            CRS = self.get_CRS_from_batch(batch)
            batch_geospatial_bounds = self.get_geospatial_bounds_as_shapely(batch)

            # Iterate over samples in the batch so we can yield them one at a time
            for preds_geometry, preds_data, geospatial_bounds in zip(
                batch_preds_geospatial_geometries,
                batch_preds_data,
                batch_geospatial_bounds,
            ):
                # Create a region detections object
                region_detections = RegionDetections(
                    detection_geometries=preds_geometry,
                    data=preds_data,
                    CRS=CRS,
                    geospatial_prediction_bounds=geospatial_bounds,
                )

                if postprocess_region_detections:
                    # Apply postprocessing steps to the RegionDetections
                    for func in self.postprocessors:
                        region_detections = func(region_detections)

                # Yield this object
                yield region_detections

    def predict(
        self, inference_dataloader: DataLoader, return_as_list: bool = False, **kwargs
    ) -> Union[List[RegionDetections], RegionDetectionsSet]:
        """
        Generate predictions for every image in the dataloader. Calls self.predict_as_generator()
        and then converts to either a list or RegionDetectionSet for convenience.

        Args:
            inference_dataloader (DataLoader):
                Dataloader to generate predictions for
            return_as_list (bool, optional):
                Should a list of RegionDetections be returned rather than a single
                RegionDetectionSet. Defaults to False.

        Returns:
            Union[List[RegionDetections], RegionDetectionsSet]: Either a list of RegionDetections
            objects (on per image) or a single RegionDetectionsSet containing the same information.
        """
        # Get the generator that will generate predictions. Note this only creates the generator,
        # computation is defered until the samples are actually requested
        predictions_generator = self.predict_as_generator(
            inference_dataloader, **kwargs
        )
        # This step is where the computation actually occurs since all samples are requested to
        # build the list
        predictions_list = list(predictions_generator)
        # If we want the output to be a list, return it here
        if return_as_list:
            return predictions_list

        # Otherwise convert it to a RegionDetectionsSet and return that
        region_detection_set = RegionDetectionsSet(predictions_list)

        # Apply postprocessing steps to the RegionDetectionsSet
        for func in self.postprocessors:
            region_detection_set = func(region_detection_set)

        return region_detection_set

    def predict_raw_drone_images(
        self, inference_dataloader: DataLoader, **kwargs
    ) -> Tuple[List[RegionDetectionsSet], List[str], List[bounding_box]]:
        """
        Generate predictions for every image in the dataloader created using `CustomImageDataset` for raw drone images.
        Calls self.predict_as_generator() and retains predictions as a list.

        Args:
            inference_dataloader (DataLoader):
                Dataloader to generate predictions for

        Returns:
            region_detections_sets (List[RegionDetectionsSet]):
                List of `RegionDetectionsSet` objects
            keys (List[str]):
                List of image filepaths corresponding to region_detections_sets
            true_bounds (List[bounding_box]):
                List of image bounding box values at RegionDetections level
        """
        # Get the generator that will generate predictions. Note this only creates the generator,
        # computation is defered until the samples are actually requested
        predictions_generator = self.predict_as_generator(
            inference_dataloader, **kwargs
        )
        # This step is where the computation actually occurs since all samples are requested to
        # build the list
        predictions_list = list(predictions_generator)

        # Extract the source image names associated with each tile in the inference dataloader
        image_filenames = [
            metadata["source_image"]
            for batch in inference_dataloader
            for metadata in batch["metadata"]
        ]

        # Extract image dimensions. This is saved for a post-processing step.
        image_bounds = [
            metadata["image_bounds"]
            for batch in inference_dataloader
            for metadata in batch["metadata"]
        ]

        # Create a zip with each RegionDetections and its corresponding source image name
        preds_and_images = zip(predictions_list, image_filenames)

        # Obtain groups of RegionDetections after grouping by source image name
        groups = groupby(preds_and_images, key=lambda x: x[1])

        # Create a RegionDetectionsSet for each group
        region_detections_sets = []
        keys = []
        for key, group in groups:
            region_detections_sets.append(RegionDetectionsSet([i[0] for i in group]))
            keys.append(key)  # source image names

        return region_detections_sets, keys, image_bounds

    @abstractmethod
    def predict_batch(
        self, batch: dict
    ) -> Tuple[List[List[shapely.Geometry]], Union[None, List[dict]]]:
        """Generate predictions for a batch of samples

        Args:
            batch (dict): A batch from the torchgeo dataloader

        Returns:
            List[List[shapely.geometry]]:
                A list of predictions one per image in the batch. The predictions for each image
                are a list of shapely objects.
            Union[None, List[dict]]:
                Any additional attributes that are predicted (such as class or confidence). Must
                be formatted in a way that can be passed to gpd.GeoPandas data argument.
        """
        # Should be implemented by each derived class
        raise NotImplementedError()

    @staticmethod
    def get_geospatial_bounds_as_shapely(
        batch: DefaultDict[str, Any],
    ) -> List[shapely.geometry.Polygon]:
        """Get geospatial region bounds as shapely objects from a batch.
        Args:
            batch: (DefaultDict[str, Any]): Batch from DataLoader with image sample(s).
        Returns:
            List[shapely.geometry.Polygon]: A list of shapely Polygons representing the geospatial bounds.
        """
        batch_bounds = batch["bounds"]
        return [
            shapely.box(
                xmin=tile_bounds.minx,
                ymin=tile_bounds.miny,
                xmax=tile_bounds.maxx,
                ymax=tile_bounds.maxy,
            )
            for tile_bounds in batch_bounds
        ]

    @staticmethod
    def get_pixel_to_geospatial_transforms(
        batch: DefaultDict[str, Any],
    ) -> List[Tuple]:
        """Compute a list of affine transforms for each sample in the batch

        Args:
            batch (DefaultDict[str, Any]): Batch from the dataloader

        Returns:
            List[Tuple]:
                A list of tuples specifying the transform from pixel to geospatial coordinates
                following the convention described here:
                https://shapely.readthedocs.io/en/stable/manual.html#shapely.affinity.affine_transform
        """
        # Get the geospatial bounds for each tile in the batch
        geospatial_bounds = Detector.get_geospatial_bounds_as_shapely(batch)

        # Compute the pixel-space bounds
        image_shape = batch["image"].shape[-2:]
        # Note that the y min bounds are reversed from the expected convention. This is because
        # they are measured in pixel coordinates, which start at the top and go down. So this
        # convention matches how the geospatial bounding box is represented.
        single_image_bounds = shapely.box(
            xmin=0, ymin=image_shape[0], xmax=image_shape[1], ymax=0
        )
        # Duplicate the bounds for each sample in the batch
        image_bounds = [single_image_bounds] * batch["image"].shape[0]

        # Compute the transform for each sample in the batch
        transforms = []
        for geo_bound, image_bound in zip(geospatial_bounds, image_bounds):
            # Compute the array corresponding to the corners of the tile. Note that the duplicated
            # start/end point is removed. The starting point and direction is assumed to be
            # consistent between the two.
            geospatial_corners_array = shapely.get_coordinates(geo_bound)[:-1]
            pixel_corners_array = shapely.get_coordinates(image_bound)[:-1]

            # Representing the correspondences as rasterio ground control points
            ground_control_points = [
                rasterio.control.GroundControlPoint(
                    col=pixel_vertex[0],
                    row=pixel_vertex[1],
                    x=geospatial_vertex[0],
                    y=geospatial_vertex[1],
                )
                for pixel_vertex, geospatial_vertex in zip(
                    pixel_corners_array, geospatial_corners_array
                )
            ]
            # Solve the affine transform that best transforms from the pixel to geospatial coordinates
            pixel_to_CRS_transform = rasterio.transform.from_gcps(ground_control_points)

            # Get the transform in the format expected by shapely
            shapely_transform = pixel_to_CRS_transform.to_shapely()

            transforms.append(shapely_transform)

        return transforms

    @staticmethod
    def convert_detections_to_geospatial(
        detection_geometries: List[List[BaseGeometry]],
        batch: DefaultDict[str, Any],
    ) -> List[List[BaseGeometry]]:
        """Convert detection geometries from pixel to geospatial coordinates

        Args:
            detection_geometries (List[List[BaseGeometry]]):
                The input geometries in pixel coordinates. Each element in the outer list should
                correspond to a tile in the `batch`
            batch (DefaultDict[str, Any]):
                A batch from the dataloader from which the `detection_geometries` were derived

        Returns:
            List[List[BaseGeometry]]: Detections transformed into the CRS of the batch
        """
        # Compute the pixel-to-geospatial transform for each element of the batch
        batch_trasforms = Detector.get_pixel_to_geospatial_transforms(batch)

        # Transform each set of detections based on the corresponding transform for that tile
        batch_transformed_detections = [
            [
                affinity.affine_transform(detection, tile_transform)
                for detection in tile_detections
            ]
            for tile_detections, tile_transform in zip(
                detection_geometries, batch_trasforms
            )
        ]

        return batch_transformed_detections

    @staticmethod
    def get_CRS_from_batch(batch):
        # Assume that the CRS is the same across all elements in the batch
        CRS = batch["crs"][0]
        # Get the CRS EPSG value and convert it to a pyproj object
        # This is to avoid relying on WKT strings which are more likely to be invalid
        if CRS is not None:
            CRS = pyproj.CRS(CRS.to_epsg())
        return CRS


class RandomDetector(Detector):
    """A detector that produces random detections primarily used for testing"""

    def predict_batch(
        self,
        batch: dict,
        detections_per_tile: int = 10,
        detection_size_fraction: float = 0.1,
        score_column: str = "score",
    ) -> Tuple[List[List[shapely.Geometry]], List[dict]]:
        """Generates random detections for each image in the batch

        Args:
            batch (dict): The batch of images. Only used to obtain the size in pixels.
            detections_per_tile (int, optional): How many detections to generate per image. Defaults to 10.
            detection_size_fraction (float, optional): What fraction of the image size should each detection be. Defaults to 0.1.
            score_column (str, optional): What column name to use for the randomly-generated score. Defaults to "score".

        Returns:
            List[List[shapely.Geometry]]: The list of lists of random rectangles per image
            List[dict]: The random scores for each detection
        """
        # Check the parameters
        if detection_size_fraction < 0 or detection_size_fraction > 1:
            raise ValueError(
                f"detection_size_fraction must be between 0 and 1 but instead was {detection_size_fraction}"
            )

        if detections_per_tile < 0:
            raise ValueError(
                f"detections_per_tile must be positive but instead was {detections_per_tile}"
            )

        # Determine the shape of the image in pixels
        tile_size = batch["image"].shape[-2:]
        # Create lists for the whole batch to append to
        batch_geometries = []
        batch_datas = []

        # Each sample is randomly generated for each sample in the batch
        for _ in range(batch["image"].shape[0]):
            # Expand the size so it can be broadcast with the 2D variables
            broadcastable_size = np.expand_dims(tile_size, 0)
            # Compute the detection size as a fraction of the total image size
            detection_size = broadcastable_size * detection_size_fraction
            # Randomly compute the top left corner locations by using the region that will not
            # cause the detection to exceed the upper bound of the image
            tile_tl = (
                np.random.random((detections_per_tile, 2))
                * broadcastable_size
                * (1 - detection_size_fraction)
            )
            # Compute the bottom right corner by adding the (constant) size to the top left corners
            tile_br = tile_tl + detection_size

            # Convert these corners to a list of shapely objects
            detection_boxes = shapely.box(
                tile_tl[:, 0],
                tile_tl[:, 1],
                tile_br[:, 0],
                tile_br[:, 1],
            )
            # Create random scores for each detection
            data = {score_column: np.random.random(detections_per_tile)}

            # Append the geometries and data to the lists
            batch_geometries.append(detection_boxes)
            batch_datas.append(data)

        return batch_geometries, batch_datas


class GeometricTreeTopDetector(Detector):

    def __init__(
        self,
        a: float = 0,
        b: float = 0.0325,
        c: float = 0.25,
        min_ht: int = 5,
        filter_shape: str = "circle",
        blur_sigma: float = 0,
        confidence_feature: str = "distance",
        postprocessors=None,
    ):
        """Detector to detect treetops for CHM data. Implementation of the variable window filter algorithm of Popescu and Wynne (2004).
        Args:
           a (float, optional): Coefficient for the quadratic term in the radius calculation. Defaults to 0.
           b (float, optional): Coefficient for the linear term in the radius calculation. Defaults to 0.0325.
           c (float, optional): Constant term in the radius calculation. Defaults to 0.25.
           min_ht (int, optional): Minimum height for a pixel to be considered as a tree. Defaults to 5.
           filter_shape (str, optional): Shape of the filter to use for local maxima detection.
               Choose from "circle", "square", "none". Defaults to "circle". Defaults to "circle".
           blur_sigma (float, optional): Standard deviation of the 2D Gaussian smoothing kernel.
           confidence_feature (str, optional): Feature to use to compute the confidence scores for the predictions.
                Choose "height" or "distance". Defaults to "distance".
           postprocessors (list, optional):
               See docstring for Detector class. Defaults to None.
        """
        super().__init__(postprocessors=postprocessors)
        self.a = a
        self.b = b
        self.c = c
        self.min_ht = min_ht
        self.filter_shape = filter_shape
        self.blur_sigma = blur_sigma
        self.confidence_feature = confidence_feature

    def get_treetops(self, image: np.ndarray) -> tuple[List[Point], List[float]]:
        """Calculate treetop coordinates using pre-filtering to identify potential maxima.

        Args:
            image (np.ndarray): A single-channel CHM image.

        Returns:
            tuple[List[Point], List[float]] containing:
                all_treetop_pixel_coords (List[Point]): Detected treetop coordinates in pixel units.
                all_treetop_heights (List[float]): Treetop heights corresponding to the coordinates.
        """
        all_treetop_pixel_coords = []
        all_treetop_heights = []

        # Apply a maximum filter to the CHM to identify potential treetops based on the local maxima
        # Calculate the minimum suppression radius as a function of the minimum height
        min_radius = (self.a * (self.min_ht**2)) + (self.b * self.min_ht) + self.c

        # Determine filter size in pixels
        min_radius_pixels = int(np.floor(min_radius / self.data_resolution))

        # Blur the chip if needed
        if self.blur_sigma is not None and self.blur_sigma != 0.0:
            # Retain the original to query the heights
            unsmoothed_image = image.copy()
            # Smooth the image with a 2D gaussian blur
            image = gaussian(
                image, sigma=self.blur_sigma / self.data_resolution, preserve_range=True
            )
        else:
            # Image is unchanged since there is no blurring. And the unsmoothed image is not
            # actually copied, it's the same object.
            unsmoothed_image = image

        if self.filter_shape == "circle":
            # Create a circular footprint
            y, x = np.ogrid[
                -min_radius_pixels : min_radius_pixels + 1,
                -min_radius_pixels : min_radius_pixels + 1,
            ]
            footprint = x**2 + y**2 <= min_radius_pixels**2

            # Use a sliding window to find the maximum value in the region
            filtered_image = maximum_filter(
                image, footprint=footprint, mode="constant", cval=0
            )

        elif self.filter_shape == "square":
            # Create a square window using the computed radius
            # Reduce filter size by 1/sqrt(2) to keep corners within the suppression radius
            window_size = int((min_radius_pixels * 2) / np.sqrt(2))

            # Use a sliding window to find the maximum value in the region
            filtered_image = maximum_filter(
                image, size=window_size, mode="constant", cval=0
            )

        elif self.filter_shape == "none":
            # Local maxima filtering step is skipped
            logging.info("No filter applied to the image. Using brute-force method.")
            filtered_image = image

        else:
            raise ValueError(
                "Invalid filter_shape. Choose from: 'circle', 'square', 'none'."
            )

        # Create a mask for pixels that are above the min_ht threshold (left condition)
        # and are local maxima (right condition) if the image was filtered
        thresholded_mask = (image >= self.min_ht) & (image == filtered_image)

        # Get the selected coordinates
        selected_indices = np.argwhere(thresholded_mask)

        for i, j in selected_indices:
            ht = image[i, j]

            # Calculate the radius based on the pixel height
            radius = (self.a * (ht**2)) + (self.b * ht) + self.c
            # Ensure the radius is at least 1 pixel
            radius_pixels = max(radius / self.data_resolution, 1)
            side = int(np.ceil(radius_pixels))

            # Define bounds for the neighborhood
            i_min = max(0, i - side)
            i_max = min(image.shape[0], i + side + 1)
            j_min = max(0, j - side)
            j_max = min(image.shape[1], j + side + 1)

            # Create column and row vectors for the neighborhood
            region_i = np.arange(i_min, i_max)[:, np.newaxis]
            region_j = np.arange(j_min, j_max)[np.newaxis, :]

            # Calculate the distances to every point within the region
            distances = np.sqrt((region_i - i) ** 2 + (region_j - j) ** 2)

            # Create a mask for pixels inside the circle
            mask = distances <= radius_pixels

            # Apply the mask to the neighborhood
            neighborhood = image[i_min:i_max, j_min:j_max][mask]

            # Check if the pixel has the max height within the neighborhood
            if ht == np.max(neighborhood):
                unsmoothed_height = unsmoothed_image[i, j]
                # Ensure that the unsmoothed height is also above the minimum
                if unsmoothed_height > self.min_ht:
                    all_treetop_pixel_coords.append(Point(j, i))
                    # The height is computed using the un-smoothed CHM since the smoothing will
                    # underestimate the true height.
                    all_treetop_heights.append(unsmoothed_height)

        return all_treetop_pixel_coords, all_treetop_heights

    def predict_batch(self, batch):
        """Generate predictions for a batch of samples

        Args:
            batch (dict): A batch from the torchgeo dataloader

        Returns:
            List[List[shapely.geometry]]:
                A list of predictions one per image in the batch. The predictions for each image
                are a list of shapely objects.
            Union[None, List[dict]]:
                Any additional attributes that are predicted (such as class or confidence). Must
                be formatted in a way that can be passed to gpd.GeoPandas data argument.
        """
        # List to store every CHM tile's detections
        batch_detections = []
        batch_detections_data = []
        for chm_tile in batch["image"]:
            chm_tile = chm_tile.squeeze()
            # Set NaN values to zero
            chm_tile = np.nan_to_num(chm_tile)

            # Get the treetop locations from the CHM
            treetop_pixel_coords, treetop_heights = self.get_treetops(chm_tile)

            # Create a GeoDataFrame to store information associated with the image
            tile_gdf = gpd.GeoDataFrame(
                {
                    "geometry": treetop_pixel_coords,
                    "treetop_height": treetop_heights,
                }
            )

            batch_detections.append(
                treetop_pixel_coords
            )  # List[List[shapely.geometry]]
            # Calculate scores based on treetop height or distance from edge
            confidence_scores = calculate_scores(
                "geometry", self.confidence_feature, tile_gdf, chm_tile.shape
            )
            batch_detections_data.append(
                {"height": treetop_heights, "score": confidence_scores}
            )
        return batch_detections, batch_detections_data


class GeometricTreeCrownDetector(Detector):

    def __init__(
        self,
        approach: str = "watershed",
        radius_factor: float = 0.6,
        threshold_factor: float = 0.3,
        confidence_feature: str = "area",
        contour_backend: str = "cv2",
        tree_height_column: str = "height",
        min_height: float = 5,
        simplify_tolerance: float = 2.0,
        postprocessors=None,
    ):
        """Detect tree crowns for CHM data. This class requires the treetops to be detected first.

        Args:
            approach (str, optional):
                Which approach to use for tree crown computation. Choose from "watershed" or "silva".
                Defaults to "watershed". For more details about the approaches, see the function docstrings.
            radius_factor (float, optional):
                Factor to determine the radius of the tree crown. Defaults to 0.6.
                Used in "silva" approach.
            threshold_factor (float, optional):
                Factor to determine the threshold for the binary mask. Defaults to 0.3.
                Used in "silva" approach.
            confidence_feature (str, optional):
                Feature to use to compute the confidence scores for the predictions.
                Choose from "height", "area", "distance", "all". Defaults to "area".
                Used in all approaches.
            contour_backend (str, optional):
                The backend to use for contour extraction to generate treecrowns.
                Choose from "cv2" and "contourpy". Used in "silva" approach.
            tree_height_column (str, optional):
                Column name in the vector data that contains the treetop heights. Defaults to "height".
                Used in all approaches.
            min_height (float, optional):
                Only include pixels from CHM that are above this value. Defaults to 5.
                Used in "watershed" approach.
            simplify_tolerance (float, optional):
                Simplify tree crown polygons by this tolerance value. Units is in pixels.
                Defaults to 2. Used in "watershed" approach.
            postprocessors (list, optional):
                See docstring for Detector class. Defaults to None.

        """
        super().__init__(postprocessors=postprocessors)
        self.approach = approach
        self.radius_factor = radius_factor
        self.threshold_factor = threshold_factor
        self.confidence_feature = confidence_feature
        self.backend = contour_backend
        self.tree_height_column = tree_height_column
        self.min_height = min_height
        self.simplify_tolerance = simplify_tolerance

        logging.info(f"Using {self.approach} approach to compute the tree crowns.")

    def get_tree_crowns_watershed(
        self,
        image: np.ndarray,
        all_treetop_pixel_coords: List[Point],
        all_treetop_heights: List[float],
        all_treetop_ids: Optional[List[str]] = None,
    ) -> Tuple[gpd.GeoDataFrame, List[float]]:
        """
        Marker-Controlled Watershed Segmentation from detected treetop points. This replicates the behaviour of R function "mcws".
        https://github.com/andrew-plowright/ForestTools/blob/master/R/mwcs.R

        Args:
            image (np.ndarray): A single channel CHM image
            all_treetop_pixel_coords (List[Point]): A list with all detected treetop coordinates in pixel units
            all_treetop_heights (List[float]): A list with treetop heights in the same sequence as the coordinates
            all_treetop_ids: (List[str]) : Detected treetop IDs. Defaults to None
            min_height: (float) : Minimum CHM threshold to keep

        Returns:
            (GeoDataFrame, List[float]): GeoDataFrame containing crowns, treetops, tree heights and a list with confidence scores
        """

        # Generate default treetop IDs if none provided
        if all_treetop_ids is None:
            treetop_ids_provided = False
            all_treetop_ids = [f"{i:05d}" for i in range(len(all_treetop_pixel_coords))]
        else:
            treetop_ids_provided = True

        # Filter CHM values to only include pixels above the minimum height
        chm_masked = np.where(image >= self.min_height, image, 0)

        # Filter treetops that are valid (inside CHM and above height threshold)
        H, W = image.shape
        filtered = [
            (pt, uid, h)
            for pt, uid, h in zip(
                all_treetop_pixel_coords, all_treetop_ids, all_treetop_heights
            )
            if 0 <= pt.x < W and 0 <= pt.y < H and h >= self.min_height
        ]
        if len(filtered) == 0:
            # Tile has no treetops above threshold. Return an empty GeoDataFrame.
            columns = ["tree_crown", "treetop_height", "treetop_pixel_coords"]
            if treetop_ids_provided:
                columns.append("treetop_unique_ID")
            empty_gdf = gpd.GeoDataFrame(columns=columns, geometry="tree_crown")
            return empty_gdf, []

        # Unpack the list of (point, ID, height) tuples into separate lists
        filtered_points, filtered_ids, filtered_heights = zip(*filtered)

        # Rasterize treetops as markers
        markers = rasterize(
            # Convert uid to int since it's originally a string with 0 padded values
            # Add 1 to the value because watershed expects the background to be represented by 0s
            [(pt, int(uid) + 1) for pt, uid in zip(filtered_points, filtered_ids)],
            out_shape=image.shape,
            fill=0,
        )

        # Invert the CHM so that high points (tree tops) become basins for watershed
        elevation = -1 * chm_masked
        # Perform watershed segmentation with treetop markers
        labels = watershed(elevation, markers=markers, mask=chm_masked > 0)

        # Create a mapping of treetop IDs and heights
        id_to_height = {
            int(uid): height for uid, height in zip(filtered_ids, filtered_heights)
        }
        id_to_treetop_pixel_coords = {
            int(uid): tpc for uid, tpc in zip(filtered_ids, filtered_points)
        }

        # Convert the labeled raster into polygon geometries using rasterio's shapes().
        # Each contiguous region of the same label value/crown ID is extracted as a polygon
        # `tree_id` is the ID associated with that region.
        # `mask` ensures that only non-zero crown areas are included in the output.
        crowns = []
        for geom, tree_id in shapes(labels.astype("int32"), mask=(labels > 0)):
            # Subtract one to account for the +1 offset when the mask was created.
            tree_id = int(tree_id) - 1
            # Return the crown polygon, and height and location of the tree top
            data_dict = {
                "tree_crown": shape(geom),
                "treetop_height": id_to_height.get(tree_id),
                "treetop_pixel_coords": id_to_treetop_pixel_coords.get(tree_id),
            }
            # Return treetop IDs only if they were separately detected
            if treetop_ids_provided:
                data_dict["treetop_unique_ID"] = f"{tree_id:05d}"
            crowns.append(data_dict)

        # Create a gdf for the output
        crown_gdf = gpd.GeoDataFrame(
            crowns,
            geometry="tree_crown",
            columns=[
                "tree_crown",
                "treetop_height",
                "treetop_unique_ID",
                "treetop_pixel_coords",
            ],
        )

        # Simplify by tolerance value to get smoother crown polygons
        crown_gdf["tree_crown"] = crown_gdf["tree_crown"].simplify(
            tolerance=self.simplify_tolerance, preserve_topology=True
        )

        # Calculate pseudo-confidence scores for the detections
        confidence_scores = calculate_scores(
            "tree_crown", self.confidence_feature, crown_gdf, image.shape
        )

        return (
            crown_gdf,
            confidence_scores,
        )

    def get_tree_crowns_silva(
        self,
        image: np.ndarray,
        all_treetop_pixel_coords: List[Point],
        all_treetop_heights: List[float],
        all_treetop_ids: Optional[List[str]] = None,
    ) -> Tuple[gpd.GeoDataFrame, List[float]]:
        """Generate tree crowns by implementing algorithm described by Silva et al. (2016) for crown segmentation.

        Args:
            image (np.ndarray): A single channel CHM image
            all_treetop_pixel_coords (List[Point]): A list with all detected treetop coordinates in pixel units
            all_treetop_heights (List[float]): A list with treetop heights in the same sequence as the coordinates
            all_treetop_ids: (List[str]) : Detected treetop IDs. Defaults to None

        Returns:
            (GeoDataFrame, List[float]): GeoDataFrame containing crowns, treetops, tree heights and a list with confidence scores
        """

        # Store the individual polygons from Voronoi diagram in the same sequence as the treetop points
        if len(all_treetop_pixel_coords) == 1:
            # This is a special case where the voronoi tesselation would return an "EMPTY" geometry
            # instead, the corresponding geometry is set to the bounding rectangle of the tile
            ordered_polygons = [box(0, 0, image.shape[1], image.shape[0])]
        else:
            # Get Voronoi Diagram from the calculated treetop points
            voronoi_diagram = shapely.voronoi_polygons(
                MultiPoint(all_treetop_pixel_coords)
            )

            ordered_polygons = []
            for treetop_point in all_treetop_pixel_coords:
                for polygon in voronoi_diagram.geoms:
                    # Check if the treetop is inside the polygon
                    if polygon.contains(treetop_point):
                        ordered_polygons.append(polygon)
                        break

        # Create a GeoDataFrame to store information associated with the image
        tile_gdf = gpd.GeoDataFrame(
            {
                "geometry": ordered_polygons,
                "treetop_pixel_coords": all_treetop_pixel_coords,
                "treetop_height": all_treetop_heights,
            }
        )

        if all_treetop_ids is not None:
            tile_gdf["treetop_unique_ID"] = all_treetop_ids

        # Get the image dimensions
        img_h, img_w = image.shape

        treetop_heights = tile_gdf["treetop_height"].to_numpy()
        treetop_coords = gpd.GeoSeries(tile_gdf["treetop_pixel_coords"])
        # Calculate the radius for each treetop based on the height
        radii = (self.radius_factor * treetop_heights) / self.data_resolution
        # Create a circle around each treetop coordinates with the calculated radii
        circles = treetop_coords.buffer(radii)

        all_radius_in_pixels = radii.tolist()
        all_circles = list(circles)
        all_polygon_masks = []

        for circle, treetop_height in zip(all_circles, treetop_heights):
            # Get the circle and bounding box containing the whole circle
            minx, miny, maxx, maxy = circle.bounds

            # Clip to image bounds
            min_row = max(int(miny), 0)
            max_row = min(int(np.ceil(maxy)), img_h)
            min_col = max(int(minx), 0)
            max_col = min(int(np.ceil(maxx)), img_w)

            # Crop a patch around the circle
            patch = image[min_row:max_row, min_col:max_col]

            # Calculate threshold value for the binary mask as a fraction of the treetop height
            threshold = self.threshold_factor * treetop_height
            binary_patch = patch > threshold
            # Convet the mask to a shapely object
            patch_mask_poly = mask_to_shapely(binary_patch, backend=self.backend)

            # When cropping the patch, the coordinates are relative to the patch
            # It needs to be translated back to the global coordinates of the image
            mask_global_poly = shapely.affinity.translate(
                patch_mask_poly, xoff=min_col, yoff=min_row
            )
            all_polygon_masks.append(mask_global_poly)

        # Add the calculated radii, circles and polygon masks to the GeoDataFrame
        tile_gdf["radius_in_pixels"] = all_radius_in_pixels
        tile_gdf["circle"] = all_circles
        tile_gdf["multipolygon_mask"] = all_polygon_masks

        # Fix invalid polygons by buffering 0
        tile_gdf["multipolygon_mask"] = gpd.GeoSeries(
            tile_gdf["multipolygon_mask"]
        ).buffer(0)

        # The final tree crown is computed as the intersection of voronoi polygon, circle, and mask
        tile_gdf["tree_crown"] = (
            gpd.GeoSeries(tile_gdf["geometry"])
            .intersection(gpd.GeoSeries(tile_gdf["circle"]))
            .intersection(gpd.GeoSeries(tile_gdf["multipolygon_mask"]))
        )

        # Remove columns that are not needed in the final output
        tree_crown_gdf = tile_gdf.drop(
            columns=["radius_in_pixels", "circle", "multipolygon_mask", "geometry"]
        )

        # Explode the 'tree_crown' column so that MultiPolygons are split into individual Polygons
        tree_crown_gdf = gpd.GeoDataFrame(
            tree_crown_gdf, geometry="tree_crown", crs=tile_gdf.crs
        )
        tree_crown_gdf_exploded = tree_crown_gdf.explode(ignore_index=True)

        # Retain only the rows where the treetop is within the corresponding tree crown polygon
        tree_crown_gdf_filtered = tree_crown_gdf_exploded[
            tree_crown_gdf_exploded["tree_crown"].contains(
                gpd.GeoSeries(tree_crown_gdf_exploded["treetop_pixel_coords"])
            )
        ]

        # Compute the three attributes of a valid row 1) the geometry is valid 2) the area is
        # greater than 0 and 3) it is a polygon
        valid_geometry = tree_crown_gdf_filtered.is_valid
        nonzero_area = tree_crown_gdf_filtered.area > 0
        is_polygon = tree_crown_gdf_filtered.geom_type == "Polygon"
        # Take the logical and of all three attributes
        valid_rows = valid_geometry & nonzero_area & is_polygon
        # Retain only valid rows
        tree_crown_gdf_cleaned = tree_crown_gdf_filtered[valid_rows].reset_index(
            drop=True
        )

        # Calculate pseudo-confidence scores for the detections
        confidence_scores = calculate_scores(
            "tree_crown", self.confidence_feature, tree_crown_gdf_cleaned, image.shape
        )

        return (
            tree_crown_gdf_cleaned,
            confidence_scores,
        )

    def predict_batch(self, batch):
        """Generate predictions for a batch of samples

        Args:
            batch (dict): A batch from the torchgeo dataloader created with an IntersectionDataset.

        Returns:
            List[List[shapely.geometry]]:
                A list of predictions one per image in the batch. The predictions for each image
                are a list of shapely objects.
            Union[None, List[dict]]:
                Any additional attributes that are predicted (such as class or confidence). Must
                be formatted in a way that can be passed to gpd.GeoPandas data argument.
        """
        # List to store every tile's detections
        batch_detections = []
        batch_detections_data = []

        # Get the tree crown generation method that corresponds to the approach
        if self.approach == "watershed":
            get_tree_crowns = self.get_tree_crowns_watershed

        elif self.approach == "silva":
            get_tree_crowns = self.get_tree_crowns_silva

        else:
            raise ValueError(
                f"The 'approach' value {self.approach} is invalid. Choose 'watershed' or 'silva'."
            )

        for image, treetop, attribute in zip(
            batch["image"], batch["shapes"], batch["attributes"]
        ):
            image = image.squeeze()
            # Set NaN values to zero
            image = np.nan_to_num(image)

            # If attribute is empty, it means that the tile does not have any treetops
            # detected. Such cases get ignored.
            if not attribute:
                continue

            # Get the treetop coordinates and corresponding heights for the tile
            treetop_pixel_coords = [shape[0] for shape in treetop]
            treetop_heights = attribute[self.tree_height_column]

            # Compute the polygon tree crown
            if "unique_ID" in attribute:
                detected_crowns_gdf, confidence_scores = get_tree_crowns(
                    image, treetop_pixel_coords, treetop_heights, attribute["unique_ID"]
                )
            else:
                detected_crowns_gdf, confidence_scores = get_tree_crowns(
                    image,
                    treetop_pixel_coords,
                    treetop_heights,
                )

            batch_detections.append(detected_crowns_gdf["tree_crown"].tolist())

            data = {
                "score": confidence_scores,
                "height": detected_crowns_gdf["treetop_height"].tolist(),
            }

            if "treetop_unique_ID" in detected_crowns_gdf.columns:
                # Create a new column in the RegionDetections
                data["treetop_unique_ID"] = detected_crowns_gdf[
                    "treetop_unique_ID"
                ].tolist()

            batch_detections_data.append(data)

        return batch_detections, batch_detections_data


class GeometricDetector(GeometricTreeTopDetector, GeometricTreeCrownDetector):

    def __init__(
        self,
        a: float = 0.0,
        b: float = 0.0325,
        c: float = 0.25,
        min_ht: int = 5,
        blur_sigma: float = 0.0,
        approach: str = "watershed",
        radius_factor: float = 0.6,
        threshold_factor: float = 0.3,
        confidence_feature: str = "height",
        filter_shape: str = "circle",
        contour_backend: str = "cv2",
        tree_height_column: str = "height",
        simplify_tolerance: float = 2.0,
        postprocessors=None,
    ):
        """Learning-free algorithm to detect tree crowns using CHM data. This first detects the treetops
        using the algorithm of Popescu and Wynne (2004) and then uses either Watershed method or the Silva et al. (2016)
        algorithm to compute the tree crowns. Refer docstring of `GeometricTreeTopDetector` and `GeometricTreeCrownDetector`
        for details about the args."""
        GeometricTreeTopDetector.__init__(
            self,
            a=a,
            b=b,
            c=c,
            min_ht=min_ht,
            blur_sigma=blur_sigma,
            filter_shape=filter_shape,
        )
        GeometricTreeCrownDetector.__init__(
            self,
            approach=approach,
            radius_factor=radius_factor,
            threshold_factor=threshold_factor,
            confidence_feature=confidence_feature,
            contour_backend=contour_backend,
            tree_height_column=tree_height_column,
            min_height=min_ht,
            simplify_tolerance=simplify_tolerance,
            postprocessors=postprocessors,
        )

    def predict_batch(self, batch):
        """Generate predictions for a batch of samples

        Args:
            batch (dict): A batch from the torchgeo dataloader

        Returns:
            List[List[shapely.geometry]]:
                A list of predictions one per image in the batch. The predictions for each image
                are a list of shapely objects.
            Union[None, List[dict]]:
                Any additional attributes that are predicted (such as class or confidence). Must
                be formatted in a way that can be passed to gpd.GeoPandas data argument.
        """
        # Get the tree crown generation method that corresponds to the approach
        if self.approach == "watershed":
            get_tree_crowns = self.get_tree_crowns_watershed

        elif self.approach == "silva":
            get_tree_crowns = self.get_tree_crowns_silva

        # List to store every image's detections
        batch_detections = []
        batch_detections_data = []
        for image in batch["image"]:
            image = image.squeeze()
            # Set NaN values to zero
            image = np.nan_to_num(image)

            # Get the treetop locations from the CHM
            treetop_pixel_coords, treetop_heights = self.get_treetops(image)

            # Compute the polygon tree crown
            detected_crowns_gdf, confidence_scores = get_tree_crowns(
                image, treetop_pixel_coords, treetop_heights
            )
            batch_detections.append(
                detected_crowns_gdf["tree_crown"].tolist()
            )  # List[List[shapely.geometry]]
            batch_detections_data.append({"score": confidence_scores})
        return batch_detections, batch_detections_data


class LightningDetector(Detector):
    model: lightning.LightningModule

    def setup(self):
        # This method should implement setup tasks that are common to all LightningDetectors.
        # Method-specific tasks should be defered to setup_model
        raise NotImplementedError()

    @abstractmethod
    def setup_model(self, param_dict: dict) -> lightning.LightningModule:
        """Set up the lightning model, including loading pretrained weights if required

        Args:
            param_dict (dict): Dictionary of configuration paramters.

        Returns:
            lightning.LightningModule: A configured model
        """
        # Should be implemented in each derived class since it's algorithm-specific
        raise NotImplementedError()

    def setup_trainer(self, param_dict: dict) -> lightning.Trainer:
        """Create a pytorch lightning trainer from a parameter dictionary

        Args:
            param_dict (dict): Dictionary of configuration paramters.

        Returns:
            lightning.Trainer: A configured trainer
        """
        raise NotImplementedError()

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, **kwargs):
        """Train a model

        Args:
            train_dataloader (DataLoader): The training dataloader
            val_dataloader (DataLoader): The validation dataloader
        """
        # Should be implemented here
        raise NotImplementedError()

    def save_model(self, save_file: PATH_TYPE):
        """Save a model to disk

        Args:
            save_file (PATH_TYPE):
                Where to save the model. Containing folders will be created if they don't exist.
        """
        # Should be implemented here
        raise NotImplementedError()


class DeepForestDetector(LightningDetector):

    def __init__(self, module: DeepForestModule, postprocessors=None):
        """Create a DeepForestDetector object
        Args:
            module (DeepForestModule): LightningModule for DeepForest
            postprocessors (list, optional): See docstring for Detector class. Defaults to None.
        """
        super().__init__(postprocessors=postprocessors)
        # Setup steps for LightningModule
        self.setup_model(module)

    def setup_model(self, module: DeepForestModule):
        """Setup the DeepForest model and use latest release.

        Args:
            model (DeepForestModule): LightningModule derived object for DeepForest
        """
        self.lightningmodule = module

    def setup_trainer(self):
        """Create a pytorch lightning trainer from a parameter dictionary

        Args:
            param_dict (dict): Dictionary of configuration paramters

        Returns:
            lightning.Trainer: A configured trainer
        """
        # convert param dict to trainer
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/",
            monitor="box_recall",
            mode="max",
            save_top_k=3,
            filename="box_recall-{epoch:02d}-{box_recall:.2f}",
        )
        logger = TensorBoardLogger(save_dir="logs/")

        trainer = lightning.Trainer(
            logger=logger,
            max_epochs=self.lightningmodule.param_dict["train"]["epochs"],
            enable_checkpointing=self.lightningmodule.param_dict[
                "enable_checkpointing"
            ],
            callbacks=[checkpoint_callback],
        )
        return trainer

    def predict_batch(self, batch):
        """
        Returns:
            List[List[shapely.geometry]]:
                A list of predictions one per image in the batch. The predictions for each image
                are a list of shapely objects.
            Union[None, List[dict]]:
                Any additional attributes that are predicted (such as class or confidence). Must
                be formatted in a way that can be passed to gpd.GeoPandas data argument.
        """
        self.lightningmodule.eval()
        images = batch["image"]
        # DeepForest requires input image pixel values to be normalized to range 0-1
        with torch.no_grad():
            # TODO: Make dataloaders more flexible so that the user can provide the correct data type/range
            if images.min() >= 0 and images.max() <= 1:
                outputs = self.lightningmodule(images[:, :3, :, :])
            else:
                outputs = self.lightningmodule(images[:, :3, :, :] / 255)

        all_geometries = []
        all_data_dicts = []

        for pred_dict in outputs:
            boxes = pred_dict["boxes"].cpu().detach().numpy()
            shapely_boxes = shapely.box(
                boxes[:, 0],
                boxes[:, 1],
                boxes[:, 2],
                boxes[:, 3],
            )
            all_geometries.append(shapely_boxes)

            scores = pred_dict["scores"].cpu().detach().numpy()
            labels = pred_dict["labels"].cpu().detach().numpy()
            all_data_dicts.append({"score": scores, "labels": labels})

        return all_geometries, all_data_dicts

    def train(
        self,
        datamodule: CustomDataModule,
    ):
        """Train a model

        Args:
            model (DeepForestModule): LightningModule for DeepForest
            datamodule (CustomDataModule): LightningDataModule that creates train-val-test dataloaders
        """

        # Create and configure lightning.Trainer
        self.trainer = self.setup_trainer()

        # Begin training
        self.trainer.fit(self.lightningmodule, datamodule)

    def save_model(self, save_file: PATH_TYPE):
        """Save a model to disk

        Args:
            save_file (PATH_TYPE):
                Where to save the model. Containing folders will be created if they don't exist.
        """
        # Should be implemented here
        raise NotImplementedError()


class MaskRCNNDetector(LightningDetector):
    """Detectron2 Mask R-CNN detector supporting any module with a `.cfg` attribute.

    Accepts both Detectree2Module and TCDModule. If cfg.MODEL.WEIGHTS is not
    a local path it is treated as a HuggingFace Hub repo ID and model.pth is
    downloaded automatically.
    """

    def __init__(self, module, postprocessors=None):
        """
        Args:
            module: A module with a `.cfg` attribute (e.g. Detectree2Module or TCDMModule).
            postprocessors (list, optional): See Detector base class. Defaults to None.
        """
        super().__init__(postprocessors=postprocessors)
        if DETECTRON2_AVAILABLE is False:
            raise ImportError(
                "MaskRCNNDetector requires detectron2. Please install it to use this detector."
            )
        self.module = module
        self.setup_predictor()

    def setup_predictor(self):
        """Build the Detectron2 model and load weights.

        If cfg.MODEL.WEIGHTS is a HuggingFace repo ID (not a local path),
        downloads model.pth from the Hub before loading.
        """
        self.cfg = self.module.cfg.clone()

        weights = self.cfg.MODEL.WEIGHTS
        if not os.path.exists(weights):
            try:
                from huggingface_hub import hf_hub_download

                logging.info("Downloading checkpoint from HuggingFace Hub: %s", weights)
                weights = hf_hub_download(repo_id=weights, filename="model.pth")
                self.cfg.MODEL.WEIGHTS = weights
            except Exception as exc:
                logging.warning(
                    "Could not download checkpoint '%s' from HuggingFace Hub: %s",
                    self.cfg.MODEL.WEIGHTS,
                    exc,
                )

        self.model = build_model(self.cfg)
        self.model.to(self.cfg.MODEL.DEVICE)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)

        # ResizeShortestEdge with MIN_SIZE_TEST=0 only caps at MAX_SIZE_TEST
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST,
        )
        self.input_format = self.cfg.INPUT.FORMAT

    def call_predict(self, batch):
        """Preprocess a batch and run Detectron2 inference.

        Args:
            batch (Tensor): Float tensor [N, C, H, W] from the dataloader.
                            Values may be in [0, 1] or [0, 255].

        Returns:
            List of dicts (one per image) each containing an "instances" key.
        """
        with torch.no_grad():
            inputs = []
            for image in batch:
                # Rescale [0, 1] to [0, 255]
                if image.min() >= 0 and image.max() <= 1:
                    image = (image * 255).byte()

                # Drop extra channels beyond RGB
                if image.shape[0] != 3:
                    image = image[:3]

                # HWC numpy for the Detectron2 augmentation API
                image_np = image.permute(1, 2, 0).numpy()

                # COCO-pretrained Detectron2 models expect BGR; flip from the RGB dataloader
                if self.input_format == "BGR":
                    image_np = np.flip(image_np, axis=2)

                height, width = image_np.shape[:2]
                image_np = self.aug.get_transform(image_np).apply_image(image_np)
                tensor = torch.as_tensor(
                    image_np.astype("float32").transpose(2, 0, 1)
                ).to(self.cfg.MODEL.DEVICE)

                inputs.append({"image": tensor, "height": height, "width": width})

            return self.model(inputs)

    def predict_batch(self, batch):
        """Run inference on one batch of tiles.

        Args:
            batch (defaultDict): A batch from the dataloader with an "image" key
                                 (tensor of shape [N, C, H, W]).

        Returns:
            all_geometries (List[List[shapely.Geometry]]):
                One list of shapely geometries per image in the batch.
            all_data_dicts (List[dict]):
                One dict per image with keys "score", "labels", and "bbox".
        """
        images = batch["image"]
        batch_preds = self.call_predict(images)

        all_geometries = []
        all_data_dicts = []

        for pred in batch_preds:
            instances = pred["instances"].to("cpu")

            # Convert predicted masks to shapely geometries
            pred_masks = instances.pred_masks.numpy()
            shapely_objects = [mask_to_shapely(m) for m in pred_masks]

            # Convert predicted boxes to shapely boxes
            np_bboxes = instances.pred_boxes.tensor.numpy()
            shapely_bboxes = shapely.box(
                xmin=np_bboxes[:, 0],
                ymin=np_bboxes[:, 1],
                xmax=np_bboxes[:, 2],
                ymax=np_bboxes[:, 3],
            )

            all_geometries.append(shapely_objects)
            all_data_dicts.append(
                {
                    "score": instances.scores.numpy(),
                    "labels": instances.pred_classes.numpy(),
                    "bbox": shapely_bboxes,
                }
            )

        return all_geometries, all_data_dicts
