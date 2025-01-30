import logging
from abc import abstractmethod
from itertools import groupby
from typing import Any, DefaultDict, Dict, Iterator, List, Optional, Tuple, Union

import detectron2.data.transforms as T
import geopandas as gpd
import lightning
import numpy as np
import pyproj
import shapely
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from scipy.ndimage import maximum_filter
from shapely.geometry import (
    GeometryCollection,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.detection.models import DeepForestModule
from tree_detection_framework.detection.region_detections import (
    RegionDetections,
    RegionDetectionsSet,
)
from tree_detection_framework.preprocessing.derived_geodatasets import (
    CustomDataModule,
    bounding_box,
)
from tree_detection_framework.utils.geometric import mask_to_shapely

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Detector:
    @abstractmethod
    def setup(self):
        """Any setup tasks that should be performed once when the Detector is instantiated"""
        # This should not be implemented here unless there are setup tasks that are shared by every detector
        raise NotImplementedError()

    def predict_as_generator(
        self, inference_dataloader: DataLoader, **kwargs
    ) -> Iterator[RegionDetections]:
        """
        A generator that yields a RegionDetections object for each image in the dataloader. Note
        that the dataloader may have batched data but predictions will be returned individually.

        Args:
            inference_dataloader (DataLoader): Dataloader to generate predictions for
        """
        # Iterate over each batch in the dataloader
        for batch in tqdm(
            inference_dataloader, desc="Performing prediction on batches"
        ):

            # This is the expensive step, generate the predictions using predict_batch from the
            # derived class. The additional arguments are also passed to this method with kwargs
            batch_preds_geometries, batch_preds_data = self.predict_batch(
                batch, **kwargs
            )

            # If the prediction doesn't generate any data, set it to a list of None for
            # compatability with downstream steps
            if batch_preds_data is None:
                batch_preds_data = [None] * len(batch_preds_geometries)

            # Extract attributes from the batch
            batch_image_bounds = self.get_image_bounds_as_shapely(batch)
            batch_geospatial_bounds = self.get_geospatial_bounds_as_shapely(batch)
            CRS = self.get_CRS_from_batch(batch)

            # Iterate over samples in the batch so we can yield them one at a time
            for preds_geometry, preds_data, image_bounds, geospatial_bounds in zip(
                batch_preds_geometries,
                batch_preds_data,
                batch_image_bounds,
                batch_geospatial_bounds,
            ):
                # Create a region detections object
                region_detections = RegionDetections(
                    detection_geometries=preds_geometry,
                    data=preds_data,
                    CRS=CRS,
                    input_in_pixels=True,
                    pixel_prediction_bounds=image_bounds,
                    geospatial_prediction_bounds=geospatial_bounds,
                )
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
    def get_image_bounds_as_shapely(
        batch: DefaultDict[str, Any]
    ) -> List[shapely.geometry.Polygon]:
        """Get pixel image bounds as shapely objects from a batch.
        Args:
            batch: (DefaultDict[str, Any]): Batch from DataLoader with image sample(s).
        Returns:
            List[shapely.geometry.Polygon]: A list of shapely Polygons representing the pixel bounds.
        """
        image_shape = batch["image"].shape[-2:]
        # Note that the y min bounds are reversed from the expected convention. This is because
        # they are measured in pixel coordinates, which start at the top and go down. So this
        # convention matches how the geospatial bounding box is represented.
        image_bounds = shapely.box(
            xmin=0, ymin=image_shape[0], xmax=image_shape[1], ymax=0
        )
        return [image_bounds] * batch["image"].shape[0]

    @staticmethod
    def get_geospatial_bounds_as_shapely(
        batch: DefaultDict[str, Any]
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


class GeometricDetector(Detector):

    def __init__(
        self,
        a: float = 0.00901,
        b: float = 0,
        c: float = 2.52503,
        res: float = 0.2,
        min_ht: int = 5,
        radius_factor: float = 0.6,
        threshold_factor: float = 0.3,
        confidence_factor: str = "height",
        filter_shape: str = "circle",
        contour_backend: str = "cv2",
    ):
        """Create a GeometricDetector object

        Args:
            a (float, optional): Coefficient for the quadratic term in the radius calculation. Defaults to 0.00901.
            b (float, optional): Coefficient for the linear term in the radius calculation. Defaults to 0.
            c (float, optional): Constant term in the radius calculation. Defaults to 2.52503.
            res (float, optional): Resolution of the CHM image. Defaults to 0.2.
            min_ht (int, optional): Minimum height for a pixel to be considered as a tree. Defaults to 5.
            radius_factor (float, optional): Factor to determine the radius of the tree crown. Defaults to 0.6.
            threshold_factor (float, optional): Factor to determine the threshold for the binary mask. Defaults to 0.3.
            confidence_factor (str, optional): Feature to use to compute the confidence scores for the predictions.
                Choose from "height", "area", "distance", "all". Defaults to "height".
            filter_shape (str, optional): Shape of the filter to use for local maxima detection.
                Choose from "circle", "square", "none". Defaults to "circle". Defaults to "circle".
            contour_backend (str, optional): The backend to use for contour extraction to generate treecrowns.
                Choose from "cv2" and "contourpy".

        """
        self.a = a
        self.b = b
        self.c = c
        self.res = res
        self.min_ht = min_ht
        self.radius_factor = radius_factor
        self.threshold_factor = threshold_factor
        self.confidence_factor = confidence_factor
        self.filter_shape = filter_shape
        self.backend = contour_backend

    # TODO: See creating the height mask is more efficient by first cropping the tile CHM to the maximum possible bounds of the tree crown,
    # as opposed to applying the mask to the whole tile CHM for each tree

    def calculate_scores(
        self, tile_gdf: gpd.GeoDataFrame, image_shape: tuple
    ) -> List[float]:
        """Calculate pseudo-confidence scores for the detections based on the following features of the tree crown:

            1. Height - Taller trees are generally more easier to detect, making their confidence higher
            2. Area - Larger tree crowns are easier to detect, hence less likely to be false positives
            3. Distance - Trees near the edge of a tile might have incomplete data, reducing confidence
            4. All - An option to compute a weighted combination of all factors as the confidence score

        Args:
            tile_gdf (gpd.GeoDataFrame): A geopandas dataframe with 'treetop_height' and 'tree_crown' columns
            image_shape (tuple): The (i, j, channel) shape of the image that predictions were generated from

        Returns:
            List[float]: Calculated confidence scores.
        """
        if self.confidence_factor not in ["height", "area", "distance", "all"]:
            raise ValueError(
                "Invalid confidence_factor provided. Choose from: `height`, `area`, `distance`, `all`"
            )

        if self.confidence_factor == "height":
            # Use height values as scores
            confidence_scores = tile_gdf["treetop_height"]

        elif self.confidence_factor == "area":
            # Use area values as scores
            confidence_scores = tile_gdf["tree_crown"].apply(lambda geom: geom.area)

        elif self.confidence_factor == "distance":
            # Calculate the centroid of each tree crown
            tile_gdf["centroid"] = tile_gdf["tree_crown"].apply(
                lambda geom: geom.centroid if not geom.is_empty else None
            )

            # Calculate distances to the closest edge for each centroid
            def calculate_edge_distance(centroid):
                if centroid is None:  # Check if centroid is None (empty geometry case)
                    return 0
                x, y = centroid.x, centroid.y
                distances = [
                    x,  # left edge
                    image_shape[1] - x,  # right edge
                    y,  # bottom edge
                    image_shape[0] - y,  # top edge
                ]
                # Return the distance to the closest edge
                return min(distances)

            tile_gdf["edge_distance"] = tile_gdf["centroid"].apply(
                calculate_edge_distance
            )
            # Use edge distance values as scores
            confidence_scores = tile_gdf["edge_distance"]

        return list(confidence_scores)

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
        min_radius_pixels = int(np.floor(min_radius / self.res))

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
            radius_pixels = radius / self.res
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
                all_treetop_pixel_coords.append(Point(j, i))
                all_treetop_heights.append(ht)

        return all_treetop_pixel_coords, all_treetop_heights

    def get_tree_crowns(
        self,
        image: np.ndarray,
        all_treetop_pixel_coords: List[Point],
        all_treetop_heights: List[float],
    ) -> Tuple[List[shapely.Polygon], List[float]]:
        """Generate tree crowns for an image.

        Args:
            image (np.ndarray): A single channel CHM image
            all_treetop_pixel_coords (List[Point]): A list with all detected treetop coordinates in pixel units
            all_treetop_heights (List[float]): A list with treetop heights in the same sequence as the coordinates

        Returns:
            filtered_crowns (List[shapely.Polygon]): Detected tree crowns as shapely polygons/multipolygons
            confidence_scores (List[float]): Pseudo-confidence scores for the detections
        """

        # Get Voronoi Diagram from the calculated treetop points
        voronoi_diagram = shapely.voronoi_polygons(MultiPoint(all_treetop_pixel_coords))

        # Store the individual polygons from Voronoi diagram in the same sequence as the treetop points
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

        # Next, we get 2 new sets of polygons:
        # 1. A circle for every detected treetop
        # 2. A set of multipolygons geenrated from the binary mask of the image
        all_radius_in_pixels = []
        all_circles = []
        all_polygon_masks = []

        for treetop_point, treetop_height in zip(
            tile_gdf["treetop_pixel_coords"], tile_gdf["treetop_height"]
        ):
            # Compute radius as a fraction of the height, divide by resolution to convert unit to pixels
            radius = (self.radius_factor * treetop_height) / self.res
            all_radius_in_pixels.append(radius)

            # Create a circle by buffering it by the radius value and add to list
            all_circles.append(treetop_point.buffer(radius))

            # Calculate threshold value for the binary mask as a fraction of the treetop height
            threshold = self.threshold_factor * treetop_height
            # Thresholding the tile image
            binary_mask = image > threshold
            # Convert the mask to shapely polygons, returned as a MultiPolygon
            shapely_polygon_mask = mask_to_shapely(binary_mask, backend=self.backend)
            all_polygon_masks.append(shapely_polygon_mask)

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

        filtered_crowns = []
        indices_to_drop = []

        for index, (tree_crown, treetop_point) in enumerate(
            zip(tile_gdf["tree_crown"], tile_gdf["treetop_pixel_coords"])
        ):
            # Only keep valid polygons
            if (
                isinstance(tree_crown, Polygon)
                and tree_crown.is_valid
                and tree_crown.area > 0
            ):
                filtered_crowns.append(tree_crown)
            elif isinstance(tree_crown, (MultiPolygon, GeometryCollection)):
                # Iterate through each polygon in the MultiPolygon
                for i, geom in enumerate(tree_crown.geoms):
                    # Exclude LineString/Point/MultiPoint objects in case geom is a GeometryCollection
                    if isinstance(geom, Polygon) and geom.is_valid and geom.area > 0:
                        # If the polygon with the treetop has been found, add it to list and ignore all other polygons
                        if geom.contains(treetop_point):
                            filtered_crowns.append(geom)
                            break
                    # For other cases, add an empty polygon to avoid having a mismatch in number of rows in the gdf
                    if i == (len(tree_crown.geoms) - 1):
                        filtered_crowns.append(Polygon())
            else:
                # If tree_crown is not valid, mark the row for deletion
                indices_to_drop.append(index)

        # Drop the rows with invalid polygons
        tile_gdf = tile_gdf.drop(indices_to_drop)

        # Calculate pseudo-confidence scores for the detections
        confidence_scores = self.calculate_scores(tile_gdf, image.shape)

        return filtered_crowns, confidence_scores

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
        # List to store every image's detections
        batch_detections = []
        batch_detections_data = []
        for image in batch["image"]:
            image = image.squeeze()
            # Set NaN values to zero
            image = np.nan_to_num(image)

            treetop_pixel_coords, treetop_heights = self.get_treetops(image)
            final_tree_crowns, confidence_scores = self.get_tree_crowns(
                image, treetop_pixel_coords, treetop_heights
            )
            batch_detections.append(final_tree_crowns)  # List[List[shapely.geometry]]
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

    def __init__(self, module: DeepForestModule):
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


class Detectree2Detector(LightningDetector):

    def __init__(self, module):
        # TODO: Add lightning module implementation
        # Note: For now, `module` only references to `cfg`
        self.module = module
        self.setup_predictor()

    def setup_predictor(self):
        """Build predictor model architecture and load model weights from config.
        Based on `__init__` of `DefaultPredictor` from `detectron2`"""

        self.cfg = self.module.cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)

        self.model.to(self.cfg.MODEL.DEVICE)

        # Set the model to eval mode
        self.model.eval()
        if len(self.module.cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(self.module.cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.module.cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [self.module.cfg.INPUT.MIN_SIZE_TEST, self.module.cfg.INPUT.MIN_SIZE_TEST],
            self.module.cfg.INPUT.MAX_SIZE_TEST,
        )

        self.input_format = self.module.cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def call_predict(self, batch):
        """
        Largely based on `__call__` method of `DefaultPredictor` from `detectron2` for single image prediction.
        Modifications have been made to preprocess images from the torchgeo dataloader and predict for a batch of images.

        Args:
            batch (Tensor): 4 dims Tensor with the first dimension having number of images in the batch

        Returns:
            batch_preds (List[Dict[str, Instances]]): An iterable with a dictionary per image having "instances" value
            as an `Instances` object containing prediction results.
        """

        with torch.no_grad():
            inputs = []
            if batch.shape[1] == 3:  # RGB image
                for original_image in batch:
                    height, width = original_image.shape[1], original_image.shape[2]

                    # Convert image pixel values to 0-255 range
                    if original_image.min() >= 0 and original_image.max() <= 1:
                        original_image = original_image * 255

                    # Create a dict with each image and its properties
                    input = {"image": original_image, "height": height, "width": width}

                    # Add the dictionary to batch image list
                    inputs.append(input)
            else:
                for original_image in batch:
                    original_image = original_image.permute(1, 2, 0).byte().numpy()
                    original_image = original_image[:, :, :3]

                    height, width = original_image.shape[:2]
                    # Resize the image if required
                    image = self.aug.get_transform(original_image).apply_image(
                        original_image
                    )
                    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                    image = image.to(self.cfg.MODEL.DEVICE)
                    # Create a dict with each image and its properties
                    input = {"image": image, "height": height, "width": width}
                    # Add the dictionary to batch image list
                    inputs.append(input)

            batch_preds = self.model(inputs)
            return batch_preds

    def predict_batch(self, batch):
        """
        Get predictions for a batch of images.

        Args:
            batch (defaultDict): A batch from the dataloader

        Returns:
            all_geometries (List[List[shapely.MultiPolygon]]):
                A list of predictions one per image in the batch. The predictions for each image
                are a list of shapely objects.
            all_data_dicts (Union[None, List[dict]]):
                Predicted scores and classes
        """
        # Get images from batch
        images = batch["image"]
        batch_preds = self.call_predict(images)

        # To store all predicted polygons
        all_geometries = []
        # To store other related information such as scores and labels
        all_data_dicts = []

        # Iterate through predictions for each tile in the batch
        for pred in batch_preds:

            # Get the Instances object
            instances = pred["instances"].to("cpu")

            # Get the predicted masks for this tile
            pred_masks = instances.pred_masks.numpy()
            # Convert each mask to a shapely multipolygon
            shapely_objects = [mask_to_shapely(pred_mask) for pred_mask in pred_masks]
            all_geometries.append(shapely_objects)

            # Get prediction scores
            scores = instances.scores.numpy()
            # Get predicted classes
            labels = instances.pred_classes.numpy()
            all_data_dicts.append({"score": scores, "labels": labels})

        return all_geometries, all_data_dicts
