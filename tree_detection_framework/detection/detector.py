import logging
import warnings
from abc import abstractmethod
from typing import Any, DefaultDict, Dict, Iterator, List, Optional, Tuple, Union

import detectron2.data.transforms as T
import lightning
import numpy as np
import pandas as pd
import shapely
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.detection.models import DeepForestModule
from tree_detection_framework.detection.region_detections import (
    RegionDetections,
    RegionDetectionsSet,
)
from tree_detection_framework.preprocessing.derived_geodatasets import CustomDataModule
from tree_detection_framework.utils.detection import use_release_df
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
        return batch["crs"][0]


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
        with torch.no_grad():
            outputs = self.lightningmodule(images[:, :3, :, :] / 255)  # .model ??

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
        self.module = module

    def setup_model(self):
        raise NotImplementedError()

    def setup_trainer(self):
        raise NotImplementedError()
    
    def setup_predictor(self):
        """Build predictor model architecture and load model weights from config.
        Based on `__init__` of `DefaultPredictor` from `detectron2`"""

        self.cfg = self.module.cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)

        # Set the model to eval mode
        self.model.eval()
        if len(self.module.cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(self.module.cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.module.cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [self.module.cfg.INPUT.MIN_SIZE_TEST, self.module.cfg.INPUT.MIN_SIZE_TEST], self.module.cfg.INPUT.MAX_SIZE_TEST
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
        self.setup_predictor()

        with torch.no_grad():
            inputs = []
            for original_image in batch:
                original_image = original_image.permute(1, 2, 0).byte().numpy()
                original_image = original_image[:, :, :3]
                print("Original image shape = ", original_image.shape)
                
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                image.to(self.cfg.MODEL.DEVICE)
                print("Final image shape = ", image.shape)
                
                input = {"image": image, "height": height, "width": width}
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

    def train(
        self,
        datamodule: CustomDataModule,
    ):
        raise NotImplementedError()

    def save_model(self, save_file: PATH_TYPE):
        """Save a model to disk

        Args:
            save_file (PATH_TYPE):
                Where to save the model. Containing folders will be created if they don't exist.
        """
        # Should be implemented here
        raise NotImplementedError()
