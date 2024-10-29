from abc import abstractmethod
from typing import Any, DefaultDict, Iterator, List, Tuple, Union
from typing import Optional
from typing import List, Dict, Union
from typing import List, Dict, Union, DefaultDict, Any
import warnings
import logging
import os
import warnings
from abc import abstractmethod
from typing import Any, DefaultDict, Dict, List, Union

import lightning
import numpy as np
import pandas as pd
import shapely
import torch
import torchvision
from deepforest import main
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, optim
from torch.utils.data import DataLoader
from torchgeo.datasets import unbind_samples
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import (
    AnchorGenerator,
    RetinaNet,
    RetinaNet_ResNet50_FPN_Weights,
)

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.detection.region_detections import RegionDetections, RegionDetectionsSet
from tree_detection_framework.utils.detection import use_release_df
from tree_detection_framework.preprocessing.preprocessing import CustomDataModule

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
        for batch in inference_dataloader:
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
        image_bounds = shapely.box(
            xmin=0, ymin=0, xmax=image_shape[1], ymax=image_shape[0]
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

    def predict(
        self, inference_dataloader: DataLoader, **kwargs
    ) -> RegionDetectionsSet:
        # Should be implemented here
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
    
    @staticmethod
    def get_image_bounds_as_shapely(batch: DefaultDict[str, Any]) -> List[shapely.geometry.Polygon]:
        """Get pixel image bounds as shapely objects from a batch.
        
        Args:
            batch: (DefaultDict[str, Any]): Batch from DataLoader with image sample(s).

        Returns:
            List[shapely.geometry.Polygon]: A list of shapely Polygons representing the pixel bounds.
        """
        image_shape = batch["image"].shape[-2:]
        image_bounds = shapely.box(xmin=0, ymin=0, xmax=image_shape[1], ymax=image_shape[0])
        return [image_bounds] * batch["image"].shape[0]

    @staticmethod
    def get_geospatial_bounds_as_shapely(batch: DefaultDict[str, Any]) -> List[shapely.geometry.Polygon]:
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
                ymax=tile_bounds.maxy
            )
            for tile_bounds in batch_bounds
        ]

class RetinaNetModel:
    def __init__(self, param_dict):
        self.param_dict = param_dict

    def load_backbone(self):
        """A torch vision retinanet model"""
        backbone = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)

        return backbone

    def create_anchor_generator(self,
                                sizes=((8, 16, 32, 64, 128, 256, 400),),
                                aspect_ratios=((0.5, 1.0, 2.0),)):
        """
        Create anchor box generator as a function of sizes and aspect ratios
        Documented https://github.com/pytorch/vision/blob/67b25288ca202d027e8b06e17111f1bcebd2046c/torchvision/models/detection/anchor_utils.py#L9
        let's make the network generate 5 x 3 anchors per spatial
        location, with 5 different sizes and 3 different aspect
        ratios. We have a Tuple[Tuple[int]] because each feature
        map could potentially have different sizes and
        aspect ratios
        Args:
            sizes:
            aspect_ratios:

        Returns: anchor_generator, a pytorch module

        """
        anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)

        return anchor_generator

    def create_model(self):
        """Create a retinanet model
        Args:
            num_classes (int): number of classes in the model
            nms_thresh (float): non-max suppression threshold for intersection-over-union [0,1]
            score_thresh (float): minimum prediction score to keep during prediction  [0,1]
        Returns:
            model: a pytorch nn module
        """
        resnet = self.load_backbone()
        backbone = resnet.backbone

        model = RetinaNet(backbone=backbone, num_classes=self.param_dict["num_classes"])
        # model.nms_thresh = self.param_dict["nms_thresh"]
        # model.score_thresh = self.param_dict["retinanet"]["score_thresh"]

        return model
    
class DeepForestModule(lightning.LightningModule):
    def __init__(self, param_dict: Dict[str, Any]):
        super().__init__() 
        self.param_dict = param_dict

        if param_dict['backbone'] == 'retinanet':
            retinanet = RetinaNetModel(param_dict)
        else:
            raise ValueError("Only 'retinanet' backbone is currently supported.")
        
        self.model = retinanet.create_model()

    def use_release(self, check_release=True):
        """Use the latest DeepForest model release from github and load model.
        Optionally download if release doesn't exist.
        Args:
            check_release (logical): whether to check github for a model recent release. 
            In cases where you are hitting the github API rate limit, set to False and any local model will be downloaded.
            If no model has been downloaded an error will raise.
        """
        # Download latest model from github release
        release_tag, self.release_state_dict = use_release_df(
            check_release=check_release)
        self.model.load_state_dict(torch.load(self.release_state_dict))

        # load saved model and tag release
        self.__release_version__ = release_tag
        print("Loading pre-built model: {}".format(release_tag))

    def forward(self, images: List[Tensor], targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """ Calls the model's forward method.
        Args:
            images (list[Tensor]): Images to be processed
            targets (list[Dict[Tensor]]): Ground-truth boxes present in the image

        Returns:
            result (list[BoxList] or dict[Tensor]):
                The output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
        """
        return self.model.forward(images, targets)  # Model specific forward
    
    def training_step(self, batch):
        # Ensure model is in train mode
        self.model.train()
        device = next(self.model.parameters()).device

        # Image is expected to be a list of tensors, each of shape [C, H, W] in 0-1 range.
        image_batch = (batch['image'][:, :3, :, :] / 255.0).to(device)
        image_batch_list = [image_batch[i] for i in range(image_batch.size(0))]

        # To store every image's target - a dictionary containing `boxes` and `labels`
        targets = []
        for tile in batch['bounding_boxes']:
            # Convert from list to FloatTensor[N, 4]
            boxes_tensor = torch.tensor(tile, dtype=torch.float32).to(device)
            # Need to remove boxes that go out-of-bounds. Has negative values.
            valid_mask = (boxes_tensor >= 0).all(dim=1)
            filtered_boxes_tensor = boxes_tensor[valid_mask]
            # Create a label tensor. Single class for now.
            class_labels = torch.zeros(filtered_boxes_tensor.shape[0], dtype=torch.int64).to(device)
            # Dictionary for the tile
            d = {"boxes": filtered_boxes_tensor, "labels": class_labels}
            targets.append(d)
    
        loss_dict = self.forward(image_batch_list, targets)

        final_loss = sum([loss for loss in loss_dict.values()])
        print('loss: ',final_loss)
        return final_loss


    def configure_optimizers(self):
        # similar to the one in deepforest
        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.param_dict["train"]["lr"],
                              momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.1,
                                                               patience=10,
                                                               verbose=True,
                                                               threshold=0.0001,
                                                               threshold_mode='rel',
                                                               cooldown=0,
                                                               min_lr=0,
                                                               eps=1e-08)
        # # Monitor rate is val data is used
        # if self.config["validation"]["csv_file"] is not None:
        #     return {
        #         'optimizer': optimizer,
        #         'lr_scheduler': scheduler,
        #         "monitor": 'val_classification'
        #     }
        # else:
        #     return optimizer
        return optimizer
    

class DeepForestDetector(LightningDetector):

    def setup_model(self, model: DeepForestModule):
        """Setup the DeepForest model and use latest release.
        
        Args:
            model (DeepForestModule): LightningModule derived object for DeepForest
        """
        self.model = model
        self.model.use_release()

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
            filename="box_recall-{epoch:02d}-{box_recall:.2f}"
        )
        logger = TensorBoardLogger(save_dir="logs/")

        trainer = lightning.Trainer(logger=logger,
                                  max_epochs=self.param_dict["train"]["epochs"],
                                  enable_checkpointing=self.param_dict["enable_checkpointing"],
                                  callbacks=[checkpoint_callback]
                                  )
        return trainer
        

    def predict(
        self, inference_dataloader: DataLoader, **kwargs
    ) -> RegionDetectionsSet:
        """ Get predictions from `DeepForest` for the given dataloader as a `RegionDetectionsSet`.

        Args:
            inference_dataloader (DataLoader): PyTorch DataLoader with imput samples containing images and metadata.

        Returns:
            `RegionDetectionsSet`: A collection of region detection objects, where each detection object contains: 
            - Predicted bounding geometries (in shapely format) for detected regions 
            - Associated prediction dataframe
            - Pixel-based and geospatial bounding boxes for each detection 
            - CRS information
        """
        # Load deepforest model
        model = main.deepforest()
        # Get the latest model weights
        # TODO: Support loading a different pretrained model from checkpoint
        model.use_release()

        # Store all batched elemts from the dataloader in a list
        all_tiles = list(inference_dataloader)
        logging.info("Getting predictions from deepforest...")
        output_dataframes = []
        for tile in all_tiles:
            image = tile['image'][0] # assumes batch_size 1
            # Convert to datatype expected by deepforest
            image_np = image.numpy().astype(np.float32)
            # Retain first 3 channels and reorder it so that RGB channels are the last dimension
            image_np = image_np[:3, :, :].transpose(1, 2, 0)
            # Prediction output from deepforest is returned as a dataframe
            output = model.predict_image(image = image_np)
            output_dataframes.append(output)

        # Create a list of RegionDetection objects
        region_detections = []
        logging.info("Converting predictions to RegionDetectionsSet object...")
        for sample, prediction in zip(inference_dataloader, output_dataframes):
            # Extract the derived attributes from the sample and prediction
            # Note that the first element is taken from the ones where a batch is returned
            image_bounds = LightningDetector.get_image_bounds_as_shapely(sample)[0]
            geospatial_bounds = LightningDetector.get_geospatial_bounds_as_shapely(sample)[0]
            prediction_geometry = self.parse_deepforest_output(prediction)

            # Extract the CRS of the first (only) element in the batch
            CRS = sample["crs"][0]

            # Create the region detection
            region_detection = RegionDetections(
                detection_geometries=prediction_geometry,
                data=prediction,
                pixel_prediction_bounds=image_bounds,
                geospatial_prediction_bounds=geospatial_bounds,
                input_in_pixels=True,
                CRS=CRS,
            )
            # Append to the list
            region_detections.append(region_detection)
        
        logging.info("Done.")
        # Return the region detection set
        return RegionDetectionsSet(region_detections)

    def train(self, model: DeepForestModule, datamodule: CustomDataModule, param_dict: Dict[str, Any]):
        
        """Train a model

        Args:
            model (DeepForestModule): LightningModule for DeepForest
            datamodule (CustomDataModule): LightningDataModule that creates train-val-test dataloaders
        """
        # Setup steps for LightningModule
        self.setup_model(model)
        self.param_dict = param_dict

        # Create and configure lightning.Trainer
        self.trainer = self.setup_trainer()

        # Begin training
        self.trainer.fit(model, datamodule)


    def save_model(self, save_file: PATH_TYPE):
        """Save a model to disk

        Args:
            save_file (PATH_TYPE):
                Where to save the model. Containing folders will be created if they don't exist.
        """
        # Should be implemented here
        raise NotImplementedError()
    
    @staticmethod
    def parse_deepforest_output(prediction: pd.DataFrame) -> shapely.geometry.Polygon:
        """Parse DeepForest output into shapely geometries.
        
        Args:
            prediction (pd.DataFrame): Dataframe output containing `xmin`, `ymin`, `xmax`, `ymax` attributes.

        Returns:
            numpy.ndarray consisting of all detections in a tile as `Polygon` objects.
        """
        xmin = prediction["xmin"].to_numpy()
        ymin = prediction["ymin"].to_numpy()
        xmax = prediction["xmax"].to_numpy()
        ymax = prediction["ymax"].to_numpy()
        return shapely.box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
