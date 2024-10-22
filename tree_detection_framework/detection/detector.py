from abc import abstractmethod
from typing import Optional
import warnings

import os
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchgeo.datasets import unbind_samples
import torch
from deepforest import main
from torch import optim

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.detection.region_detections import RegionDetectionsSet
from tree_detection_framework.utils.detection import use_release_df

import torchvision
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.retinanet import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class Detector:
    @abstractmethod
    def setup(self):
        """Any setup tasks that should be performed once when the Detector is instantiated"""
        # This should not be implemented here unless there are setup tasks that are shared by every detector
        raise NotImplementedError()

    @abstractmethod
    def predict(
        self, inference_dataloader: DataLoader, **kwargs
    ) -> RegionDetectionsSet:
        """Generates predictions for each tile in a dataset

        Args:
            inference_dataloader (DataLoader): Dataloader to generate predictions for

        Returns:
            RegionDetectionsSet: One prediction per tile
        """
        # This should not be implemented here unless there are prediction tasks shared across every detector
        raise NotImplementedError()

class RetinaNetModel:
    # deepforest.models.retinanet
    def __init__(self, param_dict):
        self.param_dict = param_dict

    def load_backbone(self):
        """A torch vision retinanet model"""
        backbone = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)

        return backbone

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
    

class LightningDetector(Detector):

    def __init__(self, model, param_dict):
        self.model = model
        self.model.use_release()
        self.param_dict = param_dict
        self.trainer = self.setup_trainer() # TODO: move it into a function; on-demand trainer

    def setup_trainer(self):
        """Create a pytorch lightning trainer from a parameter dictionary

        Args:
            param_dict (dict): Dictionary of configuration paramters.

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
        # Should be implemented here
        # self.model loop
        raise NotImplementedError()

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        
        """Train a model

        Args:
            train_dataloader (DataLoader): The training dataloader
            val_dataloader (DataLoader): The validation dataloader
        """
        self.trainer.fit(self.model, train_dataloader, val_dataloader)


    def save_model(self, save_file: PATH_TYPE):
        """Save a model to disk

        Args:
            save_file (PATH_TYPE):
                Where to save the model. Containing folders will be created if they don't exist.
        """
        # Should be implemented here
        raise NotImplementedError()

class DeepForestModule(lightning.LightningModule):
    # subclass of pl.LightningModule
    def __init__(self, param_dict):
        # do the model setup here
        super().__init__() 
        self.param_dict = param_dict
        if param_dict['backbone'] == 'retinanet':
            retinanet = RetinaNetModel(param_dict)
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
            # self.config["architecture"] = "retinanet"
            # self.create_model()
        self.model.load_state_dict(torch.load(self.release_state_dict))

        # load saved model and tag release
        self.__release_version__ = release_tag
        print("Loading pre-built model: {}".format(release_tag))

    def forward(self, images, targets):
        # self.model.forward plus post processing if needed within forward()
        return self.model.forward(images, targets)  # Model specific forward
      
    def training_step(self, batch):
        # training_step, similar to the one in deepforest. below are steps from deepforest:

        # # Confirm model is in train mode
        # self.model.train()

        # # allow for empty data if data augmentation is generated
        # path, images, targets = batch

        # loss_dict = self.model.forward(images, targets)

        # # sum of regression and classification loss
        # losses = sum([loss for loss in loss_dict.values()])

        # return losses
        self.model.train() # Train mode
        images = [i['image'] for i in batch]  # List of tensors representing images
        targets = [{'boxes': torch.tensor(i['label_bboxes'])} for i in batch]  # List of dicts for ground-truth targets
        loss_dict = self.forward(images, targets)
        losses = sum([loss for loss in loss_dict.values()])
        return losses
    
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