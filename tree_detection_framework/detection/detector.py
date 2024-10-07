from abc import abstractmethod
from typing import Optional

import lightning
from torch.utils.data import DataLoader

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.detection.region_detections import RegionDetectionsSet


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


class LightningDetector(Detector):
    model: lightning.LightningModule
    trainer: lightning.Trainer

    def setup(self):
        # This method should implement setup tasks that are common to all LightningDetectors.
        # Method-specific tasks should be defered to setup_model
        raise NotImplementedError()

    @abstractmethod
    def setup_model(self):
        """Set up the lightning model, including loading pretrained weights if required"""
        # Should be implemented in each derived class since it's algorithm-specific
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
