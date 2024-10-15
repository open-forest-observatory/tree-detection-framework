from abc import abstractmethod
from typing import Optional

import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchgeo.datasets import unbind_samples
from deepforest import main

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

    def __init__(self):
        self.model = self.setup_model({})

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

class DeepForestDetector(LightningDetector):

    def __init__(self):
        self.model = self.setup_model({})

    def setup_model(self, param_dict: dict) -> lightning.LightningModule:
        model = main.deepforest()
        model.use_release()
        return model

    def setup_trainer(self, param_dict: dict) -> lightning.Trainer:
        raise NotImplementedError()

    def predict(self, inference_dataloader: DataLoader, **kwargs) -> list:
        predictions = []
        for i, tile in enumerate(inference_dataloader):
            sample = unbind_samples(tile)[0]
            image = sample["image"].permute(1, 2, 0).byte().numpy()
            output = self.model.predict_image(image[:, :, :3])
            predictions.append(output)
        return predictions

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, **kwargs):
        raise NotImplementedError()

    def save_model(self, save_file: PATH_TYPE):
        self.model.save_model(save_file)