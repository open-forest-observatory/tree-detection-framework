from abc import abstractmethod

from torch.utils.data import DataLoader

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.detection.region_detections import RegionDetectionsSet


class Detector:
    def setup(self):
        raise NotImplementedError()

    def predict(
        self, inference_dataloader: DataLoader, **kwargs
    ) -> RegionDetectionsSet:
        raise NotImplementedError()


class LightningDetector(Detector):
    @abstractmethod
    def setup_model(self):
        raise NotImplementedError()

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, **kwargs):
        raise NotImplementedError()

    def save_model(self, save_file: PATH_TYPE):
        raise NotImplementedError()
