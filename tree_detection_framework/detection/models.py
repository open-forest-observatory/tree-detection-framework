import logging
import os
import time
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning
import torch
import torchvision
import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
)
from detectron2.evaluation import print_csv_format
from detectron2.evaluation.testing import flatten_results_dict

from tools.train_net import build_evaluator

from torch import Tensor, optim
from torchvision.models.detection.retinanet import (
    AnchorGenerator,
    RetinaNet,
    RetinaNet_ResNet50_FPN_Weights,
)

from tree_detection_framework.utils.detection import use_release_df
from tree_detection_framework.preprocessing.derived_geodatasets import CustomDataModule

from torchvision.models.detection import maskrcnn_resnet50_fpn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2")

class RetinaNetModel:
    """A backbone class for DeepForest"""

    def __init__(self, param_dict):
        self.param_dict = param_dict

    def load_backbone(self):
        """A torch vision retinanet model"""
        backbone = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1
        )

        return backbone

    def create_anchor_generator(
        self, sizes=((8, 16, 32, 64, 128, 256, 400),), aspect_ratios=((0.5, 1.0, 2.0),)
    ):
        """Create anchor box generator as a function of sizes and aspect ratios"""
        anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)

        return anchor_generator

    def create_model(self):
        """Create a retinanet model

        Returns:
            model: a pytorch nn module
        """
        resnet = self.load_backbone()
        backbone = resnet.backbone

        model = RetinaNet(backbone=backbone, num_classes=self.param_dict["num_classes"])
        # TODO: do we want to set model.nms_thresh and model.score_thresh?

        return model


class DeepForestModule(lightning.LightningModule):
    def __init__(self, param_dict: Dict[str, Any]):
        super().__init__()
        self.param_dict = param_dict

        if param_dict["backbone"] == "retinanet":
            retinanet = RetinaNetModel(param_dict)
        else:
            raise ValueError("Only 'retinanet' backbone is currently supported.")

        self.model = retinanet.create_model()
        self.use_release()

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
            check_release=check_release
        )
        self.model.load_state_dict(torch.load(self.release_state_dict))

        # load saved model and tag release
        self.__release_version__ = release_tag
        print("Loading pre-built model: {}".format(release_tag))

    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Dict[str, Tensor]:
        """Calls the model's forward method.
        Args:
            images (list[Tensor]): Images to be processed
            targets (list[Dict[Tensor]]): Ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]):
                The output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
        """
        # Move the data to the same device as the model
        images = images.to(self.device)
        return self.model.forward(images, targets=targets)  # Model specific forward

    def training_step(self, batch):
        # Ensure model is in train mode
        self.model.train()
        device = next(self.model.parameters()).device

        # Image is expected to be a list of tensors, each of shape [C, H, W] in 0-1 range.
        image_batch = (batch["image"][:, :3, :, :] / 255.0).to(device)
        image_batch_list = [image for image in image_batch]

        # To store every image's target - a dictionary containing `boxes` and `labels`
        targets = []
        for tile in batch["bounding_boxes"]:
            # Convert from list to FloatTensor[N, 4]
            boxes_tensor = torch.tensor(tile, dtype=torch.float32).to(device)
            # Need to remove boxes that go out-of-bounds. Has negative values.
            valid_mask = (boxes_tensor >= 0).all(dim=1)
            filtered_boxes_tensor = boxes_tensor[valid_mask]
            # Create a label tensor. Single class for now.
            class_labels = torch.zeros(
                filtered_boxes_tensor.shape[0], dtype=torch.int64
            ).to(device)
            # Dictionary for the tile
            d = {"boxes": filtered_boxes_tensor, "labels": class_labels}
            targets.append(d)

        loss_dict = self.forward(image_batch_list, targets=targets)

        final_loss = sum([loss for loss in loss_dict.values()])
        print("loss: ", final_loss)
        return final_loss

    def configure_optimizers(self):
        # similar to the one in deepforest
        optimizer = optim.SGD(
            self.model.parameters(), lr=self.param_dict["train"]["lr"], momentum=0.9
        )

        # TODO: Setup lr_scheduler
        # TODO: Return 'optimizer', 'lr_scheduler', 'monitor' when validation data is set

        return optimizer
    
class Detectree2Module(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.weights = torch.load("/ofo-share/repos-amritha/detectree2-code/230103_randresize_full.pth")
        self.model = maskrcnn_resnet50_fpn(weights=self.weights)

    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Dict[str, Tensor]:
        """Calls the model's forward method.
        Args:
            images (list[Tensor]): Images to be processed
            targets (list[Dict[Tensor]]): Ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]):
                The output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
        """
        # Move the data to the same device as the model
        images = images.to(self.device)
        return self.model.forward(images, targets=targets)  # Model specific forward

    def training_step(self, batch):
        # Ensure model is in train mode
        self.model.train()
        device = next(self.model.parameters()).device

        # Image is expected to be a list of tensors, each of shape [C, H, W] in 0-1 range.
        image_batch = (batch["image"][:, :3, :, :] / 255.0).to(device)
        image_batch_list = [image for image in image_batch]

        # To store every image's target - a dictionary containing `boxes` and `labels`
        targets = []
        for tile in batch["bounding_boxes"]:
            # Convert from list to FloatTensor[N, 4]
            boxes_tensor = torch.tensor(tile, dtype=torch.float32).to(device)
            # Need to remove boxes that go out-of-bounds. Has negative values.
            valid_mask = (boxes_tensor >= 0).all(dim=1)
            filtered_boxes_tensor = boxes_tensor[valid_mask]
            # Create a label tensor. Single class for now.
            class_labels = torch.zeros(
                filtered_boxes_tensor.shape[0], dtype=torch.int64
            ).to(device)
            # Dictionary for the tile
            d = {"boxes": filtered_boxes_tensor, "labels": class_labels}
            targets.append(d)

        loss_dict = self.forward(image_batch_list, targets=targets)

        final_loss = sum([loss for loss in loss_dict.values()])
        print("loss: ", final_loss)
        return final_loss

    def configure_optimizers(self):
        # similar to the one in deepforest
        optimizer = optim.SGD(
            self.model.parameters(), lr=self.param_dict["train"]["lr"], momentum=0.9
        )

        # TODO: Setup lr_scheduler
        # TODO: Return 'optimizer', 'lr_scheduler', 'monitor' when validation data is set

        return optimizer


############# TRIAL 2 ####################################
# class Detectree2Module(lightning.LightningModule):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.model = build_model(self.cfg)

#     def training_step(self, batch):
#          image_batch = batch["image"][:, :3, :, :]
#          # image_batch_list = [image for image in image_batch]
#          loss_dict = self.model([image_batch])
#          print("LOSS: ", loss_dict)
#          return sum(loss_dict.values())
    
#     def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
#         if not isinstance(batch, List):
#             batch = [batch]
#         outputs = self.model(batch)
#         self._evaluators[dataloader_idx].process(batch, outputs)

#     def configure_optimizers(self):
#         optimizer = build_optimizer(self.cfg, self.model)
#         self._best_param_group_id = hooks.LRScheduler.get_best_param_group_id(optimizer)
#         scheduler = build_lr_scheduler(self.cfg, optimizer)
#         return [optimizer], [{"scheduler": scheduler, "interval": "step"}]





############ TRIAL 1 ####################################
# class Detectree2Module(lightning.LightningModule):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg

#         if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
#             setup_logger()
#         self.cfg = DefaultTrainer.auto_scale_workers(self.cfg, comm.get_world_size())
#         self.storage: EventStorage = None
#         self.model = build_model(self.cfg)

#         self.start_iter = 0
#         self.max_iter = self.cfg.SOLVER.MAX_ITER


#     def training_step(self, batch, batch_idx):
#         if self.storage is None:
#             self.storage = EventStorage(0)
#             self.storage.__enter__()
#             # self.iteration_timer.trainer = weakref.proxy(self)
#             # self.iteration_timer.before_step()
#             self.writers = (
#                 default_writers(self.cfg.OUTPUT_DIR, self.max_iter)
#                 if comm.is_main_process()
#                 else {}
#             )

#         loss_dict = self.model(batch)
#         # SimpleTrainer.write_metrics(loss_dict, data_time)
#         print("LOSS: ", loss_dict)

#         opt = self.optimizers()
#         self.storage.put_scalar(
#             "lr",
#             opt.param_groups[self._best_param_group_id]["lr"],
#             smoothing_hint=False,
#         )
#         # self.iteration_timer.after_step()
#         self.storage.step()
#         # A little odd to put before step here, but it's the best way to get a proper timing
#         # self.iteration_timer.before_step()

#         if self.storage.iter % 20 == 0:
#             for writer in self.writers:
#                 writer.write()
#         return sum(loss_dict.values())

#     def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
#         if not isinstance(batch, List):
#             batch = [batch]
#         outputs = self.model(batch)
#         self._evaluators[dataloader_idx].process(batch, outputs)

#     def configure_optimizers(self):
#         optimizer = build_optimizer(self.cfg, self.model)
#         self._best_param_group_id = hooks.LRScheduler.get_best_param_group_id(optimizer)
#         scheduler = build_lr_scheduler(self.cfg, optimizer)
#         return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    
# def train(cfg, data_module):
#     trainer_params = {
#         # training loop is bounded by max steps, use a large max_epochs to make
#         # sure max_steps is met first
#         "max_epochs": 10**8,
#         "max_steps": cfg.SOLVER.MAX_ITER,
#         "val_check_interval": 1, # cfg.TEST.EVAL_PERIOD if cfg.TEST.EVAL_PERIOD > 0 else 10**8,
#         "num_nodes": 1,
#         # "gpus": 1,
#         "num_sanity_val_steps": 0,
#     }
#     if cfg.SOLVER.AMP.ENABLED:
#         trainer_params["precision"] = 16


#     trainer = lightning.Trainer(**trainer_params)

#     module = Detectree2Module(cfg)
#     logger.info("Running training")
#     trainer.fit(module, data_module)

#     # if args.eval_only:
#     #     logger.info("Running inference")
#     #     trainer.validate(module, data_module)
#     # else:
#     #     logger.info("Running training")
#     #     trainer.fit(module, data_module)
