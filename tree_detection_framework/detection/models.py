from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning
import torch
import torchvision
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from torch import Tensor, optim
from torchvision.models.detection.retinanet import (
    AnchorGenerator,
    RetinaNet,
    RetinaNet_ResNet50_FPN_Weights,
)

from tree_detection_framework.utils.detection import use_release_df


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


class Detectree2Module:
    def __init__(self, param_dict: Optional[Dict[str, Any]] = None):
        super().__init__()
        # If param_dict is not provided, ensure it is an empty dictionary
        self.param_dict = param_dict or {}
        self.cfg = self.setup_cfg(**self.param_dict)

    def setup_cfg(
        self,
        base_model: str = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        trains=("trees_train",),
        tests=("trees_val",),
        update_model=None,
        workers=2,
        ims_per_batch=2,
        gamma=0.1,
        backbone_freeze=3,
        warm_iter=120,
        momentum=0.9,
        batch_size_per_im=1024,
        base_lr=0.0003389,
        weight_decay=0.001,
        max_iter=1000,
        num_classes=1,
        eval_period=100,
        out_dir="./train_outputs",
        resize=True,
    ):
        """Set up config object.
        Args:
            base_model: base pre-trained model from detectron2 model_zoo
            trains: names of registered data to use for training
            tests: names of registered data to use for evaluating models
            update_model: updated pre-trained model from detectree2 model_garden
            workers: number of workers for dataloader
            ims_per_batch: number of images per batch
            gamma: gamma for learning rate scheduler
            backbone_freeze: backbone layer to freeze
            warm_iter: number of iterations for warmup
            momentum: momentum for optimizer
            batch_size_per_im: batch size per image
            base_lr: base learning rate
            weight_decay: weight decay for optimizer
            max_iter: maximum number of iterations
            num_classes: number of classes
            eval_period: number of iterations between evaluations
            out_dir: directory to save outputs
            resize: whether to resize input images
        """
        # Initialize configuration
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(base_model))

        # Assign values, prioritizing those in param_dict
        cfg.DATASETS.TRAIN = self.param_dict.get("trains", trains)
        cfg.DATASETS.TEST = self.param_dict.get("tests", tests)
        cfg.DATALOADER.NUM_WORKERS = self.param_dict.get("workers", workers)
        cfg.SOLVER.IMS_PER_BATCH = self.param_dict.get("ims_per_batch", ims_per_batch)
        cfg.SOLVER.GAMMA = self.param_dict.get("gamma", gamma)
        cfg.MODEL.BACKBONE.FREEZE_AT = self.param_dict.get(
            "backbone_freeze", backbone_freeze
        )
        cfg.SOLVER.WARMUP_ITERS = self.param_dict.get("warm_iter", warm_iter)
        cfg.SOLVER.MOMENTUM = self.param_dict.get("momentum", momentum)
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = self.param_dict.get(
            "batch_size_per_im", batch_size_per_im
        )
        cfg.SOLVER.WEIGHT_DECAY = self.param_dict.get("weight_decay", weight_decay)
        cfg.SOLVER.BASE_LR = self.param_dict.get("base_lr", base_lr)
        cfg.OUTPUT_DIR = self.param_dict.get("out_dir", out_dir)
        cfg.SOLVER.MAX_ITER = self.param_dict.get("max_iter", max_iter)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.param_dict.get(
            "num_classes", num_classes
        )
        cfg.TEST.EVAL_PERIOD = self.param_dict.get("eval_period", eval_period)
        cfg.RESIZE = self.param_dict.get("resize", resize)
        cfg.INPUT.MIN_SIZE_TRAIN = 1000

        # Create output directory if it doesn't exist
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        # Set model weights
        cfg.MODEL.WEIGHTS = self.param_dict.get(
            "update_model", update_model
        ) or model_zoo.get_checkpoint_url(base_model)

        return cfg


# future TODO: add module configs for sam2, currently implemented for default configs
