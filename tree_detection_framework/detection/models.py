from typing import Any, Dict, List, Optional

import lightning
import torch
import torchvision
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
