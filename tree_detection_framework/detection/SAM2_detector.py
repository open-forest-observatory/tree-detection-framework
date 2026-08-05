import logging
from pathlib import Path

import torch
from shapely.geometry import box

from tree_detection_framework.constants import CHECKPOINTS_FOLDER, DEFAULT_DEVICE
from tree_detection_framework.detection.detector import Detector
from tree_detection_framework.utils.geometric import mask_to_shapely

try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    logging.warning(
        "SAM2 is not installed. SAMV2Detector will be disabled. See README for install instructions"
    )


# follow README for download instructions
class SAMV2Detector(Detector):

    def __init__(
        self,
        device=DEFAULT_DEVICE,
        sam2_checkpoint=Path(CHECKPOINTS_FOLDER, "sam2.1_hiera_large.pt"),
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        postprocessors=None,
        score_metric="predicted_iou",
        points_per_side=64,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=25.0,
        use_m2m=True,
    ):
        """
        Create a SAM2 detector.
        Args:
            device (torch.device): Device to run the model on.
            sam2_checkpoint (Path): Path to the SAM2 checkpoint.
            model_cfg (str): Path to the SAM2 model config.
            postprocessors (list, optional): See docstring for Detector class. Defaults to None.
            score_metric (str): Metric to use as prediction score. Either "stability_score" or
                "predicted_iou". Defaults to "predicted_iou".
            points_per_side (int): Number of points sampled along each side of the image.
            points_per_batch (int): Number of points processed simultaneously.
            pred_iou_thresh (float): Threshold for filtering masks by predicted IoU.
            stability_score_thresh (float): Threshold for filtering masks by stability score.
            stability_score_offset (float): Offset applied when computing stability score.
            crop_n_layers (int): Number of crop layers to use.
            box_nms_thresh (float): NMS threshold for bounding boxes.
            crop_n_points_downscale_factor (int): Point downscale factor per crop layer.
            min_mask_region_area (float): Minimum area (pixels) for mask regions.
            use_m2m (bool): Whether to use mask-to-mask refinement.
        """
        if not SAM2_AVAILABLE:
            raise ImportError(
                "SAMV2Detector requires SAM2. Please install it to use this detector."
            )
        if score_metric not in ("stability_score", "predicted_iou"):
            raise ValueError(
                f"score_metric must be 'stability_score' or 'predicted_iou', got '{score_metric}'"
            )
        super().__init__(postprocessors=postprocessors)

        self.device = device
        self.score_metric = score_metric

        self.sam2 = build_sam2(
            model_cfg, sam2_checkpoint, device=self.device, apply_postprocessing=False
        )
        # Create the mask generator with the optimal set of parameters found by
        # Michelle Chen & Jane Wu based on qualitative experiments
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            crop_n_layers=crop_n_layers,
            box_nms_thresh=box_nms_thresh,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
            use_m2m=use_m2m,
        )

    def call_predict(self, batch):
        """
        Args:
            batch (Tensor): 4 dims Tensor with the first dimension having number of images in the batch

        Returns:
            masks List[List[Dict]]: list of dictionaries for each mask in the batch
        """

        with torch.no_grad():
            masks = []
            for original_image in batch:
                if original_image.shape[0] < 3:
                    raise ValueError("Original image has less than 3 channels")

                original_image = original_image.permute(1, 2, 0)
                # If the pixels are in [0, 255] range, convert to [0, 1] range
                if original_image.max() > 1:
                    original_image = original_image.byte().numpy()
                else:
                    original_image = original_image.numpy()
                rgb_image = original_image[:, :, :3]
                mask = self.mask_generator.generate(
                    rgb_image
                )  # model expects rgb 0-255 range (h, w, 3)
                # FUTURE TODO: Support batched predictions
                masks.append(mask)

            return masks

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
        images = batch["image"]

        # computational bottleneck
        batch_preds = self.call_predict(images)

        # To store all predicted polygons
        all_geometries = []
        # To store other related information such as scores and labels
        all_data_dicts = []

        # Iterate through predictions for each tile in the batch
        for pred in batch_preds:

            # Get the Instances object
            segmentations = [dic["segmentation"].astype(float) for dic in pred]

            # Convert each mask to a shapely multipolygon
            shapely_objects = [
                mask_to_shapely(pred_mask) for pred_mask in segmentations
            ]

            all_geometries.append(shapely_objects)

            # Compute axis-aligned minimum area bounding box as Polygon objects
            bounding_boxes = [box(*polygon.bounds) for polygon in shapely_objects]

            # Get prediction scores
            scores = [dic[self.score_metric] for dic in pred]
            all_data_dicts.append({"score": scores, "bbox": bounding_boxes})

        return all_geometries, all_data_dicts
