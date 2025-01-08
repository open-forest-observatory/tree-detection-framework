import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

from tree_detection_framework.constants import DEFAULT_DEVICE
from tree_detection_framework.detection.detector import Detector
from tree_detection_framework.utils.geometric import mask_to_shapely


# follow README for download instructions
class SAMV2Detector(Detector):

    def __init__(
        self,
        device=DEFAULT_DEVICE,
        sam2_checkpoint="checkpoints/sam2.1_hiera_large.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    ):
        self.device = device

        self.sam2 = build_sam2(
            model_cfg, sam2_checkpoint, device=self.device, apply_postprocessing=False
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2)

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

                original_image = original_image.permute(1, 2, 0).byte().numpy()
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

            # Get prediction scores
            scores = [dic["stability_score"] for dic in pred]
            all_data_dicts.append({"score": scores})

        return all_geometries, all_data_dicts
