import logging
import os
from pathlib import Path

from shapely.geometry import box

from tree_detection_framework.constants import CHECKPOINTS_FOLDER, DEFAULT_DEVICE
from tree_detection_framework.detection.detector import Detector
from tree_detection_framework.utils.geometric import mask_to_shapely

try:
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    logging.warning(
        "SAM3 is not installed. SAM3Detector will be disabled. See README for install instructions."
    )

# Default path to the BPE vocab file bundled with the SAM3 package
_SAM3_DEFAULT_BPE_PATH = (
    Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if SAM3_AVAILABLE
    else None
)


class SAM3Detector(Detector):

    def __init__(
        self,
        device=DEFAULT_DEVICE,
        bpe_path=_SAM3_DEFAULT_BPE_PATH,
        text_prompt="tree",
        confidence_threshold=0.4,
        checkpoint_path=None,
        huggingface_token=None,
        postprocessors=None,
    ):
        """
        Create a SAM3 detector. This detector uses a text prompt "tree" to detect tree crowns.
        Refer README for steps to generate a HuggingFace access token to download the weights.

        Args:
            device (torch.device): Device to run the model on.
            bpe_path (Path): Path to the BPE vocab file bundled with SAM3.
                Defaults to the assets folder inside the installed sam3 package.
            text_prompt (str): Text prompt used to guide SAM3 segmentation.
                Defaults to "tree".
            confidence_threshold (float): Minimum confidence score for keeping a
                predicted mask. Defaults to 0.4.
            checkpoint_path (str, optional): Path to a local SAM3 checkpoint
                file (.pt). When provided, weights are loaded from disk and
                HuggingFace download is skipped.
            huggingface_token (str, optional): HuggingFace API token for downloading
                SAM3 model weights. Only used when checkpoint_path is not given.
                Can also be set via the HF_TOKEN environment variable.
            postprocessors (list, optional): See docstring for Detector class.
                Defaults to None.
        """
        if not SAM3_AVAILABLE:
            raise ImportError(
                "SAM3Detector requires SAM3. Please install it to use this detector."
            )
        super().__init__(postprocessors=postprocessors)

        self.device = device
        self.text_prompt = text_prompt
        self.confidence_threshold = confidence_threshold

        if checkpoint_path is not None:
            # Load weights from a local file; skip HuggingFace entirely.
            self.model = build_sam3_image_model(
                bpe_path=str(bpe_path),
                checkpoint_path=str(checkpoint_path),
                load_from_HF=False,
            )
        else:
            # Fall back to HuggingFace download path.
            token = huggingface_token or os.environ.get("HF_TOKEN")
            if token:
                from huggingface_hub import login

                login(token=token)
            else:
                logging.info(
                    "No HuggingFace token provided. Assuming SAM3 weights are already cached. "
                    "Pass huggingface_token= or set the HF_TOKEN environment variable if download is needed."
                )
            self.model = build_sam3_image_model(bpe_path=str(bpe_path))
        self.processor = Sam3Processor(
            self.model, confidence_threshold=self.confidence_threshold
        )

    def call_predict(self, batch):
        """
        Run SAM3 inference on a batch of image tensors.

        Args:
            batch (Tensor): 4D tensor (N, C, H, W)

        Returns:
            List[dict]: One dict per image in the batch, each containing:
                - 'masks' (List[Tensor]): Binary masks of shape (1, H, W)
                - 'scores' (List[Tensor]): Confidence scores for each mask
                - 'boxes' (List[Tensor]): Bounding boxes for each mask
                - 'masks_logits' (List[Tensor]): Raw unnormalized logits of shape (1, H, W)
                - 'original_height' (int): Height of the input image
                - 'original_width' (int): Width of the input image
        """
        results = []
        for original_image in batch:
            if original_image.shape[0] < 3:
                raise ValueError("Image has fewer than 3 channels.")

            # Convert tensor (C, H, W) to uint8, slicing to 3 channels for RGB
            image = original_image[:3]  # drop alpha if present
            if image.max() <= 1.0:
                image = (image * 255).byte()
            else:
                image = image.byte()

            # Run SAM3 with text prompt
            inference_state = self.processor.set_image(image)
            self.processor.reset_all_prompts(inference_state)
            inference_state = self.processor.set_text_prompt(
                state=inference_state, prompt=self.text_prompt
            )
            # Append results for this image to the batch results
            results.append(inference_state)

        return results

    def predict_batch(self, batch):
        images = batch["image"]
        batch_preds = self.call_predict(images)

        all_geometries = []
        all_data_dicts = []

        for inference_state in batch_preds:
            masks = inference_state["masks"]  # list of (H, W) arrays
            scores = inference_state["scores"]  # list of floats

            shapely_objects = []
            kept_scores = []
            bounding_boxes = []

            for mask, score in zip(masks, scores):
                # Convert masks to shapely polygons
                shapely_mask = mask_to_shapely(
                    mask.squeeze().cpu().numpy().astype(float)
                )

                if shapely_mask is None or shapely_mask.is_empty:
                    continue

                shapely_objects.append(shapely_mask)
                kept_scores.append(score.item() if hasattr(score, "item") else score)
                bounding_boxes.append(box(*shapely_mask.bounds))

            all_geometries.append(shapely_objects)
            all_data_dicts.append({"score": kept_scores, "bbox": bounding_boxes})

        return all_geometries, all_data_dicts
