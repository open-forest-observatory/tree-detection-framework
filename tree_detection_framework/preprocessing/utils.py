import kornia.augmentation as K
from math import ceil


class KorniaTransformWrapper:
    """
    Ensure that the dimension of a single sample is not inflated, as is the case with
    using Kornia directly. E.g. as sample intended to be (1, 256, 256) -> (1, 1, 256, 256).
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        img = sample["image"]
        img = self.transforms(img.unsqueeze(0)).squeeze(0)
        sample["image"] = img
        return sample


def create_guassian_blur_transform(kernel_sigma_pixels: float):
    """Create a gaussian blurring transform for preprocessing samples

    Args:
        kernel_sigma_pixels (float): The standard diviation of the gaussian kernel

    Returns:
        function: A function which can transfrom a data dictionary for a batch of samples
    """
    # Set the kernel size at over two sigmas, which captures the vast majority of the probability density
    kernel_size = 2 * ceil(kernel_sigma_pixels) + 1
    # Create a gaussian blur operation. The kernel sigma is normally random, but we set the upper and lower
    # values to be identical. The probability is 1.0 so it's always applied. The wrapper ensures that
    # the shape of singleton batches is maintained.
    transform = KorniaTransformWrapper(
        K.AugmentationSequential(
            K.RandomGaussianBlur(
                kernel_size=kernel_size,
                sigma=(kernel_sigma_pixels, kernel_sigma_pixels),
                border_type="replicate",
                p=1.0,
            ),
        )
    )
    return transform


def create_box_blur_transform(kernel_size_pixels):
    """Create a box blurring transform for preprocessing samples

    Args:
        kernel_size_pixels (int): The number of pixels in the filter. Should be odd.

    Raises:
        ValueError: if the kernel size is not odd

    Returns:
        function: A function which can transfrom a data dictionary for a batch of samples
    """
    if kernel_size_pixels % 2 == 0:
        raise ValueError("kernel size is even")

    # Create a box blurring operation
    # The probability is 1.0 so it's always applied. The wrapper ensures that
    # the shape of singleton batches is maintained.
    transform = KorniaTransformWrapper(
        K.AugmentationSequential(
            K.RandomBoxBlur(
                kernel_size=(kernel_size_pixels, kernel_size_pixels),
                border_type="replicate",
                p=1.0,
            ),
        )
    )
    return transform
