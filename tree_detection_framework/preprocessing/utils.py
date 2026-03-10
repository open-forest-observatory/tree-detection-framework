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
