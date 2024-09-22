import argparse
import os
import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler, Units
from torchvision.transforms import ToPILImage

class CustomOrthoDataset(RasterDataset):
    filename_glob = '*.tif'  # To match all TIFF files
    is_image = True
    separate_files = False

def chip_orthomosaics(root, size, stride, units="pixel", res=None, save_dir=None, visualize_n=None):
    # Create dataset instance
    dataset = CustomOrthoDataset(paths=root, res=res)
    units = Units.CRS if units == "meters" else Units.PIXELS

    #GridGeoSampler to get contiguous tiles
    sampler = GridGeoSampler(dataset, size=size, stride=stride, units=units)
    dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

    total_tiles = len(sampler)
    # Randomly pick indices for visualization if visualize_n is specified
    visualize_indices = random.sample(range(total_tiles), visualize_n) if visualize_n else []

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, batch in enumerate(dataloader):
        sample = unbind_samples(batch)[0]
        image = sample['image']
        image_tensor = torch.clamp(image / 255.0, min=0, max=1)

        # Saving logic
        if save_dir:
            pil_image = ToPILImage()(image_tensor)
            pil_image.save(save_dir + '/tile_' + str(i) + '.png')

        # Visualization logic
        if visualize_n and i in visualize_indices:
            plot(sample)
            plt.axis('off')
            plt.show()

    # Action summary
    if save_dir:
        print("Saved " + str(i + 1) + " tiles to " + save_dir)
    if visualize_n:
        print("Visualized " + str(len(visualize_indices)) + " tiles")


# Could be moved to a separate utils file
def plot(sample):
    image = sample['image'].permute(1, 2, 0)
    image = torch.clamp(image / 255.0, min=0, max=1).numpy()
    fig, ax = plt.subplots()
    ax.imshow(image)
    return fig

def parse_args():
    parser = argparse.ArgumentParser(description="Chipping orthomosaic images")
    parser.add_argument('--path', type=str, required=True, help="Path to folder or individual orthomosaic")
    parser.add_argument('--res', type=float, required=False, help="Resolution of the dataset in units of CRS (defaults to the resolution of the first file found)")
    parser.add_argument('--size', type=int, required=True, help="Single value used for height and width dim")
    parser.add_argument('--stride', type=int, required=True, help="Distance to skip between each patch")
    parser.add_argument('--units', type=str, required=False, choices=["pixels", "meters"], default="pixels", help="Units for tile size and stride")
    parser.add_argument('--save_dir', type=str, required=False, help="Directory to save chips")
    parser.add_argument('--visualize_n', type=int, required=False, help="Number of tiles to visualize")
    # to add: arg to accept different regex patterns

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    chip_orthomosaics(
        root=args.path,
        size=args.size,
        stride=args.stride,
        units=args.units,
        res=args.res,
        save_dir=args.save_dir,
        visualize_n=args.visualize_n
    )





