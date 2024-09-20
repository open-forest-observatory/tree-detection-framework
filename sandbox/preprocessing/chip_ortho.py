from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler, Units
from torch.utils.data import DataLoader
import torch
import random
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import argparse

class CustomOrthoDataset(RasterDataset):
    filename_glob = '*.tif'  # To match all TIFF files
    is_image = True
    separate_files = False

def chip_orthomosaics(root, size, stride, units="pixel", res=None, save_dir=None, visualize_n=None):
    # create dataset instance
    dataset = CustomOrthoDataset(paths=root, res=res)
    units = Units.CRS if units == "meters" else Units.PIXELS

    if save_dir:
        sampler = GridGeoSampler(dataset, size=size, stride=stride, units=units) #GridGeoSampler to get contiguous tiles
        dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        i = 0
        for batch in dataloader:
            sample = unbind_samples(batch)[0]
            image = sample['image']
            image_tensor = torch.clamp(image / 255.0, min=0, max=1)
            pil_image = ToPILImage()(image_tensor)
            pil_image.save(save_dir+"/tile_"+str(i)+".png")
            i += 1
        print("Saved "+str(i)+" tiles")

    if visualize_n:
        sampler = RandomGeoSampler(dataset, size=size, length=visualize_n) # Randomly selects 'visualize_n' number of tiles
        dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)
        for batch in dataloader:
            sample = unbind_samples(batch)[0]
            plot(sample)
            plt.axis('off')
            plt.show()

# could be moved to a separate utils file
def plot(sample):
    image = sample['image'].permute(1, 2, 0)
    image = torch.clamp(image / 255.0, min=0, max=1).numpy()
    fig, ax = plt.subplots()
    ax.imshow(image)
    return fig

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Chipping orthomosaic images")
    parser.add_argument('--path', type=str, required=True, help="Path to folder or individual orthomosaic")
    parser.add_argument('--res', type=float, required=False, help="Resolution of the dataset in units of CRS (defaults to the resolution of the first file found)")
    parser.add_argument('--size', type=int, required=True, help="Single value used for height and width dim")
    parser.add_argument('--stride', type=int, required=True, help="Distance to skip between each patch")
    parser.add_argument('--units', type=str, required=False, choices=["pixels", "meters"], default="pixels", help="Units for tile size and stride")
    parser.add_argument('--save_dir', type=str, required=False, help="Directory to save chips")
    parser.add_argument('--visualize_n', type=int, required=False, help="Number of tiles to visualize")

    args = parser.parse_args()

    chip_orthomosaics(
        root=args.path,
        size=args.size,
        stride=args.stride,
        units=args.units,
        res=args.res,
        save_dir=args.save_dir,
        visualize_n=args.visualize_n
    )





