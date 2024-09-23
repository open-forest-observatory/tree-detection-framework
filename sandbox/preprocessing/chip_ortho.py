import argparse
import random
import json

import matplotlib.pyplot as plt
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples
from torchgeo.samplers import GridGeoSampler, Units
from torchvision.transforms import ToPILImage

class CustomOrthoDataset(RasterDataset):
    filename_glob = '*.tif'  # To match all TIFF files
    is_image = True
    separate_files = False

def chip_orthomosaics(path, size, stride=None, overlap=None, units="pixel", res=None, use_units_meters=False, save_dir=None, visualize_n=None):
    # Create dataset instance
    dataset = CustomOrthoDataset(paths=path, res=res)
    units = Units.CRS if use_units_meters == True else Units.PIXELS
    print("Units = ", units)

    # Calculate stride if overlap is provided
    if overlap is not None:
        stride = size * (1 - overlap / 100.0)
        print("Calculated stride based on overlap: "+str(stride))
    elif stride is None:
        raise ValueError("Either 'stride' or 'overlap' must be provided.")
    print("Stride = ", stride)
    
    #GridGeoSampler to get contiguous tiles
    sampler = GridGeoSampler(dataset, size=size, stride=stride, units=units)
    dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

    total_tiles = len(sampler)
    # Randomly pick indices for visualization if visualize_n is specified
    visualize_indices = random.sample(range(total_tiles), visualize_n) if visualize_n else []

    # Creates save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(dataloader):
        sample = unbind_samples(batch)[0]

        # Saving logic
        if save_dir:
            image = sample['image']
            image_tensor = torch.clamp(image / 255.0, min=0, max=1)
            pil_image = ToPILImage()(image_tensor)
            pil_image.save(Path(save_dir) / f'tile_{i}.png')

            # Save tile metadata to a json file
            metadata = {
                "crs": sample['crs'].to_string(),  
                "bounds": list(sample['bounds']),
            }
            with open(Path(save_dir) / f'tile_{i}.json', 'w') as f:
                json.dump(metadata, f, indent=4)

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
    image = image.byte().numpy()
    fig, ax = plt.subplots()
    ax.imshow(image)
    return fig

def parse_args():
    parser = argparse.ArgumentParser(description="Chipping orthomosaic images")
    parser.add_argument('--path', type=str, required=True, help="Path to folder or individual orthomosaic")
    parser.add_argument('--res', type=float, required=False, help="Resolution of the dataset in units of CRS (defaults to the resolution of the first file found)")
    parser.add_argument('--size', type=float, required=True, help="Single value used for height and width dim")
    parser.add_argument('--stride', type=float, required=False, help="Distance to skip between each patch")
    parser.add_argument('--overlap', type=float, required=False, help="Percentage overlap between the tiles (0-100%)")
    parser.add_argument('--use-units-meters', action='store_true', help="Whether to set units for tile size and stide as meters")
    parser.add_argument('--save-dir', type=str, required=False, help="Directory to save chips")
    parser.add_argument('--visualize-n', type=int, required=False, help="Number of tiles to visualize")
    # to add: arg to accept different regex patterns

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    chip_orthomosaics(**args.__dict__)

