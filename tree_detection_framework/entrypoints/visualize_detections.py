import argparse
from pathlib import Path
import warnings

import geopandas as gpd
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

from tree_detection_framework.detection.region_detections import RegionDetections

# We know we don't care about these warnings from rasterio
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def visualize_detections(
    image_dir, detection_dir, out_dir, n_images, step, show_centroid,
):

    # Get sorted lists of image and gpkg files
    gpkg_paths = sorted(detection_dir.glob("*.gpkg"))

    # Sample indices according to n_images and step
    indices = [i for i in range(0, len(gpkg_paths), step)][:n_images]

    for idx in tqdm(indices, f"Saving images to {out_dir}"):
        gpkg_path = gpkg_paths[idx]
        image_path = image_dir / (gpkg_path.stem + ".JPG")

        detections = RegionDetections(
            detection_geometries="geometry",
            data=gpd.read_file(gpkg_path),
        )

        axis = detections.plot(
            plt_show=False,
            raster_file=image_path,
            plt_points=show_centroid,
            raster_vis_downsample=1,
        )

        figure = axis.get_figure()
        figure.savefig(
            out_dir / (gpkg_path.stem + ".png"), dpi=300, bbox_inches="tight"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize detection results on raw images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing raw images.",
    )
    parser.add_argument(
        "detection_dir",
        type=Path,
        help="Directory containing detection .gpkg files.",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Directory to save visualizations. Will be created if it does not exist.",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=5,
        help="Number of images to visualize.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size between visualized images.",
    )
    parser.add_argument(
        "--no-centroid",
        action="store_true",
        help="Do not show centroid in visualization.",
    )

    args = parser.parse_args()

    # Check that image_dir and detection_dir exist and are not empty
    for dir_path in [args.image_dir, args.detection_dir]:
        assert dir_path.is_dir(), f"{dir_path} not found"
        assert any(dir_path.iterdir()), f"{dir_path} empty"

    # Check that each detection file matches an image file (by stem, with .JPG extension)
    image_stems = {p.stem for p in args.image_dir.glob("*.JPG")}
    for gpkg_file in args.detection_dir.glob("*.gpkg"):
        assert (
            gpkg_file.stem in image_stems
        ), f"No matching image for detection {gpkg_file}"

    # Create out_dir if it doesn't exist
    args.out_dir.mkdir(parents=True, exist_ok=True)

    return args


def main():
    args = parse_args()
    visualize_detections(
        image_dir=args.image_dir,
        detection_dir=args.detection_dir,
        out_dir=args.out_dir,
        n_images=args.n_images,
        step=args.step,
        show_centroid=not args.no_centroid,
    )


if __name__ == "__main__":
    main()
