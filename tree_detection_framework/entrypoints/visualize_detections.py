import argparse
import warnings
from pathlib import Path

import geopandas as gpd
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

from tree_detection_framework.detection.region_detections import RegionDetections

# We know we don't care about these warnings from rasterio
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def visualize_detection_sets(
    image_dir: Path,
    detection_dir: Path,
    out_dir: Path,
    n_images: int,
    step: int,
    show_centroid: bool,
    image_ext: str,
) -> None:
    """
    Visualize detection results by pairing images and detection files, sampling
    according to n_images and step.

    Args:
        image_dir: Directory containing raw images.
        detection_dir: Directory containing detection .gpkg files.
        out_dir: Directory to save visualizations.
        n_images: Number of images to visualize.
        step: Step size between visualized images.
        show_centroid: Whether to show centroid in visualization.
        image_ext: Extension of image files (e.g., 'JPG', 'tif').
    Returns: None. Saves files in the output directory.
    """
    # Get sorted lists of image and gpkg files
    gpkg_paths = sorted(detection_dir.glob("*.gpkg"))

    # Sample indices according to n_images and step
    indices = list(range(0, min(len(gpkg_paths), step * n_images), step))

    for idx in tqdm(indices, f"Saving images to {out_dir}"):
        gpkg_path = gpkg_paths[idx]
        image_path = image_dir / f"{gpkg_path.stem}.{image_ext}"

        detections = RegionDetections(
            detection_geometries="geometry",
            data=gpd.read_file(gpkg_path),
        )

        axis = detections.plot(
            plt_show=False,
            raster_file=image_path,
            show_centroid=show_centroid,
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
    parser.add_argument(
        "--image-ext",
        type=str,
        default="JPG",
        help="Extension of image files (e.g., 'JPG', 'tif').",
    )

    args = parser.parse_args()

    # Check that image_dir and detection_dir exist and are not empty
    for dir_path in [args.image_dir, args.detection_dir]:
        assert dir_path.is_dir(), f"{dir_path} not found"
        assert any(dir_path.iterdir()), f"{dir_path} empty"

    # Check that each detection file matches an image file (by stem, with user-specified extension)
    image_stems = {p.stem for p in args.image_dir.glob(f"*.{args.image_ext}")}
    for gpkg_file in args.detection_dir.glob("*.gpkg"):
        assert (
            gpkg_file.stem in image_stems
        ), f"No matching image for detection file {gpkg_file.name}"

    # Create out_dir if it doesn't exist
    args.out_dir.mkdir(parents=True, exist_ok=True)

    return args


def main():
    args = parse_args()
    visualize_detection_sets(
        image_dir=args.image_dir,
        detection_dir=args.detection_dir,
        out_dir=args.out_dir,
        n_images=args.n_images,
        step=args.step,
        show_centroid=not args.no_centroid,
        image_ext=args.image_ext,
    )


if __name__ == "__main__":
    main()
