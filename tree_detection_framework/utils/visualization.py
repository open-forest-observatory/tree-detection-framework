from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw

from tree_detection_framework.constants import PATH_TYPE
from tree_detection_framework.detection.region_detections import RegionDetectionsSet


def show_filtered_detections(
    impath: PATH_TYPE,
    detection1: Union[RegionDetectionsSet, PATH_TYPE],
    detection2: Union[RegionDetectionsSet, PATH_TYPE],
    mask: np.ndarray,
    mask_colormap: Optional[dict] = None,
) -> Tuple[np.ndarray]:
    """
    Visualizes a full set of detections (detection1) against a filtered set of
    detections (detection2) and the mask that acted as the filter.

    Arguments:
        impath (PATH_TYPE): Path to an (M, N, 3) RGB image.
        detection1 (Union[RegionDetectionsSet, PATH_TYPE]): RegionDetectionsSet
            derived from a specific drone image, or a geospatial file containing
            the detections from a drone image.
        detection2 (Union[RegionDetectionsSet, PATH_TYPE]): Same as detection1,
            but this is assumed to be a filtered version (subset) of detection1.
        mask (np.ndarray): (M, N) array of masked areas. Could be [True, False]
            or it could have difference mask values per area, such as [0, 1, 2]
            where each value means something like [invalid, ground, tree].
        mask_colormap (Optional[dict]): If None, the matplotlib tab20 color map
            is applied to the mask values. A.k.a. a mask value of 1 is given a
            color of tab20(1). If a dict is given, it should be of the form
                {mask value: (4 element RGBA tuple, 0-255)}
            Note that if you use the dict you can leave out certain mask values,
            and they will be left uncolored. Defaults to None.

    Returns: Tuple of two images:
        [0] Image with detections visualized
        [1] Image with the mask visualized
    """

    # Get an alpha-channel image
    image = Image.open(impath).convert("RGBA")

    # Load the given detection types as geopandas dataframes
    def to_gdf(detection):
        """Helper to unify the two input types"""
        if isinstance(detection, RegionDetectionsSet):
            return detection.merge().detections
        else:
            return gpd.read_file(detection)
    gdf1 = to_gdf(detection1)
    gdf2 = to_gdf(detection2)

    # Draw the detections, coloring each detection in gdf1 based on whether
    # it exists in gdf2
    detection_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    detection_draw = ImageDraw.Draw(detection_overlay, "RGBA")
    for idx, row in gdf1.iterrows():
        has_match = row["unique_ID"] in gdf2["unique_ID"].values
        color = (0, 255, 0, 70) if has_match else (255, 0, 0, 70)
        if row.geometry.geom_type == "Polygon":
            polygons = [row.geometry]
        elif row.geometry.geom_type == "MultiPolygon":
            polygons = list(row.geometry.geoms)
        else:
            raise NotImplementedError(f"Can't handle {type(row.geometry)}")
        for poly in polygons:
            # Convert polygon to pixel coordinates
            coords = list(poly.exterior.coords)
            # coords are (x, y), but PIL expects (col, row) so we're good
            detection_draw.polygon(coords, outline=(0, 0, 0, 255), fill=color)

    # Create a colored overlay based on the given mask
    mask_overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    for value in np.unique(mask):
        if mask_colormap is None:
            color = (np.array(colormaps["tab20"](value)) * 255).astype(np.uint8)
            # Set the alpha channel to partially transparent
            color[3] = 100
        else:
            color = mask_colormap.get(value, None)
        if color is not None:
            mask_overlay[mask == value] = color
    mask_overlay = Image.fromarray(mask_overlay, mode="RGBA")

    # Return RGB image arrays
    def to_ndarray(overlay):
        combo = Image.alpha_composite(image, overlay)
        return np.asarray(combo.convert("RGB"))
    return (
        to_ndarray(detection_overlay),
        to_ndarray(mask_overlay),
    )