from typing import List, Optional, Union

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
) -> List[np.ndarray]:
    """
    Arguments:

    Returns:
    """

    # TODO
    image = Image.open(impath).convert("RGBA")

    # TODO
    def to_gdf(detection):
        """Helper to unify the two input types"""
        if isinstance(detection, RegionDetectionsSet):
            return detection.merge().detections
        else:
            return gpd.read_file(detection)
    gdf1 = to_gdf(detection1)
    gdf2 = to_gdf(detection2)

    # TODO
    detection_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    detection_draw = ImageDraw.Draw(detection_overlay, "RGBA")
    for idx, row in gdf1.iterrows():
        has_match = row["unique_ID"] in gdf2["unique_ID"]
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

    # TODO
    mask_overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    for value in np.unique(mask):
        if mask_colormap is None:
            color = (np.array(colormaps["tab20"](value)) * 255).astype(np.uint8)
            # Set the alpha channel to partially transparent
            color[3] = 100
        else:
            color = mask_colormap[value]
        mask_overlay[mask == value] = color
    mask_overlay = Image.fromarray(mask_overlay, mode="RGBA")

    # TODO
    def to_ndarray(overlay):
        combo = Image.alpha_composite(image, overlay)
        return np.asarray(combo.convert("RGB"))
    return (
        to_ndarray(detection_overlay),
        to_ndarray(mask_overlay),
    )