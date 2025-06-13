import json
import os
import urllib
from typing import List

import geopandas as gpd
import pandas as pd
from deepforest.utilities import DownloadProgressBar


def calculate_scores(
    geometry_column: str,
    confidence_factor: str,
    tile_gdf: gpd.GeoDataFrame,
    image_shape: tuple,
) -> List[float]:
    """Calculate pseudo-confidence scores for the detections based on the following features of the tree crown:

        1. Height - Taller trees are generally more easier to detect, making their confidence higher
        2. Area - Larger tree crowns are easier to detect, hence less likely to be false positives
        3. Distance - Trees near the edge of a tile might have incomplete data, reducing confidence
        4. All - An option to compute a weighted combination of all factors as the confidence score

    Args:
        geometry_column (str): The name of the column in the GeoDataFrame which has the geometries to be used for scoring
        confidence_factor (str): The factor to use for scoring. Choose from: "height", "area", "distance", "all"
        tile_gdf (gpd.GeoDataFrame): A GeoDataFrame with "treetop_height" (if using "height" based scoring) and other relevant columns
        image_shape (tuple): The (i, j, channel) shape of the image that predictions were generated from

    Returns:
        List[float]: Calculated confidence scores.
    """
    if confidence_factor not in ["height", "area", "distance", "all"]:
        raise ValueError(
            "Invalid confidence_factor provided. Choose from: `height`, `area`, `distance`, `all`"
        )

    if confidence_factor == "height":
        # Use height values as scores
        confidence_scores = tile_gdf["treetop_height"]

    elif confidence_factor == "area":
        # Check if all geometries are Polygon
        if (tile_gdf[geometry_column].geom_type == "Point").any():
            raise TypeError(f"Cannot compute scores based on area for Point shapes.")

        # Use area values as scores
        confidence_scores = tile_gdf[geometry_column].apply(lambda geom: geom.area)

    elif confidence_factor == "distance":
        # Calculate the centroid of each tree crown
        tile_gdf["centroid"] = tile_gdf[geometry_column].apply(
            lambda geom: geom.centroid if not geom.is_empty else None
        )

        # Calculate distances to the closest edge for each centroid
        def calculate_edge_distance(centroid):
            if centroid is None:  # Check if centroid is None (empty geometry case)
                return 0
            x, y = centroid.x, centroid.y
            distances = [
                x,  # left edge
                image_shape[1] - x,  # right edge
                y,  # bottom edge
                image_shape[0] - y,  # top edge
            ]
            # Return the distance to the closest edge
            return min(distances)

        tile_gdf["edge_distance"] = tile_gdf["centroid"].apply(calculate_edge_distance)
        # Use edge distance values as scores
        confidence_scores = tile_gdf["edge_distance"]

    elif confidence_factor == "all":
        raise NotImplementedError()

    return list(confidence_scores)


def use_release_df(
    save_dir=os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../../data/"),
    prebuilt_model="NEON",
    check_release=True,
):
    """
    Check the existence of, or download the latest model release from github
    Args:
        save_dir: Directory to save filepath, default to "data" in deepforest repo
        prebuilt_model: Currently only accepts "NEON", but could be expanded to include other prebuilt models. The local model will be called prebuilt_model.h5 on disk.
        check_release (logical): whether to check github for a model recent release. In cases where you are hitting the github API rate limit, set to False and any local model will be downloaded. If no model has been downloaded an error will raise.

    Returns: release_tag, output_path (str): path to downloaded model

    """
    os.makedirs(save_dir, exist_ok=True)

    # Naming based on pre-built model
    output_path = os.path.join(save_dir, prebuilt_model + ".pt")

    if check_release:
        # Find latest github tag release from the DeepLidar repo
        _json = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    "https://api.github.com/repos/Weecology/DeepForest/releases/latest",
                    headers={"Accept": "application/vnd.github.v3+json"},
                )
            ).read()
        )
        asset = _json["assets"][0]
        url = asset["browser_download_url"]

        # Check the release tagged locally
        try:
            release_txt = pd.read_csv(save_dir + "current_release.csv")
        except BaseException:
            release_txt = pd.DataFrame({"current_release": [None]})

        # Download the current release it doesn't exist
        if not release_txt.current_release[0] == _json["html_url"]:

            print(
                "Downloading model from DeepForest release {}, see {} "
                "for details".format(_json["tag_name"], _json["html_url"])
            )

            with DownloadProgressBar(
                unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
            ) as t:
                urllib.request.urlretrieve(
                    url, filename=output_path, reporthook=t.update_to
                )

            print("Model was downloaded and saved to {}".format(output_path))

            # record the release tag locally
            release_txt = pd.DataFrame({"current_release": [_json["html_url"]]})
            release_txt.to_csv(save_dir + "current_release.csv")
        else:
            print(
                "Model from DeepForest release {} was already downloaded. "
                "Loading model from file.".format(_json["html_url"])
            )

        return _json["html_url"], output_path
    else:
        try:
            release_txt = pd.read_csv(save_dir + "current_release.csv")
        except BaseException:
            raise ValueError(
                "Check release argument is {}, but no release "
                "has been previously downloaded".format(check_release)
            )

        return release_txt.current_release[0], output_path
