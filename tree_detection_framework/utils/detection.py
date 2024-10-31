import json
import os
import urllib

import pandas as pd
from deepforest.utilities import DownloadProgressBar


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
