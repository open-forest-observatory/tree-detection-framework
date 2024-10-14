import pyproj


def get_projected_CRS(
    lat: float, lon: float, assume_western_hem: bool = True
) -> pyproj.CRS:
    """
    Returns a projected Coordinate Reference System (CRS) based on latitude and longitude.

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        assume_western_hem (bool): Assumes the longitude is in the Western Hemisphere. Defaults to True.

    Returns:
        pyproj.CRS: The projected CRS corresponding to the UTM zone for the given latitude and longitude.
    """

    if assume_western_hem and lon > 0:
        lon = -lon
    epgs_code = 32700 - round((45 + lat) / 90) * 100 + round((183 + lon) / 6)
    crs = pyproj.CRS.from_epsg(epgs_code)
    return crs
