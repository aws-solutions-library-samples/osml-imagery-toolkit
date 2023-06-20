from math import degrees, floor
from typing import Optional

from .coordinates import GeodeticWorldCoordinate
from .digital_elevation_model import DigitalElevationModelTileSet


class SRTMTileSet(DigitalElevationModelTileSet):
    """
    A tile set for SRTM content downloaded from the USGS website.
    """

    def __init__(self, prefix: str = "", version: str = "1arc_v3", format_extension: str = ".tif") -> None:
        """
        Construct a tile set from a limited collection of configurable parameters. This implementation is flexible
        enough to support both SRTM 1-arc second and 3-arc second datasets in whatever raster format (e.g. GeoTIFF).

        :param prefix: an optional prefix (possibly a subdirectory) for the tile set
        :param version: the version (e.g. 1arc_v3 or 3arc_v2)
        :param format_extension: the image extension (e.g. .tif)

        :return: None
        """
        super().__init__()
        self.prefix = prefix
        self.version = version
        self.format_extension = format_extension

    def find_tile_id(self, geodetic_world_coordinate: GeodeticWorldCoordinate) -> Optional[str]:
        """
        This method creates tile IDs that match the file names for grid tiles downloaded using the USGS Earth Eplorer.
        It appears they are following a <latitude degrees>_<longitude degrees>_<resolution>_<version><format>
        convention. Examples:

        - n47_e034_3arc_v2.tif: 3 arc second resolution
        - n47_e034_1arc_v3.tif: 1 arc second resolution

        :param geodetic_world_coordinate: the world coordinate of interest

        :return: the tile path or None if the DEM does not have coverage for this location
        """
        longitude_degrees = floor(degrees(geodetic_world_coordinate.longitude))
        latitude_degrees = floor(degrees(geodetic_world_coordinate.latitude))

        # The SRTM mission only covers latitudes N59 through S56 so if the requested location is outside those
        # ranges we know there is no file available for it.
        if latitude_degrees > 59 or latitude_degrees < -56:
            return None

        longitude_direction = "e"
        if longitude_degrees < 0:
            longitude_direction = "w"

        latitude_direction = "n"
        if latitude_degrees < 0:
            latitude_direction = "s"

        return (
            f"{self.prefix}"
            f"{latitude_direction}{abs(latitude_degrees):02d}_"
            f"{longitude_direction}{abs(longitude_degrees):03d}_"
            f"{self.version}"
            f"{self.format_extension}"
        )
