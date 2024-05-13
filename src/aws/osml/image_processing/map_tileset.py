#  Copyright 2024 Amazon.com, Inc. or its affiliates.

from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass

from aws.osml.photogrammetry import GeodeticWorldCoordinate

MapTileId = namedtuple("TildId", "tile_matrix tile_row tile_col")
MapTileId.__doc__ = """
This type represents the unique id of a map tile within a map tile set. It is implemented as a named tuple to make
use of that constructs immutability and hashing features.
"""

MapTileSize = namedtuple("MapTileSize", "width height")
MapTileSize.__doc__ = """
This type represents the size of a map tile (width, height). These dimensions are typically the same (i.e. square
tiles) but this is not required.
"""

MapTileBounds = namedtuple("MapTileBounds", "min_lon min_lat max_lon max_lat")
MapTileBounds.__doc__ = """
This type represents the geodetic bounds of a map tile (min_lon, min_lat, max_lon, max_lat).
"""


@dataclass
class MapTile:
    """
    This dataclass provides a description of a map tile that is part of a well known tile set.
    """

    id: MapTileId
    size: MapTileSize
    bounds: MapTileBounds


class MapTileSet(ABC):
    """
    This class provides an abstract interface to a well known set of map tiles.
    """

    @property
    @abstractmethod
    def tile_matrix_set_id(self) -> str:
        """
        Get the identifier for this map tile set. This is the tile matrix set ID in the OGC definitions.

        :return: the tile matrix set ID
        """

    @abstractmethod
    def get_tile(self, tile_id: MapTileId) -> MapTile:
        """
        Get a description of the tile identified by a specific map tile ID.

        :param tile_id: the tile ID
        :return: the tile description
        """

    @abstractmethod
    def get_tile_for_location(self, world_coordinate: GeodeticWorldCoordinate, tile_matrix: int) -> MapTile:
        """
        Get a description of the tile containing a specific world location.

        :param world_coordinate: the location in the world
        :param tile_matrix: the tile_matrix or zoom level of interest
        :return: the tile description
        """

    def get_tile_matrix_limits_for_area(
        self, boundary_coordinates: list[GeodeticWorldCoordinate], tile_matrix: int
    ) -> tuple[int, int, int, int]:
        """
        Get a list of all tiles that intersect a specific area.

        :param boundary_coordinates: the boundary of the area
        :param tile_matrix: the tile_matrix or zoom level of interest
        :return: the (min_col, min_row, max_col, max_row) limits of tiles containing all points
        """
        map_tiles_for_corners = [
            self.get_tile_for_location(world_corner, tile_matrix=tile_matrix) for world_corner in boundary_coordinates
        ]
        tile_rows = [tile_id.id.tile_row for tile_id in map_tiles_for_corners]
        tile_cols = [tile_id.id.tile_col for tile_id in map_tiles_for_corners]

        return min(tile_cols), min(tile_rows), max(tile_cols), max(tile_rows)
