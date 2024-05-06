#  Copyright 2024 Amazon.com, Inc. or its affiliates.

from enum import Enum
from typing import Optional

from .map_tileset import MapTileSet
from .map_tileset_wmq import WebMercatorQuadMapTileSet


class WellKnownMapTileSet(str, Enum):
    """
    A partial list of well known tile sets used by this library.
    """

    WEB_MERCATOR_QUAD = "WebMercatorQuad"
    WEB_MERCATOR_QUAD_X2 = "WebMercatorQuadx2"


class MapTileSetFactory:
    """
    This class provides a means to construct / access the implementation of well known tile sets using their
    name. It allows clients to easily work with the TileSet abstraction independent of any implementation details
    associated with specific tile sets.
    """

    @staticmethod
    def get_for_id(tile_matrix_set_id: str) -> Optional[MapTileSet]:
        """
        Constructs a tile set matching the requested id.

        :param tile_matrix_set_id: the tile set id
        :return: the TileSet or None if not available
        """
        if tile_matrix_set_id == WellKnownMapTileSet.WEB_MERCATOR_QUAD.value:
            return WebMercatorQuadMapTileSet(tile_size=256, tile_matrix_set_id=WellKnownMapTileSet.WEB_MERCATOR_QUAD.value)
        elif tile_matrix_set_id == WellKnownMapTileSet.WEB_MERCATOR_QUAD_X2.value:
            return WebMercatorQuadMapTileSet(
                tile_size=512, tile_matrix_set_id=WellKnownMapTileSet.WEB_MERCATOR_QUAD_X2.value
            )
        else:
            return None
