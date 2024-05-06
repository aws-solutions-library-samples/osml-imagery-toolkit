#  Copyright 2024 Amazon.com, Inc. or its affiliates.
from unittest import TestCase

import numpy as np

from aws.osml.image_processing import MapTileId, MapTileSetFactory, WellKnownMapTileSet
from aws.osml.photogrammetry import GeodeticWorldCoordinate


class TestWebMercatorQuadTileSet(TestCase):
    def setUp(self) -> None:
        self.wmq_tile_set = MapTileSetFactory.get_for_id(WellKnownMapTileSet.WEB_MERCATOR_QUAD)
        self.wmqx2_tile_set = MapTileSetFactory.get_for_id(WellKnownMapTileSet.WEB_MERCATOR_QUAD_X2)

    def test_well_known_ids(self):
        self.assertEquals(self.wmq_tile_set.tile_matrix_set_id, WellKnownMapTileSet.WEB_MERCATOR_QUAD.value)
        self.assertEquals(self.wmqx2_tile_set.tile_matrix_set_id, WellKnownMapTileSet.WEB_MERCATOR_QUAD_X2.value)

    def test_get_tile(self):
        top_tile_id = MapTileId(tile_matrix=0, tile_row=0, tile_col=0)
        top_tile_256 = self.wmq_tile_set.get_tile(top_tile_id)
        top_tile_512 = self.wmqx2_tile_set.get_tile(top_tile_id)

        self.assertEquals(top_tile_256.id, top_tile_id)
        np.testing.assert_almost_equal(
            top_tile_256.bounds,
            (np.radians(-180.0), np.radians(-85.051128), np.radians(180.0), np.radians(85.051128)),
            decimal=7,
        )
        self.assertEquals(top_tile_256.size, (256, 256))

        self.assertEquals(top_tile_512.id, top_tile_id)
        self.assertEquals(top_tile_512.bounds, top_tile_256.bounds)
        self.assertEquals(top_tile_512.size, (512, 512))

        test_tile_id = MapTileId(tile_matrix=10, tile_row=578, tile_col=856)
        test_tile = self.wmq_tile_set.get_tile(test_tile_id)
        self.assertEquals(test_tile.id, test_tile_id)
        np.testing.assert_almost_equal(test_tile.bounds, (2.1107576, 0.3943349, 2.1168935, 0.3999932), decimal=7)
        self.assertEquals(test_tile.size, (256, 256))

        test_tile = self.wmqx2_tile_set.get_tile(test_tile_id)
        self.assertEquals(test_tile.id, test_tile_id)
        np.testing.assert_almost_equal(test_tile.bounds, (2.1107576, 0.3943349, 2.1168935, 0.3999932), decimal=7)
        self.assertEquals(test_tile.size, (512, 512))

    def test_get_tile_for_location(self):
        expected_tile_id = MapTileId(tile_matrix=10, tile_row=578, tile_col=856)
        test_location = GeodeticWorldCoordinate([2.113, 0.395, 0.0])

        test_tile = self.wmq_tile_set.get_tile_for_location(test_location, tile_matrix=expected_tile_id.tile_matrix)
        self.assertEquals(test_tile.id, expected_tile_id)

        test_tile = self.wmqx2_tile_set.get_tile_for_location(test_location, tile_matrix=expected_tile_id.tile_matrix)
        self.assertEquals(test_tile.id, expected_tile_id)
