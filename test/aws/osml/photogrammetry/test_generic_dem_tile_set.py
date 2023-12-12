import unittest
from math import radians


class TestGenericDEMTileSet(unittest.TestCase):
    def test_default_format_spec(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.generic_dem_tile_set import GenericDEMTileSet

        tile_set = GenericDEMTileSet()
        tile_path = tile_set.find_tile_id(GeodeticWorldCoordinate([radians(142), radians(3), 0.0]))
        assert "142e/03n.dt2" == tile_path


if __name__ == "__main__":
    unittest.main()
