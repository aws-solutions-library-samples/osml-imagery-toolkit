import unittest
from math import radians
from unittest import TestCase
from unittest.mock import patch

import pytest
from osgeo import gdal

from configuration import TEST_ENV_CONFIG


@patch.dict("os.environ", TEST_ENV_CONFIG, clear=True)
class TestGDALDemTileFactory(TestCase):
    def setUp(self):
        # GDAL 4.0 will begin using exceptions as the default; at this point the software is written to assume
        # no exceptions so we call this explicitly until the software can be updated to match.
        gdal.DontUseExceptions()

    def test_load_geotiff_tile(self):
        from aws.osml.gdal.gdal_dem_tile_factory import GDALDigitalElevationModelTileFactory
        from aws.osml.photogrammetry import GeodeticWorldCoordinate

        tile_factory = GDALDigitalElevationModelTileFactory("./test/data")
        elevation_array, sensor_model = tile_factory.get_tile("n47_e034_3arc_v2.tif")

        assert elevation_array is not None
        assert elevation_array.shape == (1201, 1201)
        assert sensor_model is not None

        center_image = sensor_model.world_to_image(GeodeticWorldCoordinate([radians(34.5), radians(47.5), 0.0]))

        assert center_image.x == pytest.approx(600.5, abs=1.0)
        assert center_image.y == pytest.approx(600.5, abs=1.0)


if __name__ == "__main__":
    unittest.main()
