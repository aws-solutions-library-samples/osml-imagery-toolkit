import unittest
from math import radians

import numpy as np
import pytest


class TestGDALSensorModel(unittest.TestCase):
    def test_gdal_sensor_model(self):
        from aws.osml.photogrammetry.coordinates import ImageCoordinate
        from aws.osml.photogrammetry.elevation_model import ConstantElevationModel
        from aws.osml.photogrammetry.gdal_sensor_model import GDALAffineSensorModel

        sensor_model = GDALAffineSensorModel([0.0, 0.002, 0.0, 0.0, 0.0, 0.003])
        elevation_model = ConstantElevationModel(42.0)
        image_coordinate = ImageCoordinate([200, 300])
        world_coordinate = sensor_model.image_to_world(image_coordinate, elevation_model=elevation_model)
        assert np.array_equal(world_coordinate.coordinate, np.array([radians(0.4), radians(0.9), 42.0]))
        new_image_coordinate = sensor_model.world_to_image(world_coordinate)
        assert np.array_equal(image_coordinate.coordinate, new_image_coordinate.coordinate)

    def test_gdal_sensor_model_real_example(self):
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate, ImageCoordinate
        from aws.osml.photogrammetry.gdal_sensor_model import GDALAffineSensorModel

        transform = (
            -43.681640625,
            4.487879136029412e-06,
            0.0,
            -22.939453125,
            0.0,
            -4.487879136029412e-06,
        )
        sample_gdal_sensor_model = GDALAffineSensorModel(transform)
        sample_image_bounds = [
            ImageCoordinate((0, 0)),
            ImageCoordinate((19584, 0)),
            ImageCoordinate((19584, 19584)),
            ImageCoordinate((0, 19584)),
        ]
        sample_geo_bounds = [
            GeodeticWorldCoordinate((radians(-43.681640625), radians(-22.939453125), 0.0)),
            GeodeticWorldCoordinate((radians(-43.59375), radians(-22.939453125), 0.0)),
            GeodeticWorldCoordinate((radians(-43.59375), radians(-23.02734375), 0.0)),
            GeodeticWorldCoordinate((radians(-43.681640625), radians(-23.02734375), 0.0)),
        ]
        assert (
            pytest.approx(sample_geo_bounds[0].coordinate, rel=1e-6, abs=1e-6)
            == sample_gdal_sensor_model.image_to_world(sample_image_bounds[0]).coordinate
        )
        assert (
            pytest.approx(sample_geo_bounds[1].coordinate, rel=1e-6, abs=1e-6)
            == sample_gdal_sensor_model.image_to_world(sample_image_bounds[1]).coordinate
        )
        assert (
            pytest.approx(sample_image_bounds[0].coordinate, rel=1e-6, abs=1e-6)
            == sample_gdal_sensor_model.world_to_image(sample_geo_bounds[0]).coordinate
        )
        assert (
            pytest.approx(sample_image_bounds[1].coordinate, rel=1e-6, abs=1e-6)
            == sample_gdal_sensor_model.world_to_image(sample_geo_bounds[1]).coordinate
        )

    def test_gdal_non_invertable_transform(self):
        from aws.osml.photogrammetry.gdal_sensor_model import GDALAffineSensorModel

        # This matrix can't be inverted and isn't really a valid GeoTransform.
        # Check to ensure the sensor model raises an error
        transform = [
            -43.681640625,
            0,
            0.0,
            -22.939453125,
            0.0,
            0,
        ]
        with pytest.raises(ValueError):
            GDALAffineSensorModel(transform)


if __name__ == "__main__":
    unittest.main()
