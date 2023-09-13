import unittest
from math import degrees, radians
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pytest
from defusedxml import ElementTree
from osgeo import gdal

from configuration import TEST_ENV_CONFIG

# Strictly speaking the tests in this file are not pure unit tests of the SensorModelFactory. Here we deliberately
# did not mock out the underlying sensor models and are instead testing that a fully functional model can be
# constructed from the example metadata provided. Small integration tests like these, run as part of the automated
# unit testing, will ensure that the complex sensor models perform correctly when exposed to real metadata. These
# tests do not have any dependencies on external infrastructure and should run fast enough that we shouldn't need
# to break them out from the other automated tests.


@patch.dict("os.environ", TEST_ENV_CONFIG, clear=True)
class TestSensorModelFactory(TestCase):
    def test_sensor_model_builder_ms_rpc00b_with_cscrna(self):
        from aws.osml.gdal.sensor_model_factory import SensorModelFactory
        from aws.osml.photogrammetry.composite_sensor_model import CompositeSensorModel
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate
        from aws.osml.photogrammetry.projective_sensor_model import ProjectiveSensorModel
        from aws.osml.photogrammetry.rpc_sensor_model import RPCSensorModel

        with open("test/data/sample-metadata-ms-rpc00b.xml", "rb") as xml_file:
            xml_tres = ElementTree.parse(xml_file)
            sensor_model_builder = SensorModelFactory(2048, 2048, xml_tres=xml_tres)
            sensor_model = sensor_model_builder.build()
            assert sensor_model is not None
            assert isinstance(sensor_model, CompositeSensorModel)
            assert isinstance(sensor_model.approximate_sensor_model, ProjectiveSensorModel)
            assert isinstance(sensor_model.precision_sensor_model, RPCSensorModel)

            # These are the corner coordinates taken from the CSCRNA TRE. To represent the corners of the "intelligent
            # pixels" which are the pixels that actually contain visual information (i.e. not padding pixels). A more
            # complete definition is in STDI-0002 Volume 3 Appendix B. Note that the image coordinates are not the full
            # image size but the location of the corners without the padding.
            ulcorner_world_coordinate = GeodeticWorldCoordinate([radians(121.48749), radians(25.02860), 27.1])
            ulcorner_image_coordinate = sensor_model.world_to_image(ulcorner_world_coordinate)
            assert np.allclose(ulcorner_image_coordinate.coordinate, np.array([0.0, 0.0]), atol=1.0)

            urcorner_world_coordinate = GeodeticWorldCoordinate([radians(121.68566), radians(25.01000), 234.7])
            urcorner_image_coordinate = sensor_model.world_to_image(urcorner_world_coordinate)
            assert np.allclose(urcorner_image_coordinate.coordinate, np.array([8819.0, 0.0]), atol=1.0)

            lrcorner_world_coordinate = GeodeticWorldCoordinate([radians(121.68595), radians(24.91148), 403.1])
            lrcorner_image_coordinate = sensor_model.world_to_image(lrcorner_world_coordinate)
            assert np.allclose(lrcorner_image_coordinate.coordinate, np.array([8819.0, 5211.0]), atol=1.0)

            llcorner_world_coordinate = GeodeticWorldCoordinate([radians(121.48975), radians(24.92772), 431.4])
            llcorner_image_coordinate = sensor_model.world_to_image(llcorner_world_coordinate)
            assert np.allclose(llcorner_image_coordinate.coordinate, np.array([0.0, 5211.0]), atol=1.0)

    def test_sensor_model_builder_selected_sensors(self):
        from aws.osml.gdal.sensor_model_factory import SensorModelFactory, SensorModelTypes
        from aws.osml.photogrammetry.projective_sensor_model import ProjectiveSensorModel

        with open("test/data/sample-metadata-ms-rpc00b.xml", "rb") as xml_file:
            xml_tres = ElementTree.parse(xml_file)
            sensor_model_builder = SensorModelFactory(
                2048,
                2048,
                xml_tres=xml_tres,
                selected_sensor_model_types=[SensorModelTypes.AFFINE, SensorModelTypes.PROJECTIVE],
            )
            sensor_model = sensor_model_builder.build()
            assert sensor_model is not None
            assert isinstance(sensor_model, ProjectiveSensorModel)

    def test_sensor_model_builder_ms_rpc00b_with_chip(self):
        from aws.osml.gdal.sensor_model_factory import SensorModelFactory, SensorModelTypes
        from aws.osml.photogrammetry.chipped_image_sensor_model import ChippedImageSensorModel
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate, ImageCoordinate
        from aws.osml.photogrammetry.rpc_sensor_model import RPCSensorModel

        with open("test/data/sample-metadata-rpc00b-ichipb.xml", "rb") as xml_file:
            xml_tres = ElementTree.parse(xml_file)
            sensor_model_builder = SensorModelFactory(
                512,
                512,
                xml_tres=xml_tres,
                selected_sensor_model_types=[SensorModelTypes.PROJECTIVE, SensorModelTypes.RPC],
            )
            sensor_model = sensor_model_builder.build()
            assert sensor_model is not None
            assert isinstance(sensor_model, ChippedImageSensorModel)
            assert isinstance(sensor_model.full_image_sensor_model, RPCSensorModel)

            # This is the NITF_DES_POS_01.ntf test image from JITC which has a fixed known location at these coordinates
            world_coordinate = sensor_model.image_to_world(ImageCoordinate([263, 280]))
            assert degrees(world_coordinate.longitude) == pytest.approx(44.35267, abs=0.00001)
            assert degrees(world_coordinate.latitude) == pytest.approx(33.36305, abs=0.00001)
            assert world_coordinate.elevation == pytest.approx(31.0, abs=1.0)

            image_coordinate = sensor_model.world_to_image(
                GeodeticWorldCoordinate([radians(44.35267), radians(33.36305), 31.0])
            )
            assert image_coordinate.x == pytest.approx(263, abs=3)
            assert image_coordinate.y == pytest.approx(280, abs=3)

    def test_sensor_model_builder_rsmpca(self):
        from aws.osml.gdal.sensor_model_factory import SensorModelFactory, SensorModelTypes
        from aws.osml.photogrammetry.coordinates import GeodeticWorldCoordinate, ImageCoordinate
        from aws.osml.photogrammetry.replacement_sensor_model import RSMPolynomialSensorModel

        with open("test/data/i_6130a_truncated_tres.xml") as xml_file:
            xml_tres = ElementTree.parse(xml_file)
            sensor_model_builder = SensorModelFactory(
                2048, 2048, xml_tres=xml_tres, selected_sensor_model_types=[SensorModelTypes.RSM]
            )
            sensor_model = sensor_model_builder.build()
            assert sensor_model is not None
            assert isinstance(sensor_model, RSMPolynomialSensorModel)

            geodetic_ground_domain_origin = GeodeticWorldCoordinate([radians(-117.03881), radians(33.16173), -6.7])
            image_ground_domain_origin = sensor_model.world_to_image(geodetic_ground_domain_origin)
            assert image_ground_domain_origin.x == pytest.approx(0.5, abs=1.0)
            assert image_ground_domain_origin.y == pytest.approx(0.5, abs=1.0)

            new_geodetic_ground_domain_origin = sensor_model.image_to_world(ImageCoordinate([0.5, 0.5]))
            assert new_geodetic_ground_domain_origin.longitude == pytest.approx(
                geodetic_ground_domain_origin.longitude, abs=0.00001
            )
            assert new_geodetic_ground_domain_origin.latitude == pytest.approx(
                geodetic_ground_domain_origin.latitude, abs=0.00001
            )

    def test_sensor_model_builder_cscrna(self):
        from aws.osml.gdal.sensor_model_factory import SensorModelFactory
        from aws.osml.photogrammetry.projective_sensor_model import ProjectiveSensorModel

        with open("test/data/sample-metadata-cscrna.xml", "rb") as xml_file:
            xml_tres = ElementTree.parse(xml_file)
            sensor_model_builder = SensorModelFactory(2048, 2048, xml_tres=xml_tres)
            sensor_model = sensor_model_builder.build()
            assert isinstance(sensor_model, ProjectiveSensorModel)

    def test_sensor_model_builder_gcps(self):
        from aws.osml.gdal.sensor_model_factory import SensorModelFactory
        from aws.osml.photogrammetry.coordinates import ImageCoordinate
        from aws.osml.photogrammetry.projective_sensor_model import ProjectiveSensorModel

        gcps = (
            gdal.GCP(121.67722222222223, 13.924722222222222, 0.0, 0.5, 0.5),
            gdal.GCP(121.8261111111111, 13.91861111111111, 0.0, 10239.5, 0.5),
            gdal.GCP(121.8261111111111, 13.858333333333333, 0.0, 10239.5, 4095.5),
            gdal.GCP(121.67722222222223, 13.864166666666666, 0.0, 0.5, 4095.5),
        )

        sensor_model_builder = SensorModelFactory(2048, 2048, ground_control_points=gcps)
        sensor_model = sensor_model_builder.build()

        assert sensor_model is not None
        assert isinstance(sensor_model, ProjectiveSensorModel)

        image_center = ImageCoordinate([5120, 2048])
        geodetic_image_center = sensor_model.image_to_world(image_center)

        assert np.allclose(
            geodetic_image_center.coordinate,
            np.array([radians(121.7518378), radians(13.89145147), 0.0]),
        )

    def test_sicd_sensor_models(self):
        from aws.osml.gdal.sensor_model_factory import SensorModelFactory
        from aws.osml.photogrammetry import ImageCoordinate, SICDSensorModel, geocentric_to_geodetic

        test_examples = [
            "./test/data/sicd/capella-sicd121-chip1.ntf",
            "./test/data/sicd/capella-sicd121-chip2.ntf",
            "./test/data/sicd/umbra-sicd121-chip1.ntf",
        ]
        for image_path in test_examples:
            ds = gdal.Open(image_path)
            xml_dess = ds.GetMetadata("xml:DES")
            factory = SensorModelFactory(ds.RasterXSize, ds.RasterYSize, xml_dess=xml_dess)
            sm = factory.build()
            assert sm is not None
            assert isinstance(sm, SICDSensorModel)

            scp_image_coord = ImageCoordinate(
                [
                    sm.coord_converter.scp_pixel.x - sm.coord_converter.first_pixel.x,
                    sm.coord_converter.scp_pixel.y - sm.coord_converter.first_pixel.y,
                ]
            )
            scp_world_coord = geocentric_to_geodetic(sm.coord_converter.scp_ecf)

            assert np.allclose(scp_image_coord.coordinate, sm.world_to_image(scp_world_coord).coordinate, atol=1.0)

            assert np.allclose(scp_world_coord.coordinate, sm.image_to_world(scp_image_coord).coordinate)


if __name__ == "__main__":
    unittest.main()
