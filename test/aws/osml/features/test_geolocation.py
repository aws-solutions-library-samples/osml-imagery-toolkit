#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest

import geojson
import numpy as np
from defusedxml import ElementTree

from aws.osml.features import Geolocator, ImagedFeaturePropertyAccessor


class TestGeolocation(unittest.TestCase):
    def setUp(self):
        from aws.osml.gdal.sensor_model_factory import SensorModelFactory, SensorModelTypes

        with open("test/data/sample-metadata-ms-rpc00b.xml", "rb") as xml_file:
            xml_tres = ElementTree.parse(xml_file)
            sensor_model_builder = SensorModelFactory(
                2048,
                2048,
                xml_tres=xml_tres,
                selected_sensor_model_types=[SensorModelTypes.RPC],
            )
            sensor_model = sensor_model_builder.build()

        self.geolocator = Geolocator(ImagedFeaturePropertyAccessor(), sensor_model)

    def test_geolocate_missing_features(self):
        features = []
        self.geolocator.geolocate_features(features)
        # Nothing to assert; just make sure it doesn't raise an exception.

    def test_geolocate_bbox_feature(self):
        feature = geojson.Feature(
            geometry=None, properties={ImagedFeaturePropertyAccessor.IMAGE_BBOX: [0, 0, 8819.0, 5211.0]}
        )
        self.geolocator.geolocate_features([feature])
        assert feature.bbox is not None
        assert feature.geometry is None
        assert np.allclose(feature.bbox, np.array([121.48749, 24.91148, 121.68595, 25.02860]), atol=1e-2)

    def test_geolocate_point_feature(self):
        feature = geojson.Feature(
            geometry=None,
            properties={ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {"type": "Point", "coordinates": [0, 0]}},
        )

        self.geolocator.geolocate_features([feature])
        assert feature.geometry is not None
        assert isinstance(feature.geometry, geojson.Point)
        assert np.allclose(feature.geometry.coordinates, np.array([121.48749, 25.02860, 377.0]), atol=1e-3)

    def test_geolocate_linestring_feature(self):
        feature = geojson.Feature(
            geometry=None,
            properties={
                ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {
                    "type": "LineString",
                    "coordinates": [[0, 0], [8819.0, 0.0], [8819.0, 5211.0]],
                }
            },
        )

        self.geolocator.geolocate_features([feature])
        assert feature.geometry is not None
        assert isinstance(feature.geometry, geojson.LineString)
        assert np.allclose(
            feature.geometry.coordinates,
            np.array([[121.48749, 25.02860, 377.0], [121.68566, 25.01000, 377.0], [121.68595, 24.91148, 377.0]]),
            atol=1e-3,
        )

    def test_geolocate_linearring_feature(self):
        feature = geojson.Feature(
            geometry=None,
            properties={
                ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {
                    "type": "LinearRing",
                    "coordinates": [[0, 0], [8819.0, 0.0], [8819.0, 5211.0], [0, 0]],
                }
            },
        )

        self.geolocator.geolocate_features([feature])
        assert feature.geometry is not None
        assert isinstance(feature.geometry, geojson.LineString)
        assert np.allclose(
            feature.geometry.coordinates,
            np.array(
                [
                    [
                        [121.48749, 25.02860, 377.0],
                        [121.68566, 25.01000, 377.0],
                        [121.68595, 24.91148, 377.0],
                        [121.48749, 25.02860, 377.0],
                    ]
                ]
            ),
            atol=1e-3,
        )

    def test_geolocate_polygon_feature(self):
        feature = geojson.Feature(
            geometry=None,
            properties={
                ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [8819.0, 0.0], [8819.0, 5211.0], [0, 0]]],
                }
            },
        )

        self.geolocator.geolocate_features([feature])
        assert feature.geometry is not None
        assert isinstance(feature.geometry, geojson.Polygon)
        assert np.allclose(
            feature.geometry.coordinates,
            np.array(
                [
                    [
                        [121.48749, 25.02860, 377.0],
                        [121.68566, 25.01000, 377.0],
                        [121.68595, 24.91148, 377.0],
                        [121.48749, 25.02860, 377.0],
                    ]
                ]
            ),
            atol=1e-3,
        )

    def test_geolocate_multipoint_feature(self):
        feature = geojson.Feature(
            geometry=None,
            properties={
                ImagedFeaturePropertyAccessor.IMAGE_GEOMETRY: {"type": "MultiPoint", "coordinates": [[0, 0], [8819.0, 0.0]]}
            },
        )

        self.geolocator.geolocate_features([feature])
        assert feature.geometry is not None
        assert isinstance(feature.geometry, geojson.MultiPoint)
        assert np.allclose(
            feature.geometry.coordinates, np.array([[121.48749, 25.02860, 377.0], [121.68566, 25.01000, 377.0]]), atol=1e-3
        )

    def test_geolocate_bounds_imcoords_feature(self):
        feature = geojson.Feature(
            geometry=None,
            properties={
                "bounds_imcoords": [0, 0, 10, 10],
                "detection_score": 0.95,
                "feature_types": {"foo": 0.7},
                "image_id": "fake-image-id",
            },
        )

        self.geolocator.geolocate_features([feature])

        # Check to make sure the geolocation capability finds the bounds_imcoord property and creates a
        # polygon feature
        assert feature.geometry is not None
        assert isinstance(feature.geometry, geojson.Polygon)

        # Check to ensure the exterior boundary of the polygon has all 5 points (4 corners + repeat of 1st corner)
        assert len(feature.geometry.coordinates[0]) == 5
        assert np.allclose(
            feature.geometry.coordinates,
            np.array(
                [
                    [
                        [121.489307, 25.027718, 377.0],
                        [121.489308, 25.027526, 377.0],
                        [121.489082, 25.027546, 377.0],
                        [121.489081, 25.027739, 377.0],
                        [121.489307, 25.027718, 377.0],
                    ]
                ]
            ),
            atol=1e-3,
        )

    def test_geolocate_geom_imcoords_feature(self):
        feature = geojson.Feature(
            geometry=None,
            properties={
                "geom_imcoords": [[0, 0], [8819.0, 0.0], [8819.0, 5211.0], [0, 0]],
                "detection_score": 0.95,
                "feature_types": {"aircraft": 0.7},
                "image_id": "fake-image-id",
            },
        )

        self.geolocator.geolocate_features([feature])

        # Check to make sure the geolocation capability finds the geom_imcoord property and creates a
        # polygon feature
        assert feature.geometry is not None
        assert isinstance(feature.geometry, geojson.Polygon)
        assert np.allclose(
            feature.geometry.coordinates,
            np.array(
                [
                    [
                        [121.48749, 25.02860, 377.0],
                        [121.68566, 25.01000, 377.0],
                        [121.68595, 24.91148, 377.0],
                        [121.48749, 25.02860, 377.0],
                    ]
                ]
            ),
            atol=1e-3,
        )
