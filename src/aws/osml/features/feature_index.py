from abc import ABC, abstractmethod
from typing import Iterable, Optional

import geojson
import shapely

from .imaged_feature_property_accessor import ImagedFeaturePropertyAccessor


class Feature2DSpatialIndex(ABC):
    """
    A query-only spatial index allowing clients to lookup features using 2D geometries
    """

    @abstractmethod
    def find_intersects(self, geometry: shapely.Geometry) -> Iterable[geojson.Feature]:
        """
        Return the features intersecting the input geometry.

        :param geometry: geometry to query the index
        :return: the features
        """

    @abstractmethod
    def find_nearest(self, geometry: shapely.Geometry, max_distance: Optional[float] = None) -> Iterable[geojson.Feature]:
        """
        Return the nearest feature for the input geometry based on distance within two-dimensional Cartesian space.

        :param geometry: geometry to query the index
        :param max_distance: maximum distance
        :return: the nearest features
        """


class STRFeature2DSpatialIndex(Feature2DSpatialIndex):
    """
    Implementation of the 2D spatial index for GeoJSON features using Shapely's Sort-Tile-Recursive (STR)
    tree datastructure.
    """

    def __init__(
        self,
        feature_collection: geojson.FeatureCollection,
        use_image_geometries: bool = True,
        property_accessor: ImagedFeaturePropertyAccessor = ImagedFeaturePropertyAccessor(),
    ) -> None:
        self.use_image_geometries = use_image_geometries
        self.features = feature_collection.features
        if use_image_geometries and property_accessor is not None:
            geometries = [property_accessor.find_image_geometry(feature) for feature in self.features]
        else:
            geometries = [(shapely.shape(feature.geometry), feature) for feature in self.features]

        self.index = shapely.STRtree(geometries)

    def find_intersects(self, geometry: shapely.Geometry) -> Iterable[geojson.Feature]:
        result_indexes = self.index.query(geometry, predicate="intersects")
        return [self.features[i] for i in result_indexes]

    def find_nearest(self, geometry: shapely.Geometry, max_distance: Optional[float] = None) -> Iterable[geojson.Feature]:
        if max_distance is None:
            if self.use_image_geometries:
                max_distance = 50
            else:
                max_distance = 1.0
        result_indexes = self.index.query_nearest(geometry, max_distance=max_distance)
        return [self.features[i] for i in result_indexes]
