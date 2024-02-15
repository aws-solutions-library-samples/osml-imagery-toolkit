import json
from typing import Optional

import geojson
import shapely


class ImagedFeaturePropertyAccessor:
    """
    This class contains utility functions that ensure the property names / values for features derived from imagery
    are consistently implemented. These specifications are still evolving so the intent is to encapsulate all of the
    names in this one class so that changes do not ripple through the rest of the software baseline.
    """

    IMAGE_GEOMETRY = "imageGeometry"
    IMAGE_BBOX = "imageBBox"

    BOUNDS_IMCORDS = "bounds_imcoords"
    GEOM_IMCOORDS = "geom_imcoords"
    DETECTION = "detection"
    TYPE = "type"
    COORDINATES = "coordinates"
    PIXEL_COORDINATES = "pixelCoordinates"

    def __init__(self, allow_deprecated: bool = True):
        """
        Construct an instance of the property accessor with configuration options.

        :param allow_deprecated: if true the accessor will work with deprecated property names.
        """
        self.allow_deprecated = allow_deprecated
        pass

    def find_image_geometry(self, feature: geojson.Feature) -> Optional[shapely.Geometry]:
        """
        This function searches through the properties of a GeoJSON feature that are known to contain the geometry
        of the feature in image coordinates. If found an appropriate 2D shape is constructed and returned. Note that
        this search is conducted in priority order giving preference to the current preferred "imageGeometry" and
        "bboxGeometry" properties. If neither of those is available and the accessor has been configured to search
        deprecated properties then the "geom_imcoords", "detection", and "bounds_imcoords" properties are searched
        in that order.

        :param feature: a GeoJSON feature that might contain an image geometry property
        :return: a 2D shape representing the image geometry or None
        """
        # The "imageGeometry" property is the current preferred encoding of image geometries for these
        # features. The format follows the same type and coordinates structure used by shapely so we can
        # construct the geometry directly from these values.
        if self.IMAGE_GEOMETRY in feature.properties:
            return shapely.geometry.shape(feature.properties[self.IMAGE_GEOMETRY])

        # If a full image geometry is not provided we might be able to construct a Polygon boundary from the
        # "imageBBox" property. The property contains a [minx, miny, maxx, maxy] bounding box. If available we
        # can construct a Polygon boundary from those 4 corners.
        if self.IMAGE_BBOX in feature.properties:
            bbox = feature.properties[self.IMAGE_BBOX]
            return shapely.geometry.box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])

        # !!!!! ALL PROPERTIES BELOW THIS LINE ARE DEPRECATED !!!!!
        if self.allow_deprecated:
            # The current convention for the "geom_imcoords" allows a single external ring for a Polygon boundary to be
            # captured as a list of coordinates.
            if self.GEOM_IMCOORDS in feature.properties:
                return shapely.geometry.Polygon(shell=feature.properties[self.GEOM_IMCOORDS])

            # Some inputs may have a "detection" property with child "type" and "pixelCoordinates" properties. If these
            # are found we can construct the appropriate shape.
            if self.DETECTION in feature.properties and self.PIXEL_COORDINATES in feature.properties[self.DETECTION]:
                temp_geom = {
                    self.TYPE: feature.properties[self.DETECTION][self.TYPE],
                    self.COORDINATES: feature.properties[self.DETECTION][self.PIXEL_COORDINATES],
                }
                return shapely.geometry.shape(temp_geom)

            # The current convention for "bounds_imcoords" is a [minx, miny, maxx, maxy] bounding box. If available we
            # can construct a Polygon boundary from those 4 corners.
            if self.BOUNDS_IMCORDS in feature.properties:
                bbox = feature.properties[self.BOUNDS_IMCORDS]
                return shapely.geometry.box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])

        # All properties that might contain the image geometry are missing. This feature does not have image
        # coordinates.
        return None

    def update_existing_image_geometries(self, feature: geojson.Feature, geometry: shapely.Geometry) -> None:
        """
        This function searches through the properties of a GeoJSON feature that are known to contain the geometry
        of the feature in image coordinates. If found each property is overwritten with information from the
        geometry provided. Note that for bounding box properties the bounds of the input geometry are used.

        :param feature: a GeoJSON feature that might contain an image geometry property
        :param geometry: the geometry to set property values for.
        """
        if self.IMAGE_GEOMETRY in feature.properties:
            ImagedFeaturePropertyAccessor.set_image_geometry(feature, geometry)

        if self.IMAGE_BBOX in feature.properties:
            ImagedFeaturePropertyAccessor.set_image_bbox(feature, geometry)

        # !!!!! ALL PROPERTIES BELOW THIS LINE ARE DEPRECATED !!!!!
        if self.allow_deprecated:
            if self.GEOM_IMCOORDS in feature.properties:
                coordinates = shapely.geometry.mapping(geometry)[self.COORDINATES]
                if isinstance(geometry, shapely.geometry.Polygon):
                    feature.properties[self.GEOM_IMCOORDS] = coordinates[0]
                else:
                    feature.properties[self.GEOM_IMCOORDS] = coordinates

            if self.DETECTION in feature.properties and self.PIXEL_COORDINATES in feature.properties[self.DETECTION]:
                geometry_mapping = shapely.geometry.mapping(geometry)
                feature.properties[self.DETECTION][self.TYPE] = geometry_mapping[self.TYPE]
                feature.properties[self.DETECTION][self.PIXEL_COORDINATES] = geometry_mapping[self.COORDINATES]

            if self.BOUNDS_IMCORDS in feature.properties:
                feature.properties[self.BOUNDS_IMCORDS] = list(geometry.bounds)

    @classmethod
    def get_image_geometry(cls, feature: geojson.Feature) -> Optional[shapely.Geometry]:
        if cls.IMAGE_GEOMETRY in feature["properties"]:
            return shapely.geometry.shape(feature.properties[cls.IMAGE_GEOMETRY])
        return None

    @classmethod
    def get_image_bbox(cls, feature: geojson.Feature) -> Optional[shapely.Geometry]:
        if cls.IMAGE_BBOX in feature["properties"]:
            bbox = feature.properties[cls.IMAGE_BBOX]
            return shapely.geometry.box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
        return None

    @classmethod
    def set_image_geometry(cls, feature: geojson.Feature, geometry: shapely.Geometry) -> None:
        """
        Add or set the "imageGeometry" property for a feature. This is a 2D geometry that supports a variety of
        types (points, lines, polygons, etc.)

        :param feature: a GeoJSON feature that will contain the property
        :param geometry: the geometry value
        """
        feature.properties[cls.IMAGE_GEOMETRY] = json.loads(shapely.to_geojson(geometry))

    @classmethod
    def set_image_bbox(cls, feature: geojson.Feature, geometry: shapely.Geometry) -> None:
        """
        Add or set the "imageBBox" property for a feature. this is a [minx, miny, maxx, maxy] bounds for this object.

        :param feature: a GeoJSON feature that will contain the property
        :param geometry: the geometry value
        """
        feature.properties[cls.IMAGE_BBOX] = list(geometry.bounds)
