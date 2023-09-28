from math import radians
from typing import List, Optional

from osgeo import gdal

from aws.osml.photogrammetry import GDALAffineSensorModel, GeodeticWorldCoordinate, ImageCoordinate, ProjectiveSensorModel

from .sensor_model_builder import SensorModelBuilder


class GDALAffineSensorModelBuilder(SensorModelBuilder):
    """
    This builder is used to create sensor models for images that have GDAL geo transforms.
    """

    def __init__(self, geo_transform: List[float], proj_wkt: Optional[str] = None) -> None:
        """
        Constructor for the builder accepting the required GDAL geotransform.

        :param geo_transform: the geotransform for this image
        :param proj_wkt: the well known text string of the CRS used by the image

        :return: None
        """
        super().__init__()
        self.geo_transform = geo_transform
        self.proj_wkt = proj_wkt

    def build(self) -> Optional[GDALAffineSensorModel]:
        """
        Use the GDAL GeoTransform to construct a sensor model.

        :return: affine transform based SensorModel that uses the GDAL GeoTransform
        """
        if self.geo_transform is None:
            return None
        return GDALAffineSensorModel(self.geo_transform, self.proj_wkt)


class GDALGCPSensorModelBuilder(SensorModelBuilder):
    """
    This builder is used to create sensor models for images that have GDAL ground control points (GCPs).
    """

    def __init__(self, ground_control_points: List[gdal.GCP]) -> None:
        """
        Constructor for the builder accepting the required GDAL GCPs.

        :param ground_control_points: the ground control points for this image

        :return: None
        """
        super().__init__()
        self.ground_control_points = ground_control_points

    def build(self) -> Optional[ProjectiveSensorModel]:
        """
        Use the GCPs to construct a projective sensor model.

        :return: a projective transform SensorModel that uses the GDAL GCPs provided
        """
        if not self.ground_control_points or len(self.ground_control_points) < 4:
            return None

        world_coordinates = [
            GeodeticWorldCoordinate([radians(gcp.GCPX), radians(gcp.GCPY), gcp.GCPZ]) for gcp in self.ground_control_points
        ]
        image_coordinates = [ImageCoordinate([gcp.GCPPixel, gcp.GCPLine]) for gcp in self.ground_control_points]
        return ProjectiveSensorModel(world_coordinates, image_coordinates)
