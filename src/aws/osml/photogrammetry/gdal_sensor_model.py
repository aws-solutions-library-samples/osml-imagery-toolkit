from math import degrees, radians
from typing import Any, Dict, List, Optional

import numpy as np

from .coordinates import GeodeticWorldCoordinate, ImageCoordinate
from .elevation_model import ElevationModel
from .sensor_model import SensorModel


class GDALAffineSensorModel(SensorModel):
    """
    GDAL provides a simple affine transform used to convert XY pixel values to longitude,
    latitude. See https://gdal.org/tutorials/geotransforms_tut.html

    transform[0] x-coordinate of the upper-left corner of the upper-left pixel.
    transform[1] w-e pixel resolution / pixel width.
    transform[2] row rotation (typically zero).
    transform[3] y-coordinate of the upper-left corner of the upper-left pixel.
    transform[4] column rotation (typically zero).
    transform[5] n-s pixel resolution / pixel height (negative value for a north-up image).

    The necessary transform matrix can be obtained from a dataset using the GetGeoTransform() operation.
    """

    def __init__(self, geo_transform: List) -> None:
        """
        Construct the sensor model from the affine transform provided by transform

        :param geo_transform: the 6 coefficients of the affine transform

        :return: None
        """
        super().__init__()
        try:
            # Put the geo transform parameters into a matrix form that makes the image to world calculation
            # matrix multiplication. In this arrangement the 2D image coordinates should be expanded to have a
            # 1.0 at the end to capture the constant. The third row is being added, so we can invert this matrix
            # to obtain the corresponding world to image coefficients.
            self.transform = np.array(
                [
                    [geo_transform[1], geo_transform[2], geo_transform[0]],
                    [geo_transform[4], geo_transform[5], geo_transform[3]],
                    [0.0, 0.0, 1.0],
                ]
            )
            # Use NumPy to calculate an inverse transform
            self.inv_transform = np.linalg.inv(self.transform)
        except np.linalg.LinAlgError:
            raise ValueError("GeoTransform can not be inverted. Not a valid matrix for a sensor model.")

    def image_to_world(
        self,
        image_coordinate: ImageCoordinate,
        elevation_model: Optional[ElevationModel] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GeodeticWorldCoordinate:
        """
        This function returns the longitude, latitude, elevation world coordinate associated with the x, y coordinate
        of any pixel in the image. The GDAL Geo Transforms do not provide any information about elevation, so it will
        always be 0.0 unless the optional elevation model is provided.

        :param image_coordinate: the x, y image coordinate
        :param elevation_model: an optional elevation model used to transform the coordinate
        :param options: an optional dictionary of hints, does not support any hints
        :return: the longitude, latitude, elevation world coordinate
        """
        # The transform is expecting coordinates [x, y, 1.0] as an input.
        augmented_image_coord = np.append(image_coordinate.coordinate, [1.0])
        lonlat_coordinate = np.matmul(self.transform, augmented_image_coord)
        world_coordinate = GeodeticWorldCoordinate([radians(lonlat_coordinate[0]), radians(lonlat_coordinate[1]), 0.0])
        if elevation_model:
            elevation_model.set_elevation(world_coordinate)

        return world_coordinate

    def world_to_image(self, world_coordinate: GeodeticWorldCoordinate) -> ImageCoordinate:
        """
        This function returns the x, y image coordinate associated with a given longitude, latitude, elevation world
        coordinate.

        :param world_coordinate: the longitude, latitude, elevation world coordinate

        :return: the x, y image coordinate
        """
        # The GDAL geo transform does not support elevation data. The inverse transform was created assuming the input
        # coordinate is a 2D geo + 1.0 (i.e. [longitude, latitude, 1.0]
        lonlat_degrees_coordinate = np.array([degrees(world_coordinate.longitude), degrees(world_coordinate.latitude), 1.0])
        xy_coordinate = np.matmul(self.inv_transform, lonlat_degrees_coordinate)
        return ImageCoordinate([xy_coordinate[0], xy_coordinate[1]])
