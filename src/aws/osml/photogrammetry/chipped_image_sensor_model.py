from typing import Any, Dict, List, Optional

import numpy as np

from .coordinates import GeodeticWorldCoordinate, ImageCoordinate
from .elevation_model import ElevationModel
from .sensor_model import SensorModel
from .transforms import ProjectiveTransform


class ChippedImageSensorModel(SensorModel):
    """
    This sensor model should be used when we have pixels for only a portion of the image, but we have a sensor model
    that describes the full image. In this case the image coordinates need to be converted to those of the full
    image before being used by the full sensor model.
    """

    def __init__(
        self,
        original_image_coordinates: List[ImageCoordinate],
        chipped_image_coordinates: List[ImageCoordinate],
        full_image_sensor_model: SensorModel,
    ) -> None:
        """
        Construct a chipped image sensor model given the coordinates of the chip in the original image and the
        coordinates of the chip itself. Normally the chip coordinates will be based on the size of the chipped image
        (i.e. [0,0], [0, height] ...). It is important to make sure the coordinates are in the same order.

        Note that this formulation also allows the chipped images to be scaled differently than the original.

        :param original_image_coordinates: locations in image related to chipped image bounds
        :param chipped_image_coordinates: bounds of the chipped image
        :param full_image_sensor_model: the sensor model for the full image

        :return: None
        """
        super().__init__()
        self.full_image_sensor_model = full_image_sensor_model
        src_coordinates = [image_coordinate.coordinate for image_coordinate in original_image_coordinates]
        dst_coordinates = [image_coordinate.coordinate for image_coordinate in chipped_image_coordinates]
        self.full_to_chip_transform = ProjectiveTransform.estimate(np.vstack(src_coordinates), np.vstack(dst_coordinates))

    def image_to_world(
        self,
        image_coordinate: ImageCoordinate,
        elevation_model: Optional[ElevationModel] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GeodeticWorldCoordinate:
        """
        This function returns the longitude, latitude, elevation world coordinate associated with the x, y coordinate
        of any pixel in the image.

        :param image_coordinate: the x, y image coordinate
        :param elevation_model: an elevation model used to transform the coordinate
        :param options: a dictionary of options that will be passed on to the full image sensor model

        :return: the longitude, latitude, elevation world coordinate
        """
        full_coords = self.full_to_chip_transform.inverse(np.array([image_coordinate.coordinate]))
        full_image_coordinate = ImageCoordinate(full_coords[0])
        return self.full_image_sensor_model.image_to_world(
            full_image_coordinate, elevation_model=elevation_model, options=options
        )

    def world_to_image(self, world_coordinate: GeodeticWorldCoordinate) -> ImageCoordinate:
        """
        This function returns the x, y image coordinate associated with a given longitude, latitude, elevation world
        coordinate.

        :param world_coordinate: the longitude, latitude, elevation world coordinate

        :return: the x, y image coordinate
        """
        full_image_coordinate = self.full_image_sensor_model.world_to_image(world_coordinate)
        chip_coords = self.full_to_chip_transform.forward(np.array([full_image_coordinate.coordinate]))
        chipped_image_coordinate = ImageCoordinate(chip_coords[0])
        return chipped_image_coordinate
