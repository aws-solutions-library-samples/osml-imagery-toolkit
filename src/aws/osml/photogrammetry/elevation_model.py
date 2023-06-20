from abc import ABC, abstractmethod

from .coordinates import GeodeticWorldCoordinate


class ElevationModel(ABC):
    """
    An elevation model associates a height z for a given x, y of a world coordinate. It typically provides information
    about the terrain associated with longitude, latitude locations of an ellipsoid, but it can also be used to model
    surfaces for other ground domains.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def set_elevation(self, world_coordinate: GeodeticWorldCoordinate) -> None:
        """
        This method updates the elevation component of a world coordinate to match the surface elevation at
        longitude, latitude.

        :param world_coordinate: the coordinate to update

        :return: None
        """


class ConstantElevationModel(ElevationModel):
    """
    A constant elevation model with a single value for all longitude, latitude.
    """

    def __init__(self, constant_elevation: float) -> None:
        """
        Constructs the constant elevation model.

        :param constant_elevation: the elevation value for all longitude, latitude

        :return: None
        """
        super().__init__()
        self.constant_elevation = constant_elevation

    def set_elevation(self, world_coordinate: GeodeticWorldCoordinate) -> None:
        """
        Updates world coordinate's elevation to match the constant elevation.

        :param world_coordinate: the coordinate to update

        :return: None
        """
        world_coordinate.elevation = self.constant_elevation
