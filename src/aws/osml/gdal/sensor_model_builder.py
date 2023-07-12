from abc import ABC, abstractmethod
from typing import Optional

from aws.osml.photogrammetry import SensorModel


class SensorModelBuilder(ABC):
    """
    This is an abstract base for all classes used to construct SensorModels from various types of metadata.
    """

    def __init__(self) -> None:
        """
        Constructor for the builder accepting various of required properties or format

        :return: None
        """
        pass

    @abstractmethod
    def build(self) -> Optional[SensorModel]:
        """
        Constructs the sensor model from the available information. Note that in cases where not enough information is
        available to provide any solution this method will return None.

        :return: the sensor model if available in the metadata provided
        """
