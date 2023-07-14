import logging
from math import radians
from typing import Optional
from xml.etree import ElementTree as ET

from aws.osml.photogrammetry import GeodeticWorldCoordinate, ImageCoordinate, ProjectiveSensorModel

from .sensor_model_builder import SensorModelBuilder
from .xmltre_utils import get_tre_field_value


class ProjectiveSensorModelBuilder(SensorModelBuilder):
    """
    This builder is used to create sensor models for images that have an IGEOLO or CSCRNA TREs. The inputs are not the
    TREs themselves but rather the XML formatted TREs resulting from GDAL's parsing of that information. This
    information can be obtained by accessing the xml:TRE metadata domain for a given GDAL dataset.

    See STDI-0002 Volume 3 Appendix B for more detailed information.
    """

    def __init__(self, xml_tres: ET.Element, full_image_width: float, full_image_height: float) -> None:
        """
        Constructor for the builder accepting the required XML TREs and image dimensions.

        :param xml_tres: the XML tres for this image
        :param full_image_width: the width of the image in pixels
        :param full_image_height: the height of the image in pixels

        :return: None
        """
        super().__init__()
        self.xml_tres = xml_tres
        self.full_image_width = full_image_width
        self.full_image_height = full_image_height

    def build(self) -> Optional[ProjectiveSensorModel]:
        """
        Examine the TRE metadata for corner coordinate information, parse the necessary values out of those TREs,
        and construct a projective sensor model.

        :return: a ProjectiveSensorModel if one can be constructed, None otherwise
        """

        # Check to see if an CSCRNA TRE is included with the metadata. It would contain the necessary parameters for
        # this type of sensor model.
        cscrna_tre = self.xml_tres.find("./tre[@name='CSCRNA']")
        if cscrna_tre is None:
            logging.debug("No CSCRNA TRE found.")
            return None

        return ProjectiveSensorModelBuilder.build_projective_sensor_model(
            cscrna_tre, self.full_image_width, self.full_image_height
        )

    @staticmethod
    def build_projective_sensor_model(
        cscrna_tre: ET.Element, full_image_width: float, full_image_height: float
    ) -> ProjectiveSensorModel:
        """
        This private method constructs an RPC sensor model from an RPC00B TRE.

        :param cscrna_tre: the GDAL XML for CSCRNA
        :param full_image_width: the width of the image in pixels
        :param full_image_height: the height of the image in pixels

        :return: the projective sensor model
        """

        world_coordinates = [
            GeodeticWorldCoordinate(
                [
                    radians(get_tre_field_value(cscrna_tre, f"{corner}CNR_LONG", float)),
                    radians(get_tre_field_value(cscrna_tre, f"{corner}CNR_LAT", float)),
                    radians(get_tre_field_value(cscrna_tre, f"{corner}CNR_HT", float)),
                ]
            )
            for corner in ["UL", "UR", "LR", "LL"]
        ]
        image_coordinates = [
            ImageCoordinate([0, 0]),
            ImageCoordinate([full_image_width, 0]),
            ImageCoordinate([full_image_width, full_image_height]),
            ImageCoordinate([0, full_image_height]),
        ]
        return ProjectiveSensorModel(world_coordinates, image_coordinates)
