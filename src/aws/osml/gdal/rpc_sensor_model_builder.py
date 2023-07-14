import logging
from typing import Optional
from xml.etree import ElementTree as ET

from aws.osml.photogrammetry import RPCPolynomial, RPCSensorModel

from .sensor_model_builder import SensorModelBuilder
from .xmltre_utils import get_tre_field_value, parse_rpc_coefficients


class RPCSensorModelBuilder(SensorModelBuilder):
    """
    This builder is used to create sensor models for images that have RPC TREs. The inputs are not the TREs themselves
    but rather the XML formatted TREs resulting from GDAL's parsing of that information. This information can be
    obtained by accessing the xml:TRE metadata domain for a given GDAL dataset.

    This builder only supports the RPC00B format of this metadata information. Support for other TREs can be added
    in the future if we find ourselves working with imagery containing that metadata.

     See STDI-0002 Volume 1 Appendix E for more detailed information.
    """

    def __init__(self, xml_tres: ET.Element) -> None:
        """
        Constructor for the builder accepting the required XML TREs.

        :param xml_tres: the XML tres for this image

        :return: None
        """
        super().__init__()
        self.xml_tres = xml_tres

    def build(self) -> Optional[RPCSensorModel]:
        """
        Examine the TRE metadata for RPC information, parse the necessary values out of those TREs, and construct a
        RPC sensor model.

        :return: a RPC SensorModel if one can be constructed, None otherwise
        """

        # Check to see if an RPC00B TRE is included with the metadata. It would contain the necessary parameters for
        # this type of sensor model.
        # TODO: Consider expanding support to include RPC00A which is defined in STDI-0001
        rpc_tre = self.xml_tres.find("./tre[@name='RPC00B']")
        if rpc_tre is None:
            logging.debug("No RPC00B TRE found. Skipping RPC sensor model build.")
            return None

        # Attempt to construct the RPC camera model from the metadata provided
        try:
            success = get_tre_field_value(rpc_tre, "SUCCESS", int)
            if success != 1:
                logging.info("RPC00B TRE SUCCESS field was not '1'. Skipping RPC sensor model build.")
                return None

            return RPCSensorModelBuilder.build_rpc_sensor_model(rpc_tre)

        except ValueError as ve:
            logging.warning("Unable to parse RPC00B TRE found in XML metadata. No SensorModel created.")
            logging.warning(str(ve))

            return None

    @staticmethod
    def build_rpc_sensor_model(rpc_tre: ET.Element) -> RPCSensorModel:
        """
        This private method constructs an RPC sensor model from an RPC00B TRE.

        :param rpc_tre: the GDAL XML for RPC00B

        :return: the RPC sensor model
        """
        return RPCSensorModel(
            get_tre_field_value(rpc_tre, "ERR_BIAS", float),
            get_tre_field_value(rpc_tre, "ERR_RAND", float),
            get_tre_field_value(rpc_tre, "LINE_OFF", float),
            get_tre_field_value(rpc_tre, "SAMP_OFF", float),
            get_tre_field_value(rpc_tre, "LAT_OFF", float),
            get_tre_field_value(rpc_tre, "LONG_OFF", float),
            get_tre_field_value(rpc_tre, "HEIGHT_OFF", float),
            get_tre_field_value(rpc_tre, "LINE_SCALE", float),
            get_tre_field_value(rpc_tre, "SAMP_SCALE", float),
            get_tre_field_value(rpc_tre, "LAT_SCALE", float),
            get_tre_field_value(rpc_tre, "LONG_SCALE", float),
            get_tre_field_value(rpc_tre, "HEIGHT_SCALE", float),
            RPCSensorModelBuilder.build_rpc_polynomial(rpc_tre, "LINE_NUM_COEFF"),
            RPCSensorModelBuilder.build_rpc_polynomial(rpc_tre, "LINE_DEN_COEFF"),
            RPCSensorModelBuilder.build_rpc_polynomial(rpc_tre, "SAMP_NUM_COEFF"),
            RPCSensorModelBuilder.build_rpc_polynomial(rpc_tre, "SAMP_DEN_COEFF"),
        )

    @staticmethod
    def build_rpc_polynomial(rpc_tre: ET.Element, polynomial_name: str) -> RPCPolynomial:
        """
        This private method constructs a RPC polynomial from coefficients found inthe RPC00B TRE. There are 4
        repeating groups of these coefficients for the polynomials associated with line or sample numerators and
        denominators.

        :param rpc_tre: the GDAL XML for RPC00B
        :param polynomial_name: the name of the polynomial to build

        :return: the RPC polynomial
        """
        return RPCPolynomial(parse_rpc_coefficients(rpc_tre, polynomial_name))
