import logging
from typing import List, Optional
from xml.etree import ElementTree as ET

from aws.osml.photogrammetry import (
    RSMContext,
    RSMGroundDomain,
    RSMGroundDomainForm,
    RSMImageDomain,
    RSMLowOrderPolynomial,
    RSMPolynomial,
    RSMPolynomialSensorModel,
    RSMSectionedPolynomialSensorModel,
    SensorModel,
    WorldCoordinate,
)

from .sensor_model_builder import SensorModelBuilder
from .xmltre_utils import get_tre_field_value, parse_rpc_coefficients


class RSMSensorModelBuilder(SensorModelBuilder):
    """
    This builder is used to create sensor models for images that have RSM TREs. The inputs are not the TREs themselves
    but rather the XML formatted TREs resulting from GDAL's parsing of that information. This information can be
    obtained by accessing the xml:TRE metadata domain for a given GDAL dataset.

    The actual type and number of RSM TREs included with an image will vary depending on the type of RSM sensor model
    defined. In general all images with these sensor models must have an RSMID TRE that defines the overall context
    of the sensor model along with RSMDCA and RSMECA TREs that define the direct and indirect error covariance
    data.

    The polynomial based sensor models will then have at least one RSMPCA TRE and may have multiple. If there are
    multiple then a RSMPIA TRE will also be present to describe how the various polynomial models cover the overall
    image domain.

    The grid based sensor models will have at least one RSMGGA TRE and may have multiple. If there are multiple then
    a RSMGIA TRE will also be present to describe how the various interpolation grids cover the overall image domain.

    It is also possible that adjustment parameters will be in RSMAPA TREs. These adjustments are frequently used to
    modify / update a sensor model after it was originally created.

    See STDI-0002 Volume 1 Appendix U for more detailed information.
    """

    def __init__(self, xml_tres: ET.Element) -> None:
        """
        Constructor for the builder accepting the required XML TREs.

        :param xml_tres: the XML tres for this image

        :return: None
        """
        super().__init__()
        self.xml_tres = xml_tres

    def build(self) -> Optional[SensorModel]:
        """
        Examine the TRE metadata for RSM information, parse the necessary values out of those TREs, and construct a
        RSM sensor model.

        :return: a RSM Polynomial SensorModel if one can be constructed, None otherwise
        """

        # Check to see if an RSMIDA TRE is included with the metadata. This is a mandatory TRE that will be available
        # for all RSM based sensor models. If it is not available then we can stop because the metadata does not
        # contain the basic information necessary to support RSM.
        rsmid_tre = self.xml_tres.find("./tre[@name='RSMIDA']")
        if rsmid_tre is None:
            logging.debug("No RSMID TRE found. Skipping RSM sensor model build.")
            return None

        try:
            # Use the information in the RSMID TRE to build the context. This context contains the ground and image
            # domains that bound the valid regions for the RSM model in this image.
            rsm_context = RSMSensorModelBuilder._build_rsm_context(rsmid_tre)

            # If an RSM model is using polynomial coefficients to define the sensor model then those coefficients
            # will be stored in RSMPC TREs. Note that some images may have several of these TREs defined so we will
            # construct RSMPolynomialSensorModels for every TRE of this kind.
            rsmpc_tres = self.xml_tres.findall("./tre[@name='RSMPCA']")
            rsm_polynomial_sensor_models = [
                RSMSensorModelBuilder._build_rsm_polynomial_sensor_model(rsmpc_tre, rsm_context) for rsmpc_tre in rsmpc_tres
            ]

            # If we only have one RSM polynomial sensor model then it applies to the entire RSM domain. If we have
            # multiple then we are dealing with a sectioned sensor model which will require additional TREs to be
            # parsed.
            if len(rsm_polynomial_sensor_models) > 0:
                if len(rsm_polynomial_sensor_models) == 1:
                    return rsm_polynomial_sensor_models[0]
                else:
                    # Parse RSMPI and construct a segmented polynomial sensor model
                    rsmpi_tre = self.xml_tres.find("./tre[@name='RSMPIA']")
                    if rsmpi_tre is None:
                        logging.warning(
                            "Image has multiple RSMPCA TREs but is missing a RSMPIA that assigns them to sections."
                            "No sensor model can be built! "
                        )
                        return None
                    return RSMSensorModelBuilder._build_rsm_sectioned_polynomial_sensor_model(
                        rsmpi_tre, rsm_context, rsm_polynomial_sensor_models
                    )

            # TODO: Check for RSMGGA and RSMGIA TREs and construct a RSM grid interpolation sensor model
            logging.warning(
                "Image has RSMID TRE but no polynomials. Grid based RSM not implemented so no sensor model returned."
            )
            return None

        except ValueError as ve:
            logging.warning("Unable to parse RSM TREs found in XML metadata. No SensorModel created.")
            logging.warning(str(ve))

            return None

    @staticmethod
    def _build_rsm_ground_domain(rsmid_tre: ET.Element) -> RSMGroundDomain:
        """
        This private method constructs the ground domain from information in the RSMIDA TRE.

        :param rsmid_tre: the GDAL XML for RSMIDA

        :return: the ground domain
        """

        # This is the type of ground domain for this RSM sensor model
        ground_domain_form = RSMGroundDomainForm(get_tre_field_value(rsmid_tre, "GRNDD", str))

        # The valid region of the ground domain is defined by 8 sets of world coordinates (V1 - V8)
        ground_domain_vertices = [
            WorldCoordinate(
                [
                    get_tre_field_value(rsmid_tre, f"V{vertex_number}X", float),
                    get_tre_field_value(rsmid_tre, f"V{vertex_number}Y", float),
                    get_tre_field_value(rsmid_tre, f"V{vertex_number}Z", float),
                ]
            )
            for vertex_number in range(1, 9)
        ]

        # Ground domains are either rectangular or geodetic. Rectangular ground domains have additional values
        # that define a cartesian coordinate system anchored at a point on the earth.
        rectangular_coordinate_origin = None
        rectangular_coordinate_unit_vectors = None
        if ground_domain_form == RSMGroundDomainForm.RECTANGULAR:
            # The world location for the origin (0, 0, 0) of the rectangular coordinate system
            rectangular_coordinate_origin = WorldCoordinate(
                [
                    get_tre_field_value(rsmid_tre, "XUOR", float),
                    get_tre_field_value(rsmid_tre, "YUOR", float),
                    get_tre_field_value(rsmid_tre, "ZUOR", float),
                ]
            )

            # Unit vectors defining the cartesian coordinate system
            rectangular_coordinate_unit_vectors = []
            for row_coefficient in ["XR", "YR", "ZR"]:
                row = [
                    get_tre_field_value(rsmid_tre, f"{col_coefficient}U{row_coefficient}", float)
                    for col_coefficient in ["X", "Y", "Z"]
                ]
                rectangular_coordinate_unit_vectors.append(row)

        try:
            ground_reference_point = WorldCoordinate(
                [
                    get_tre_field_value(rsmid_tre, "GRPX", float),
                    get_tre_field_value(rsmid_tre, "GRPY", float),
                    get_tre_field_value(rsmid_tre, "GRPZ", float),
                ]
            )
        except ValueError:
            # The ground reference point is optional and these elements may be filled with spaces. If we can't parse
            # floating point numbers from these fields we should assume this information has not been provided.
            ground_reference_point = None

        return RSMGroundDomain(
            ground_domain_form,
            ground_domain_vertices,
            rectangular_coordinate_origin=rectangular_coordinate_origin,
            rectangular_coordinate_unit_vectors=rectangular_coordinate_unit_vectors,
            ground_reference_point=ground_reference_point,
        )

    @staticmethod
    def _build_rsm_image_domain(rsmid_tre: ET.Element) -> RSMImageDomain:
        """
        This private method constructs the image domain from information in the RSMIDA TRE.

        :param rsmid_tre: the GDAL XML for RSMIDA
        :return: the image domain
        """
        return RSMImageDomain(
            get_tre_field_value(rsmid_tre, "MINR", int),
            get_tre_field_value(rsmid_tre, "MAXR", int),
            get_tre_field_value(rsmid_tre, "MINC", int),
            get_tre_field_value(rsmid_tre, "MAXC", int),
        )

    @staticmethod
    def _build_rsm_context(rsmid_tre: ET.Element) -> RSMContext:
        """
        This private method constructs a RSM context from information in the RSMIDA TRE.

        :param rsmid_tre: the GDAL XML for RSMIDA

        :return: the RSM context
        """
        return RSMContext(
            RSMSensorModelBuilder._build_rsm_ground_domain(rsmid_tre),
            RSMSensorModelBuilder._build_rsm_image_domain(rsmid_tre),
        )

    @staticmethod
    def _build_rsm_polynomial(rsmpc_tre: ET.Element, polynomial_prefix: str) -> RSMPolynomial:
        """
        This private method constructs an RSM polynomial from a group of related fields in the RSMPCA TRE. These
        TREs have similar fields grouped by the RN, RD, CN, and CD prefixes which correspond to the row or column
        (R or C) numerator or denominator (N or D) identifiers for the polynomial they're associated with.

        :param rsmpc_tre: the GDAL XML for RSMPCA
        :param polynomial_prefix: the prefix identifying the polynomial

        :return: RSMPolynomial = the RSM polynomial
        """
        if polynomial_prefix not in ["RN", "RD", "CN", "CD"]:
            raise ValueError(f"Unexpected prefix {polynomial_prefix}. Expecting RN, RD, CN, or CD")
        return RSMPolynomial(
            get_tre_field_value(rsmpc_tre, f"{polynomial_prefix}PWRX", int),
            get_tre_field_value(rsmpc_tre, f"{polynomial_prefix}PWRY", int),
            get_tre_field_value(rsmpc_tre, f"{polynomial_prefix}PWRZ", int),
            parse_rpc_coefficients(rsmpc_tre, f"{polynomial_prefix}PCF"),
        )

    @staticmethod
    def _build_rsm_polynomial_sensor_model(rsmpc_tre: ET.Element, rsm_context: RSMContext) -> RSMPolynomialSensorModel:
        """
        This private method constructs an RSM polynomial sensor model from an RSMPCA TRE and the context object.

        :param rsmpc_tre: the GDAL XML for RSMPCA
        :param rsm_context: the corresponding RSM context

        :return: the RSM polynomial sensor model
        """
        return RSMPolynomialSensorModel(
            rsm_context,
            get_tre_field_value(rsmpc_tre, "RSN", int),
            get_tre_field_value(rsmpc_tre, "CSN", int),
            get_tre_field_value(rsmpc_tre, "RNRMO", float),
            get_tre_field_value(rsmpc_tre, "CNRMO", float),
            get_tre_field_value(rsmpc_tre, "XNRMO", float),
            get_tre_field_value(rsmpc_tre, "YNRMO", float),
            get_tre_field_value(rsmpc_tre, "ZNRMO", float),
            get_tre_field_value(rsmpc_tre, "RNRMSF", float),
            get_tre_field_value(rsmpc_tre, "CNRMSF", float),
            get_tre_field_value(rsmpc_tre, "XNRMSF", float),
            get_tre_field_value(rsmpc_tre, "YNRMSF", float),
            get_tre_field_value(rsmpc_tre, "ZNRMSF", float),
            RSMSensorModelBuilder._build_rsm_polynomial(rsmpc_tre, "RN"),
            RSMSensorModelBuilder._build_rsm_polynomial(rsmpc_tre, "RD"),
            RSMSensorModelBuilder._build_rsm_polynomial(rsmpc_tre, "CN"),
            RSMSensorModelBuilder._build_rsm_polynomial(rsmpc_tre, "CD"),
        )

    @staticmethod
    def _build_loworder_rsm_polynomial(rsmpi_tre: ET.Element, polynomial_prefix: str) -> RSMLowOrderPolynomial:
        """
        This private method constructs a low order RSM polynomial from a group of related fields in the RSMPIA TRE.
        These TREs have similar fields grouped by the R and C prefixes which correspond to the row or column
        identifiers for the polynomial they're associated with.

        :param rsmpi_tre: the GDAL XML for RSMPIA
        :param polynomial_prefix: the prefix identifying the polynomial

        :return: the low order RSM polynomial
        """
        if polynomial_prefix not in ["R", "C"]:
            raise ValueError(f"Unexpected prefix {polynomial_prefix}. Expecting R or C")

        coefficients = []
        for coeff_suffix in ["0", "X", "Y", "Z", "XX", "XY", "XZ", "YY", "YZ", "ZZ"]:
            coefficients.append(get_tre_field_value(rsmpi_tre, f"{polynomial_prefix}{coeff_suffix}", float))
        return RSMLowOrderPolynomial(coefficients)

    @staticmethod
    def _build_rsm_sectioned_polynomial_sensor_model(
        rsmpi_tre: ET.Element,
        rsm_context: RSMContext,
        rsm_polynomial_sensor_models: List[RSMPolynomialSensorModel],
    ) -> RSMSectionedPolynomialSensorModel:
        """
        This private method constructs an RSM sectioned polynomial sensor model from an RSMPIA TRE, the context object,
        and a collection of RSMPolynomialSensorModels.

        :param rsmpi_tre: the GDAL XML for RSMPCA
        :param rsm_context: the corresponding RSM context


        :return: the RSM polynomial sensor model
        """
        num_section_rows = get_tre_field_value(rsmpi_tre, "RNIS", int)
        num_section_cols = get_tre_field_value(rsmpi_tre, "CNIS", int)
        sensor_model_grid_map = {}
        for sensor_model in rsm_polynomial_sensor_models:
            sensor_model_grid_map[(sensor_model.section_row, sensor_model.section_col)] = sensor_model

        section_sensor_model_grid = []
        for row in range(1, num_section_rows + 1):
            row_of_sensor_models = []
            for col in range(1, num_section_cols + 1):
                row_of_sensor_models.append(sensor_model_grid_map[(row, col)])
            section_sensor_model_grid.append(row_of_sensor_models)

        return RSMSectionedPolynomialSensorModel(
            rsm_context,
            num_section_rows,
            num_section_cols,
            get_tre_field_value(rsmpi_tre, "RSSIZ", float),
            get_tre_field_value(rsmpi_tre, "CSSIZ", float),
            RSMSensorModelBuilder._build_loworder_rsm_polynomial(rsmpi_tre, "R"),
            RSMSensorModelBuilder._build_loworder_rsm_polynomial(rsmpi_tre, "C"),
            section_sensor_model_grid,
        )
