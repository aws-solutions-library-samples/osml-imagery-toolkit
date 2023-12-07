import logging
from typing import List, Optional, Tuple

from xsdata.formats.dataclass.parsers import XmlParser
from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers.config import SerializerConfig

import aws.osml.formats.sidd.models.sidd_v1_0_0 as sidd100
import aws.osml.formats.sidd.models.sidd_v2_0_0 as sidd200
import aws.osml.formats.sidd.models.sidd_v3_0_0 as sidd300

logger = logging.getLogger(__name__)


class SIDDUpdater:
    def __init__(self, xml_str: str):
        """
        Construct a new instance of this class to manage a given set of SIDD metadata.

        :param xml_str: the SIDD XML metadata to update
        """
        self.xml_str = xml_str
        if self.xml_str is not None and len(self.xml_str) > 0:
            parser = XmlParser()
            self.sidd = parser.from_string(self.xml_str)

    def update_image_data_for_chip(self, chip_bounds: List[int], output_size: Optional[Tuple[int, int]]) -> None:
        """
        This adds or updates the SIDD GeometricChip structure so that the ChipSize and original corner coordinates
        are recorded. A sample of this XML structure is shown below:

                <GeometricChip>
                  <ChipSize>
                    <Row xmlns:ns1="urn:SICommon:1.0">512</Row>
                    <Col xmlns:ns1="urn:SICommon:1.0">512</Col>
                  </ChipSize>
                  <OriginalUpperLeftCoordinate>
                    <Row xmlns:ns1="urn:SICommon:1.0">7408</Row>
                    <Col xmlns:ns1="urn:SICommon:1.0">7407</Col>
                  </OriginalUpperLeftCoordinate>
                  <OriginalUpperRightCoordinate>
                    <Row xmlns:ns1="urn:SICommon:1.0">7408</Row>
                    <Col xmlns:ns1="urn:SICommon:1.0">7919</Col>
                  </OriginalUpperRightCoordinate>
                  <OriginalLowerLeftCoordinate>
                    <Row xmlns:ns1="urn:SICommon:1.0">7920</Row>
                    <Col xmlns:ns1="urn:SICommon:1.0">7407</Col>
                  </OriginalLowerLeftCoordinate>
                  <OriginalLowerRightCoordinate>
                    <Row xmlns:ns1="urn:SICommon:1.0">7920</Row>
                    <Col xmlns:ns1="urn:SICommon:1.0">7919</Col>
                  </OriginalLowerRightCoordinate>
                </GeometricChip>


        :param chip_bounds: the [col, row, width, height] of the chip boundary
        :param output_size: the [width, height] of the output chip if different from the chip boundary
        """
        if not output_size:
            output_size = chip_bounds[2], chip_bounds[3]

        # The xsdata code generators produced different types for each version of the SIDD specification.
        # in this case the types are all equivalent so the logic isn't different but this piece of code
        # ensures we're constructing the correct type from the right version of SIDD constructs.
        if isinstance(self.sidd, sidd100.SIDD):
            sidd_namespace = sidd100
        elif isinstance(self.sidd, sidd200.SIDD):
            sidd_namespace = sidd200
        elif isinstance(self.sidd, sidd300.SIDD):
            sidd_namespace = sidd300
        else:
            logger.warning("sidd_updater.py has not been updated to support a new SIDD version. Defaulting to 3.0")
            sidd_namespace = sidd300

        # The DownstreamReprocessing element is optional so if it is not set create it first.
        if not self.sidd.downstream_reprocessing:
            self.sidd.downstream_reprocessing = sidd_namespace.DownstreamReprocessingType()

        # Identify the location of the UL, UR, LR, LL corners of this chip in the full image. If the image is already
        # a chip of a full image these coordinates need to be updated, so they are still the positions of the new chip
        # in the original full image.
        full_image_chip_corners = [
            (chip_bounds[0], chip_bounds[1]),
            (chip_bounds[0] + chip_bounds[2] - 1, chip_bounds[1]),
            (chip_bounds[0] + chip_bounds[2] - 1, chip_bounds[1] + chip_bounds[3] - 1),
            (chip_bounds[0], chip_bounds[1] + chip_bounds[3] - 1),
        ]
        if self.sidd.downstream_reprocessing.geometric_chip:
            original_chip_size = (
                self.sidd.downstream_reprocessing.geometric_chip.chip_size.col,
                self.sidd.downstream_reprocessing.geometric_chip.chip_size.row,
            )
            original_corners = [
                (
                    self.sidd.downstream_reprocessing.geometric_chip.original_upper_left_coordinate.col,
                    self.sidd.downstream_reprocessing.geometric_chip.original_upper_left_coordinate.row,
                ),
                (
                    self.sidd.downstream_reprocessing.geometric_chip.original_upper_right_coordinate.col,
                    self.sidd.downstream_reprocessing.geometric_chip.original_upper_right_coordinate.row,
                ),
                (
                    self.sidd.downstream_reprocessing.geometric_chip.original_lower_right_coordinate.col,
                    self.sidd.downstream_reprocessing.geometric_chip.original_lower_right_coordinate.row,
                ),
                (
                    self.sidd.downstream_reprocessing.geometric_chip.original_lower_left_coordinate.col,
                    self.sidd.downstream_reprocessing.geometric_chip.original_lower_left_coordinate.row,
                ),
            ]

            full_image_chip_corners = [
                SIDDUpdater.chipped_coordinate_to_full(corner, original_chip_size, original_corners)
                for corner in full_image_chip_corners
            ]

        # Create the new DownstreamReprocessing.GeometricChip element that contains the information needed to
        # relate this chip to the original full image.
        self.sidd.downstream_reprocessing.geometric_chip = sidd_namespace.GeometricChipType(
            chip_size=sidd_namespace.RowColIntType(row=output_size[1], col=output_size[0]),
            original_upper_left_coordinate=sidd_namespace.RowColDoubleType(
                row=full_image_chip_corners[0][1], col=full_image_chip_corners[0][0]
            ),
            original_upper_right_coordinate=sidd_namespace.RowColDoubleType(
                row=full_image_chip_corners[1][1], col=full_image_chip_corners[1][0]
            ),
            original_lower_left_coordinate=sidd_namespace.RowColDoubleType(
                row=full_image_chip_corners[3][1], col=full_image_chip_corners[3][0]
            ),
            original_lower_right_coordinate=sidd_namespace.RowColDoubleType(
                row=full_image_chip_corners[2][1], col=full_image_chip_corners[2][0]
            ),
        )

    def encode_current_xml(self) -> str:
        """
        Returns a copy of the current SIDD metadata encoded in XML.

        :return: xml encoded SIDD metadata
        """
        serializer = XmlSerializer(config=SerializerConfig(pretty_print=False))
        updated_xml = serializer.render(self.sidd)
        return updated_xml

    @staticmethod
    def chipped_coordinate_to_full(
        chip_coordinate: Tuple[float, float],
        chip_size: Tuple[int, int],
        original_corner_coordinates: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """
        This function converts pixel locations in a chip to the pixel locations in a full image using a bi-linear
        interpolation method described in section 5.1.1 of the Sensor Independent Derived Data (SIDD) specification
        v3.0 Volume 1.

        :param chip_coordinate: the [x, y] coordinate of the pixel in the chip
        :param chip_size: the size of the chip [width, height]
        :param original_corner_coordinates: the [x, y] location of the UL, UR, LR, LL corners in the original image
        :return: the [x, y] coordinate of the pixel in the original image
        """
        # Step 1: Normalize the chip coordinates
        u = chip_coordinate[1] / (chip_size[1] - 1)
        v = chip_coordinate[0] / (chip_size[0] - 1)

        # Step 2: Compute original full image row coordinate bi-linear coefficients
        a_r = original_corner_coordinates[0][1]
        b_r = original_corner_coordinates[3][1] - original_corner_coordinates[0][1]
        d_r = original_corner_coordinates[1][1] - original_corner_coordinates[0][1]
        f_r = (
            original_corner_coordinates[0][1]
            + original_corner_coordinates[2][1]
            - original_corner_coordinates[1][1]
            - original_corner_coordinates[3][1]
        )

        # Step 3: Compute original full image column coordinate bi-linear coefficients
        a_c = original_corner_coordinates[0][0]
        b_c = original_corner_coordinates[3][0] - original_corner_coordinates[0][0]
        d_c = original_corner_coordinates[1][0] - original_corner_coordinates[0][0]
        f_c = (
            original_corner_coordinates[0][0]
            + original_corner_coordinates[2][0]
            - original_corner_coordinates[1][0]
            - original_corner_coordinates[3][0]
        )

        # Step 4: Compute the full image row and column coordinate
        r = a_r + u * b_r + v * d_r + u * v * f_r
        c = a_c + u * b_c + v * d_c + u * v * f_c

        return c, r
