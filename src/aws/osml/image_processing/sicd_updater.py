import logging
from math import floor
from typing import List, Optional, Tuple

from xsdata.formats.dataclass.parsers import XmlParser
from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers.config import SerializerConfig

logger = logging.getLogger(__name__)


class SICDUpdater:
    """
    This class provides a means to perform common updates to a SICD XML metadata document.
    """

    def __init__(self, xml_str: str):
        """
        Construct a new instance of this class to manage a given set of SICD metadata.

        :param xml_str: the SICD XML metadata to update
        """
        self.xml_str = xml_str
        if self.xml_str is not None and len(self.xml_str) > 0:
            parser = XmlParser()
            self.sicd = parser.from_string(self.xml_str)

        # Here we're storing off the original first row/col to support the case where multiple chips are
        # created from a SICD image that has already been chipped.
        self.original_first_row = self.sicd.image_data.first_row
        self.original_first_col = self.sicd.image_data.first_col

    def update_image_data_for_chip(self, chip_bounds: List[int], output_size: Optional[Tuple[int, int]]) -> None:
        """
        This updates the SICD ImageData structure so that the FirstRow, FirstCol and NumRows, NumCols
        elements match the new chip boundary.A sample of this XML structure is shown below::

            <ImageData>
                <NumRows>3859</NumRows>
                <NumCols>6679</NumCols>
                <FirstRow>0</FirstRow>
                <FirstCol>0</FirstCol>
                <FullImage>
                    <NumRows>3859</NumRows>
                    <NumCols>6679</NumCols>
                </FullImage>
            </ImageData>


        :param chip_bounds: the [col, row, width, height] of the chip boundary
        :param output_size: the [width, height] of the output chip
        """

        if output_size is not None and (output_size[0] != chip_bounds[2] or output_size[1] != chip_bounds[3]):
            raise ValueError("SICD chipping does not support scaling operations.")

        self.sicd.image_data.first_row = floor(float(self.original_first_row)) + int(chip_bounds[1])
        self.sicd.image_data.first_col = floor(float(self.original_first_col)) + int(chip_bounds[0])
        self.sicd.image_data.num_rows = int(chip_bounds[3])
        self.sicd.image_data.num_cols = int(chip_bounds[2])

    def encode_current_xml(self) -> str:
        """
        Returns a copy of the current SICD metadata encoded in XML.

        :return: xml encoded SICD metadata
        """
        serializer = XmlSerializer(config=SerializerConfig(pretty_print=False))
        updated_xml = serializer.render(self.sicd)
        return updated_xml
