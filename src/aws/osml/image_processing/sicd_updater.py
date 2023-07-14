import logging
import re
from typing import Callable, List, TypeVar
from xml.etree import ElementTree as ET

from defusedxml import ElementTree

logger = logging.getLogger(__name__)

# This is a type placeholder needed by the parse_element_text() type hints
T = TypeVar("T")


class SICDUpdater:
    """
    This class provides a means to perform common updates to a SICD XML metadata document.
    """

    def __init__(self, sicd_element: ET.Element):
        """
        Construct a new instance of this class to manage a given set of SICD metadata.

        :param sicd_element: the SICD XML metadata to update
        """
        self.sicd_element = sicd_element

        # Extract the XML namespace from the root SICD element and store it for later use in element queries
        namespace_match = re.match(r"{.*}", self.sicd_element.tag)
        self.namespace = namespace_match.group(0) if namespace_match else ""

        # We don't currently have many examples of SICD data. An attempt has been made to make this code
        # work so long as the portions of the XML schema we depend upon don't change. This warning is just
        # an attempt to provide diagnostic information incase future datasets don't work.
        if self.namespace != "{urn:SICD:1.2.1}":
            logger.warning(f"Attempting to process SICD metadata with an untested namespace {self.namespace}")

        # Here we're storing off the original first row/col to support the case where multiple chips are
        # created from a SICD image that has already been chipped.
        self.original_first_row = self.parse_element_text(".//{0}FirstRow".format(self.namespace), int)
        self.original_first_col = self.parse_element_text(".//{0}FirstCol".format(self.namespace), int)

    def update_image_data_for_chip(self, chip_bounds: List[int]) -> None:
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
        """
        first_col_element = self.sicd_element.find(".//{0}FirstCol".format(self.namespace))
        first_row_element = self.sicd_element.find(".//{0}FirstRow".format(self.namespace))
        num_cols_element = self.sicd_element.find(".//{0}NumCols".format(self.namespace))
        num_rows_element = self.sicd_element.find(".//{0}NumRows".format(self.namespace))
        if first_row_element is None or first_col_element is None or num_cols_element is None or num_rows_element is None:
            logger.warning("SICD ImageData structures were not found. No updates applied.")
            return

        first_col_element.text = str(self.original_first_col + chip_bounds[0])
        first_row_element.text = str(self.original_first_row + chip_bounds[1])
        num_cols_element.text = str(chip_bounds[2])
        num_rows_element.text = str(chip_bounds[3])

        if logger.isEnabledFor(logging.DEBUG):
            image_data_element = self.sicd_element.find(".//{0}ImageData".format(self.namespace))
            if image_data_element is not None:
                logger.debug("Updated SICD ImageData element for chip:")
                logger.debug(
                    ElementTree.tostring(
                        image_data_element,
                        encoding="unicode",
                    )
                )

    def encode_current_xml(self) -> str:
        """
        Returns a copy of the current SICD metadata encoded in XML.

        :return: xml encoded SICD metadata
        """
        return ElementTree.tostring(self.sicd_element, encoding="unicode")

    def parse_element_text(self, element_xpath: str, type_conversion: Callable[[str], T]) -> T:
        """
        This function finds the first element matching the provided xPath and then runs the text of that element
        through the provided conversion function.

        :param element_xpath: the xPath of the element
        :param type_conversion: the desired type of the output, must support construction from a string
        :return: the element text converted to the requested type
        """
        field_element = self.sicd_element.find(element_xpath)
        if field_element is None:
            raise ValueError(f"Unable to find element {element_xpath}")
        str_value = field_element.text
        if str_value is None:
            raise ValueError(f"Element {element_xpath} does not have text.")
        return type_conversion(str_value)
