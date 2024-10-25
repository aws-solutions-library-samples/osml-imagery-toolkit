#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import base64
import re
from io import StringIO
from typing import Callable, List, Optional, TypeVar
from xml.etree import ElementTree as ET

from defusedxml import ElementTree

# This is a type placeholder needed by the _get_tre_field_value() type hints
T = TypeVar("T")


class NITFDESAccessor:
    """
    This class is a facade that makes it easier to work with the XML formatted Data Extension Segments parsed by
    GDAL.
    """

    def __init__(self, gdal_xml_des_metadata: List[str]):
        """
        Construct the DES accessor given the contents of the xml:DES metadata domain from GDAL.
        ``
            raster_dataset = gdal.Open(path_to_nitf_image)
            xml_des = raster_dataset.GetMetadata("xml:DES")
            des_accessor = NITFDESAccessor(xml_des)
        ``

        :param gdal_xml_des_metadata: the GDAL parsed DES metadata
        """
        self.parsed_des_lists = []
        if gdal_xml_des_metadata is not None and len(gdal_xml_des_metadata) > 0:
            for xml_des_list in gdal_xml_des_metadata:
                # The new handling GDAL has for XML data content causes an XML document to be expanded in the middle
                # of the xml:DES data structure. An embedded xml prolog (e.g. <?xml version= ... ?>) is invalid syntax
                # that will throw off some XML parsers. The xml prolog is optional, so we can strip all of them from
                # the XML as a workaround while we look for a better way to address this recent GDAL change.
                clean_xml_string = re.sub(r"<\?xml.*?\?>", "", xml_des_list).strip()
                des_list = ElementTree.fromstring(clean_xml_string)
                self.parsed_des_lists.append(des_list)

    def get_segments_by_name(self, des_name: str) -> List[ET.Element]:
        """
        This method searches through the GDAL xml:DES metadata and returns the XML structure for any segments
        matching the provided name. This is equivalent to retrieving all segments that have a matching DESID.

        :param des_name: the DESID (e.g. XML_DATA_CONTENT)
        :return: the list of segments, multiple items in the list will occur if the NITF has multiple matching segments
        """
        result = []
        for des_list in self.parsed_des_lists:
            result.extend(des_list.findall(f"./des[@name='{des_name}']"))
        return result

    @staticmethod
    def extract_des_header(des_element: ET.Element) -> str:
        """
        This function encodes the existing values from the Data Extension Segment header into a fields appropriately
        sized for including in a NITF image. The DESDATA field is not copied because the assumption is that the data
        itself will be updated. The DE and DESID fields are not included either because GDAL adds them itself
        when writing the segment.

        :param des_element: the Data Extension Segment containing current segment
        :return: the encoded DESVER through DESSHF fields.
        """
        result_builder = StringIO()
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DESVER", int), "02d"))
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DECLAS", str), ">1"))
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DESCLSY", str), ">2"))
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DESCODE", str), ">11"))
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DESCTLH", str), ">2"))
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DESREL", str), ">20"))
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DESDCTP", str), ">2"))
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DESDCDT", str), ">8"))
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DESDCXM", str), ">4"))
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DESDG", str), ">1"))
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DESDGDT", str), ">8"))
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DESCLTX", str), ">43"))
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DESCATP", str), ">1"))
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DESCAUT", str), ">40"))
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DESCRSN", str), ">1"))
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DESSRDT", str), ">8"))
        result_builder.write(format(NITFDESAccessor.parse_field_value(des_element, "DESCTLN", str), ">15"))

        # TODO: If DESID = TRE_OVERFLOW DESOFLW, DESITEM

        subheader_length = NITFDESAccessor.parse_field_value(des_element, "DESSHL", int)
        result_builder.write(format(subheader_length, "04d"))
        if subheader_length > 0:
            result_builder.write(
                format(
                    NITFDESAccessor.parse_field_value(des_element, "DESSHF", str),
                    f">{subheader_length}",
                )
            )

        return result_builder.getvalue()

    @staticmethod
    def extract_desdata_xml(des_element: ET.Element) -> Optional[str]:
        """
        This function attempts to extract a block of XML from the field element named DESDATA. Versions of GDAL
        before 3.9 returned the XML data base64 encoded as a value attribute. Versions >=3.9 are automatically
        expanding the xml into the text area of an <xml_content> element.

        :param des_element: the root xml:DES metadata element
        :return: the xml content if it is found and can be extracted
        """
        desdata_element = des_element.find("./field[@name='DESDATA']")
        if desdata_element is None:
            return None

        value_attribute = desdata_element.get("value")
        if value_attribute:
            # This appears to be the encoding used by GDAL versions <3.9. Extract the
            # XML from the base64 encoded value attribute
            xml_bytes = base64.b64decode(value_attribute)
            return xml_bytes.decode("utf-8")

        xml_content_element = desdata_element.find("xml_content")
        if xml_content_element:
            # This appears to be a encoding used by GDAL >3.9. The XML is already parsed
            # and available as the content of this element. See: https://github.com/OSGeo/gdal/pull/8953
            return ET.tostring(xml_content_element[0], "unicode")

        # Unable to parse the XML from the data segment. This sometimes happens if GDAL
        # changes the representation of this information in their APIs
        return None

    @staticmethod
    def parse_field_value(des_element: ET.Element, field_name: str, type_conversion: Callable[[str], T]) -> T:
        """
        This is method will find a named "field" element in the children of a TRE Element and
        return the "value" attribute of that named field. A type conversion function can be provided to convert the
        attribute value to a specific Python type (e.g. int, float, or str)

        :param des_element: the root element to find the named field in
        :param field_name: the name of the field element
        :param type_conversion: the desired type of the output, must support construction from a string

        :return: the value converted to the requested type
        """
        field_element = des_element.find(f"./field[@name='{field_name}']")
        if field_element is None:
            raise ValueError(f"Unable to find TRE field named {field_name}")
        str_value = field_element.get("value")
        if str_value is None:
            raise ValueError(f"Field {field_name} does not have a value attribute.")

        return type_conversion(str_value)
