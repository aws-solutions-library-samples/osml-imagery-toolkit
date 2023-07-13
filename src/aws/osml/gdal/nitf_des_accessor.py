from io import StringIO
from typing import Callable, List, TypeVar
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
                des_list = ElementTree.fromstring(xml_des_list)
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
