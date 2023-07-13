from typing import Callable, List, TypeVar
from xml.etree import ElementTree as ET

# This is a type placeholder needed by the _get_tre_field_value() type hints
T = TypeVar("T")


def get_tre_field_value(tre: ET.Element, field_name: str, type_conversion: Callable[[str], T]) -> T:
    """
    This is a private utility function that will find a named "field" element in the children of a TRE Element and
    return the "value" attribute of that named field. A type conversion function can be provided to convert the
    attribute value to a specific Python type (e.g. int, float, or str)

    :param tre: the root element to find the named field in
    :param field_name: the name of the field element
    :param type_conversion: the desired type of the output, must support construction from a string

    :return: T = the value converted to the requested type
    """
    field_element = tre.find(f"./field[@name='{field_name}']")
    if field_element is None:
        raise ValueError(f"Unable to find TRE field named {field_name} in {tre.tag}")
    str_value = field_element.get("value")
    if str_value is None:
        raise ValueError(f"Field {field_name} does not have a value attribute.")
    return type_conversion(str_value)


def parse_rpc_coefficients(tre: ET.Element, repeated_name: str) -> List[float]:
    """
    This private utility function parses RPC coefficients from the child elements of a <repeated ...> tag in the
    XML TREs.

    :param tre: XML document
    :param repeated_name: find a specific name in the XML tree

    :return: a list of floating point values for the coefficients
    """

    repeated = tre.find(f"./repeated[@name='{repeated_name}']")
    if repeated is None:
        raise ValueError(f"XML TRE does not contain a repeated element named {repeated_name}")

    num_coefficients_str = repeated.get("number")
    if num_coefficients_str is None:
        raise ValueError("Invalid XML TRE: Repeated tag in XML TRE is missing required number attribute")
    num_coefficients = int(num_coefficients_str)

    result: List[float] = [0.0] * num_coefficients
    for group in repeated:
        index_str = group.get("index")
        if index_str is None:
            raise ValueError("Invalid XML TRE: Repeated group in XML TRE is missing required index attribute")
        index = int(index_str)

        if group[0] is None:
            raise ValueError("Invalid XML TRE: Repeated group in XML TRE is missing required field child element")

        value_str = group[0].get("value")
        if value_str is None:
            raise ValueError("Invalid XML TRE: Field in repeated group in XML TRE is missing required value attribute")
        value = float(value_str)
        result[index] = value

    return result
