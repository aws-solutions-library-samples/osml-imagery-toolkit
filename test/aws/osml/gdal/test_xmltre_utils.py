from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pytest
from defusedxml import ElementTree

from configuration import TEST_ENV_CONFIG


@patch.dict("os.environ", TEST_ENV_CONFIG, clear=True)
class TestXMLTREUtils(TestCase):
    def setUp(self):
        """
        Set up virtual DDB resources/tables for each test to use
        """
        self.sample_metadata_ms_rpc00b = self.build_metadata_ms_rpc00b()

    def test_get_tre_field_value_nominal(self):
        from aws.osml.gdal.xmltre_utils import get_tre_field_value

        piaimc_tre = self.sample_metadata_ms_rpc00b.find("./tre[@name='PIAIMC']")
        assert piaimc_tre is not None

        assert get_tre_field_value(piaimc_tre, "SENSMODE", str) == "PUSHBROOM"
        assert get_tre_field_value(piaimc_tre, "GENERATION", int) == 1
        assert get_tre_field_value(piaimc_tre, "MEANGSD", float) == 86.1

    def test_parse_rpc_coefficients(self):
        from aws.osml.gdal.xmltre_utils import parse_rpc_coefficients

        rpc_tre = self.sample_metadata_ms_rpc00b.find("./tre[@name='RPC00B']")
        assert rpc_tre is not None

        rpc_coefficients = parse_rpc_coefficients(rpc_tre, "LINE_NUM_COEFF")
        assert np.allclose(
            np.array(rpc_coefficients),
            np.array(
                [
                    -0.01219784,
                    -0.177912,
                    -1.197441,
                    -0.01962294,
                    -0.002400754,
                    -0.0001266875,
                    -0.0002864113,
                    -0.0009364538,
                    0.009023196,
                    -6.372315e-06,
                    -3.353148e-06,
                    -9.19286e-06,
                    1.11425e-06,
                    -9.60523e-06,
                    -4.486697e-05,
                    -0.0002875052,
                    -6.413221e-05,
                    -1.735976e-06,
                    -2.999621e-08,
                    -1.049317e-06,
                ]
            ),
        )

    def test_get_tre_field_value_invalid(self):
        from aws.osml.gdal.xmltre_utils import get_tre_field_value

        piaimc_tre = self.sample_metadata_ms_rpc00b.find("./tre[@name='PIAIMC']")
        with pytest.raises(ValueError) as value_error:
            get_tre_field_value(piaimc_tre, "SENSMODE_IN", int)
        assert "Unable to find TRE field named" in str(value_error.value)

        # updating the value to None
        rpt_side = piaimc_tre.find("./field[@name='SUBQUAL']")
        rpt_side.set("value", None)

        with pytest.raises(ValueError) as value_error:
            get_tre_field_value(piaimc_tre, "SUBQUAL", str)
        assert "does not have a value attribute" in str(value_error.value)

    def test_parse_rpc_coefficents_invalid_matching(self):
        from aws.osml.gdal.xmltre_utils import parse_rpc_coefficients

        rpc_tre = self.sample_metadata_ms_rpc00b.find("./tre[@name='RPC00B']")
        with pytest.raises(ValueError) as value_error:
            parse_rpc_coefficients(rpc_tre, "LINE_NUM_COEFF_INVALID")
        assert "does not contain a repeated element named" in str(value_error.value)

    def test_parse_rpc_coefficents_invalid_numbers(self):
        from aws.osml.gdal.xmltre_utils import parse_rpc_coefficients

        rpc_tre = self.sample_metadata_ms_rpc00b.find("./tre[@name='RPC00B']")

        repeated = rpc_tre.find("./repeated[@name='LINE_NUM_COEFF']")
        repeated.set("number", None)

        with pytest.raises(ValueError) as value_error:
            parse_rpc_coefficients(rpc_tre, "LINE_NUM_COEFF")
        assert "Repeated tag in XML TRE is missing required number attribute" in str(value_error.value)

    def test_parse_rpc_coefficents_invalid_group(self):
        from aws.osml.gdal.xmltre_utils import parse_rpc_coefficients

        rpc_tre = self.sample_metadata_ms_rpc00b.find("./tre[@name='RPC00B']")

        repeated = rpc_tre.find("./repeated[@name='LINE_NUM_COEFF']")
        repeated[0].set("index", None)

        with pytest.raises(ValueError) as value_error:
            parse_rpc_coefficients(rpc_tre, "LINE_NUM_COEFF")
        assert "Repeated group in XML TRE is missing required index attribute" in str(value_error.value)

        # bring back to default but set the value to None
        repeated[0].set("index", "0")
        group_index = repeated[0]
        group_index[0].set("value", None)

        with pytest.raises(ValueError) as value_error:
            parse_rpc_coefficients(rpc_tre, "LINE_NUM_COEFF")
        assert "Field in repeated group in XML TRE is missing required value attribute" in str(value_error.value)

    @staticmethod
    def build_metadata_ms_rpc00b():
        with open("test/data/sample-metadata-ms-rpc00b.xml", "rb") as xml_file:
            xml_tres = ElementTree.parse(xml_file)
            return xml_tres
