import unittest
from pathlib import Path

from xsdata.formats.dataclass.parsers import XmlParser

import aws.osml.formats.sidd.models.sidd_v2_0_0 as sidd2


class TestSIDDFormats(unittest.TestCase):
    def test_sidd_20(self):
        sidd = XmlParser().from_path(Path("./test/data/sidd/example.sidd.xml"))

        assert isinstance(sidd, sidd2.SIDD)
