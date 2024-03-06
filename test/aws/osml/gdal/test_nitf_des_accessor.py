#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest
from unittest import TestCase
from xml.etree import ElementTree as ET

from osgeo import gdal

from aws.osml.gdal import NITFDESAccessor


class TestNITFDESAccessor(TestCase):
    def test_get_segments_by_name(self):
        ds = gdal.Open("./test/data/sicd_example_1_PFA_RE32F_IM32F_HH-0-0.NITF")
        des_accessor = NITFDESAccessor(ds.GetMetadata("xml:DES"))
        xml_segments = des_accessor.get_segments_by_name("XML_DATA_CONTENT")
        assert len(xml_segments) == 1
        assert isinstance(xml_segments[0], ET.Element)

    def test_extract_des_header(self):
        ds = gdal.Open("./test/data/sicd_example_1_PFA_RE32F_IM32F_HH-0-0.NITF")
        des_accessor = NITFDESAccessor(ds.GetMetadata("xml:DES"))
        xml_segments = des_accessor.get_segments_by_name("XML_DATA_CONTENT")

        des_header = NITFDESAccessor.extract_des_header(xml_segments[0])
        assert des_header is not None
        assert des_header.startswith("01UUS")
        assert len(des_header) == 946


if __name__ == "__main__":
    unittest.main()
