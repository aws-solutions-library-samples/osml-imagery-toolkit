#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest
from secrets import token_hex
from unittest import TestCase

import numpy as np
from osgeo import gdal, gdalconst

from aws.osml.gdal import GDALCompressionOptions, GDALImageFormats, RangeAdjustmentType, load_gdal_dataset
from aws.osml.image_processing import GDALTileFactory, MapTileId, MapTileSetFactory
from aws.osml.photogrammetry import ImageCoordinate


class TestGDALTileFactory(TestCase):
    def test_create_encoded_sicd_tile_nitf(self):
        full_dataset, sensor_model = load_gdal_dataset("./test/data/sicd_example_1_PFA_RE32F_IM32F_HH-0-0.NITF")

        tile_factory = GDALTileFactory(full_dataset, sensor_model, GDALImageFormats.NITF, GDALCompressionOptions.J2K)
        encoded_tile_data = tile_factory.create_encoded_tile([10, 10, 128, 256])

        temp_ds_name = "/vsimem/" + token_hex(16) + ".NITF"
        gdal.FileFromMemBuffer(temp_ds_name, encoded_tile_data)
        tile_dataset = gdal.Open(temp_ds_name)
        assert tile_dataset.RasterXSize == 128
        assert tile_dataset.RasterYSize == 256
        assert tile_dataset.GetDriver().ShortName == GDALImageFormats.NITF

    def test_create_encoded_sicd_tile_png(self):
        full_dataset, sensor_model = load_gdal_dataset("./test/data/sicd_example_1_PFA_RE32F_IM32F_HH-0-0.NITF")

        tile_factory = GDALTileFactory(full_dataset, sensor_model, GDALImageFormats.PNG, GDALCompressionOptions.NONE)
        encoded_tile_data = tile_factory.create_encoded_tile([10, 10, 128, 256])

        temp_ds_name = "/vsimem/" + token_hex(16) + ".PNG"
        gdal.FileFromMemBuffer(temp_ds_name, encoded_tile_data)
        tile_dataset = gdal.Open(temp_ds_name)
        assert tile_dataset.RasterXSize == 128
        assert tile_dataset.RasterYSize == 256
        assert tile_dataset.GetDriver().ShortName == GDALImageFormats.PNG

    def test_create_sicd_chip_from_chip(self):
        full_dataset, sensor_model = load_gdal_dataset("./test/data/sicd/capella-sicd121-chip1.ntf")

        tile_factory = GDALTileFactory(full_dataset, sensor_model, GDALImageFormats.NITF, GDALCompressionOptions.NONE)
        encoded_tile_data = tile_factory.create_encoded_tile([10, 10, 128, 256])

        temp_ds_name = "/vsimem/" + token_hex(16) + ".NITF"
        gdal.FileFromMemBuffer(temp_ds_name, encoded_tile_data)
        tile_dataset = gdal.Open(temp_ds_name)
        assert tile_dataset.RasterXSize == 128
        assert tile_dataset.RasterYSize == 256
        assert tile_dataset.GetDriver().ShortName == GDALImageFormats.NITF

    def test_create_sidd_chip_from_chip(self):
        full_dataset, sensor_model = load_gdal_dataset("./test/data/sidd/umbra-sidd200-chip1.ntf")

        tile_factory = GDALTileFactory(full_dataset, sensor_model, GDALImageFormats.NITF, GDALCompressionOptions.NONE)
        encoded_tile_data = tile_factory.create_encoded_tile([10, 10, 128, 256])

        temp_ds_name = "/vsimem/" + token_hex(16) + ".NITF"
        gdal.FileFromMemBuffer(temp_ds_name, encoded_tile_data)
        tile_dataset = gdal.Open(temp_ds_name)
        assert tile_dataset.RasterXSize == 128
        assert tile_dataset.RasterYSize == 256
        assert tile_dataset.GetDriver().ShortName == GDALImageFormats.NITF

    def test_create_png_with_dra(self):
        full_dataset, sensor_model = load_gdal_dataset("./test/data/small.ntf")
        tile_factory = GDALTileFactory(
            full_dataset,
            sensor_model,
            GDALImageFormats.PNG,
            GDALCompressionOptions.NONE,
            output_type=gdalconst.GDT_Byte,
            range_adjustment=RangeAdjustmentType.DRA,
        )

        full_dataset.GetRasterBand(1).ComputeStatistics(approx_ok=0)
        assert full_dataset.GetRasterBand(1).GetMinimum() == 0
        assert full_dataset.GetRasterBand(1).GetMaximum() == 255
        encoded_tile_data = tile_factory.create_encoded_tile([10, 10, 128, 256])
        temp_ds_name = "/vsimem/" + token_hex(16) + ".PNG"
        gdal.FileFromMemBuffer(temp_ds_name, encoded_tile_data)
        tile_dataset = gdal.Open(temp_ds_name)
        assert tile_dataset.RasterXSize == 128
        assert tile_dataset.RasterYSize == 256
        assert tile_dataset.GetDriver().ShortName == GDALImageFormats.PNG
        tile_dataset.GetRasterBand(1).ComputeStatistics(approx_ok=0)
        assert tile_dataset.GetRasterBand(1).GetMinimum() == 0
        assert tile_dataset.GetRasterBand(1).GetMaximum() == 185

    def test_create_png_with_output_size(self):
        full_dataset, sensor_model = load_gdal_dataset("./test/data/small.ntf")
        tile_factory = GDALTileFactory(
            full_dataset,
            sensor_model,
            GDALImageFormats.PNG,
            GDALCompressionOptions.NONE,
            output_type=gdalconst.GDT_Byte,
            range_adjustment=RangeAdjustmentType.DRA,
        )

        encoded_tile_data = tile_factory.create_encoded_tile([0, 0, 256, 512], output_size=(128, 256))
        temp_ds_name = "/vsimem/" + token_hex(16) + ".PNG"
        gdal.FileFromMemBuffer(temp_ds_name, encoded_tile_data)
        tile_dataset = gdal.Open(temp_ds_name)
        assert tile_dataset.RasterXSize == 128
        assert tile_dataset.RasterYSize == 256
        assert tile_dataset.GetDriver().ShortName == GDALImageFormats.PNG

    # Test data here could be improved. We're reusing a nitf file for everything and just
    # testing a single raster scale
    def test_create_gdal_translate_kwargs(self):
        full_dataset = gdal.Open("./test/data/GeogToWGS84GeoKey5.tif")

        format_compression_combinations = [
            (GDALImageFormats.NITF, GDALCompressionOptions.NONE, "IC=NC"),
            (GDALImageFormats.NITF, GDALCompressionOptions.JPEG, "IC=C3"),
            (GDALImageFormats.NITF, GDALCompressionOptions.J2K, "IC=C8"),
            (GDALImageFormats.NITF, "FAKE", ""),
            (GDALImageFormats.NITF, None, "IC=C8"),
            (GDALImageFormats.JPEG, GDALCompressionOptions.NONE, None),
            (GDALImageFormats.JPEG, GDALCompressionOptions.JPEG, None),
            (GDALImageFormats.JPEG, GDALCompressionOptions.J2K, None),
            (GDALImageFormats.JPEG, "FAKE", None),
            (GDALImageFormats.JPEG, None, None),
            (GDALImageFormats.PNG, GDALCompressionOptions.NONE, None),
            (GDALImageFormats.PNG, GDALCompressionOptions.JPEG, None),
            (GDALImageFormats.PNG, GDALCompressionOptions.J2K, None),
            (GDALImageFormats.PNG, "FAKE", None),
            (GDALImageFormats.PNG, None, None),
            (GDALImageFormats.GTIFF, GDALCompressionOptions.NONE, None),
            (GDALImageFormats.GTIFF, GDALCompressionOptions.JPEG, None),
            (GDALImageFormats.GTIFF, GDALCompressionOptions.J2K, None),
            (GDALImageFormats.GTIFF, GDALCompressionOptions.LZW, None),
            (GDALImageFormats.GTIFF, "FAKE", None),
            (GDALImageFormats.GTIFF, None, None),
        ]

        for image_format, image_compression, expected_options in format_compression_combinations:
            tile_factory = GDALTileFactory(full_dataset, None, image_format, image_compression)
            gdal_translate_kwargs = tile_factory._create_gdal_translate_kwargs()

            assert gdal_translate_kwargs["format"] == image_format
            assert gdal_translate_kwargs["scaleParams"] == [[0, 255, 0, 255]]
            assert gdal_translate_kwargs["outputType"] == 1
            if expected_options:
                assert gdal_translate_kwargs["creationOptions"][0] == expected_options

    def test_create_map_tile_000(self):
        full_dataset, sensor_model = load_gdal_dataset("./test/data/small.ntf")
        tile_factory = GDALTileFactory(
            full_dataset,
            sensor_model,
            GDALImageFormats.PNG,
            GDALCompressionOptions.NONE,
            output_type=gdalconst.GDT_Byte,
            range_adjustment=RangeAdjustmentType.DRA,
        )

        encoded_tile_data = tile_factory.create_orthophoto_tile(
            geo_bbox=(-np.pi, -np.pi / 2, np.pi, np.pi / 2), tile_size=(256, 256)
        )
        temp_ds_name = "/vsimem/" + token_hex(16) + ".PNG"
        gdal.FileFromMemBuffer(temp_ds_name, encoded_tile_data)
        tile_dataset = gdal.Open(temp_ds_name)
        assert tile_dataset.RasterXSize == 256
        assert tile_dataset.RasterYSize == 256
        assert tile_dataset.GetDriver().ShortName == GDALImageFormats.PNG

    def test_create_map_tile_no_overlap(self):
        full_dataset, sensor_model = load_gdal_dataset("./test/data/small.ntf")
        tile_factory = GDALTileFactory(
            full_dataset,
            sensor_model,
            GDALImageFormats.PNG,
            GDALCompressionOptions.NONE,
            output_type=gdalconst.GDT_Byte,
            range_adjustment=RangeAdjustmentType.DRA,
        )

        encoded_tile_data = tile_factory.create_orthophoto_tile(geo_bbox=(0.0, 0.0, 0.01, 0.01), tile_size=(256, 256))
        assert encoded_tile_data is None

    def test_create_map_tiles_for_image(self):
        tile_set_id = "WebMercatorQuad"
        tile_set = MapTileSetFactory.get_for_id(tile_set_id)
        full_dataset, sensor_model = load_gdal_dataset("./test/data/small.ntf")
        tile_factory = GDALTileFactory(
            full_dataset,
            sensor_model,
            GDALImageFormats.PNG,
            GDALCompressionOptions.NONE,
            output_type=gdalconst.GDT_Byte,
            range_adjustment=RangeAdjustmentType.DRA,
        )

        tile_matrix = 14
        image_corners = [
            ImageCoordinate(coord)
            for coord in [
                [0, 0],
                [full_dataset.RasterXSize, 0],
                [full_dataset.RasterXSize, full_dataset.RasterYSize],
                [0, full_dataset.RasterYSize],
            ]
        ]
        world_corners = [sensor_model.image_to_world(image_coordinate) for image_coordinate in image_corners]
        min_col, min_row, max_col, max_row = tile_set.get_tile_matrix_limits_for_area(
            boundary_coordinates=world_corners, tile_matrix=tile_matrix
        )

        for tile_row in range(min_row, max_row + 1):
            for tile_col in range(min_col, max_col + 1):
                map_tile = tile_set.get_tile(MapTileId(tile_matrix=tile_matrix, tile_row=tile_row, tile_col=tile_col))
                encoded_tile_data = tile_factory.create_orthophoto_tile(geo_bbox=map_tile.bounds, tile_size=map_tile.size)
                assert encoded_tile_data is not None
                temp_ds_name = "/vsimem/" + token_hex(16) + ".PNG"
                gdal.FileFromMemBuffer(temp_ds_name, encoded_tile_data)
                tile_dataset = gdal.Open(temp_ds_name)
                assert tile_dataset.RasterXSize == 256
                assert tile_dataset.RasterYSize == 256
                assert tile_dataset.GetDriver().ShortName == GDALImageFormats.PNG


if __name__ == "__main__":
    unittest.main()
