#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import base64
import copy
import logging
from secrets import token_hex
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from osgeo import gdal, gdalconst
from scipy.interpolate import RectBivariateSpline

from aws.osml.gdal import GDALCompressionOptions, GDALImageFormats, NITFDESAccessor, RangeAdjustmentType, get_type_and_scales
from aws.osml.gdal.dynamic_range_adjustment import DRAParameters
from aws.osml.photogrammetry import GeodeticWorldCoordinate, ImageCoordinate, SensorModel

from .sicd_updater import SICDUpdater
from .sidd_updater import SIDDUpdater

logger = logging.getLogger(__name__)


class GDALTileFactory:
    """
    This class creates tiles from a larger image on request. Image metadata is retained whenever possible but updated
    as necessary to account for the new raster bounds.
    """

    def __init__(
        self,
        raster_dataset: gdal.Dataset,
        sensor_model: Optional[SensorModel] = None,
        tile_format: GDALImageFormats = GDALImageFormats.NITF,
        tile_compression: GDALCompressionOptions = GDALCompressionOptions.NONE,
        output_type: Optional[int] = None,
        range_adjustment: RangeAdjustmentType = RangeAdjustmentType.NONE,
    ):
        """
        Constructs a new factory capable of producing tiles from a given GDAL raster dataset.

        :param raster_dataset: the original raster dataset to create tiles from
        :param sensor_model: the sensor model providing mensuration support for this image
        :param tile_format: the output tile format
        :param tile_compression: the output tile compression
        :param output_type: the GDAL pixel type in the output tile
        :param range_adjustment: the type of scaling used to convert raw pixel values to the output range
        """
        self.tile_format = tile_format
        self.tile_compression = tile_compression
        self.raster_dataset = raster_dataset
        self.sensor_model = sensor_model
        self.des_accessor = None
        self.sar_updater = None
        self.sar_des_header = None
        self.range_adjustment = range_adjustment
        self.output_type = output_type

        if self.raster_dataset.GetDriver().ShortName == "NITF":
            xml_des = self.raster_dataset.GetMetadata("xml:DES")
            self.des_accessor = NITFDESAccessor(xml_des)

            xml_data_content_segments = self.des_accessor.get_segments_by_name("XML_DATA_CONTENT")
            if xml_data_content_segments is not None and len(xml_data_content_segments) > 0:
                # This appears to be SICD or SIDD data
                xml_data_segment = xml_data_content_segments[0]
                xml_bytes = self.des_accessor.parse_field_value(xml_data_segment, "DESDATA", base64.b64decode)
                xml_str = xml_bytes.decode("utf-8")
                if "SIDD" in xml_str:
                    self.sar_des_header = self.des_accessor.extract_des_header(xml_data_segment)
                    self.sar_updater = SIDDUpdater(xml_str)
                elif "SICD" in xml_str:
                    self.sar_des_header = self.des_accessor.extract_des_header(xml_data_segment)
                    self.sar_updater = SICDUpdater(xml_str)

        self.default_gdal_translate_kwargs = self._create_gdal_translate_kwargs()

    def create_encoded_tile(
        self, src_window: List[int], output_size: Optional[Tuple[int, int]] = None
    ) -> Optional[bytearray]:
        """
        This method cuts a tile from the full image, updates the metadata as needed, and finally compresses/encodes
        the result in the output format requested.

        :param src_window: the [left_x, top_y, width, height] bounds of this tile
        :param output_size: an optional size of the output tile (width, height)
        :return: the encoded image tile or None if one could not be produced
        """
        temp_ds_name = f"/vsimem/{token_hex(16)}.{self.tile_format}"

        # Use the request and metadata from the raster dataset to create a set of keyword
        # arguments for the gdal.Translate() function. This will configure that function to
        # create image tiles using the format, compression, etc. requested by the client.
        gdal_translate_kwargs = copy.deepcopy(self.default_gdal_translate_kwargs)

        if output_size is not None:
            gdal_translate_kwargs["width"] = output_size[0]
            gdal_translate_kwargs["height"] = output_size[1]

        # Create a new IGEOLO value based on the corner points of this tile
        if self.sensor_model is not None and self.tile_format == GDALImageFormats.NITF:
            gdal_translate_kwargs["creationOptions"].append("ICORDS=G")
            gdal_translate_kwargs["creationOptions"].append("IGEOLO=" + self._create_new_igeolo(src_window))

        if self.sar_updater is not None and self.tile_format == GDALImageFormats.NITF:
            # If we're outputting a SICD or SIDD tile we need to update the XML metadata to include the new chip
            # origin and size. This will allow applications using the tile to correctly interpret the remaining
            # image metadata.
            self.sar_updater.update_image_data_for_chip(src_window, output_size)
            updated_sar_des = self.sar_des_header + self.sar_updater.encode_current_xml()

            gdal_translate_kwargs["creationOptions"].append("ICAT=SAR")
            gdal_translate_kwargs["creationOptions"].append("IREP=NODISPLY")
            gdal_translate_kwargs["creationOptions"].append("IREPBAND= , ")
            gdal_translate_kwargs["creationOptions"].append("ISUBCAT=I,Q")
            gdal_translate_kwargs["creationOptions"].append("DES=XML_DATA_CONTENT=" + updated_sar_des)

        # Use GDAL to create an encoded tile of the image region
        # From GDAL documentation:
        #   srcWin --- subwindow in pixels to extract:
        #               [left_x, top_y, width, height]
        gdal.Translate(
            temp_ds_name,
            self.raster_dataset,
            srcWin=src_window,
            **gdal_translate_kwargs,
        )

        # Read the VSIFile
        vsifile_handle = None
        try:
            vsifile_handle = gdal.VSIFOpenL(temp_ds_name, "r")
            if vsifile_handle is None:
                return None
            stat = gdal.VSIStatL(temp_ds_name, gdal.VSI_STAT_SIZE_FLAG)
            vsibuf = gdal.VSIFReadL(1, stat.size, vsifile_handle)
            return vsibuf
        finally:
            if vsifile_handle is not None:
                gdal.VSIFCloseL(vsifile_handle)
            gdal.GetDriverByName(self.tile_format).Delete(temp_ds_name)

    def create_orthophoto_tile(
        self, geo_bbox: Tuple[float, float, float, float], tile_size: Tuple[int, int]
    ) -> Optional[bytearray]:
        """
        This method creates an orthorectified tile from an image assuming there is overlap in the coverage.

        IMPORTANT: This is an experimental API that may change in future minor releases of the toolkit. This
        early release is subject to the following limitations:
        - All tiles are returned in PNG format
        - An 8-bit conversion and dynamic range mapping is automatically applied using the statistics of the tile

        :param geo_bbox: the geographic bounding box of the tile in the form (min_lon, min_lat, max_lon, max_lat)
        :param tile_size: the shape of the output tile (width, height)
        :return: the encoded image tile or None if one could not be produced
        """
        min_lon, min_lat, max_lon, max_lat = geo_bbox

        # Setup 2 grids of the same size, the first is for longitude/latitude coordinates and the second is pixel
        # coordinates. These grids are evenly spaced across the map tile. Note that the latitude and pixel row
        # grids have been adjusted because the 0, 0 pixel is in the upper left corner of the map tile and as the
        # image row increases the latitude should decrease. That is why the world_y grid ranges from max to min
        # while all other grids range from min to max.
        nx, ny = (3, 3)
        world_x = np.linspace(min_lon, max_lon, nx)
        world_y = np.linspace(max_lat, min_lat, ny)
        world_xv, world_yv = np.meshgrid(world_x, world_y)

        pixel_x = np.linspace(0, tile_size[0] - 1, nx)
        pixel_y = np.linspace(0, tile_size[1] - 1, ny)

        # Use the sensor model to compute the image pixel location that corresponds to each
        # world coordinate in the map tile grid. Separate those results into arrays of the x, y
        # component.
        def world_to_image_func(lon, lat):
            # TODO: Assign the elevation from a DEM
            return self.sensor_model.world_to_image(GeodeticWorldCoordinate([lon, lat, 0.0]))

        world_to_image_func_vectorized = np.vectorize(world_to_image_func)
        src_coords = world_to_image_func_vectorized(world_xv, world_yv)
        src_x = np.vectorize(lambda image_coord: image_coord.x)(src_coords)
        src_y = np.vectorize(lambda image_coord: image_coord.y)(src_coords)

        # Find min/max x and y for this grid and check to make sure it actually overlaps the image.
        src_bbox = (
            int(np.floor(np.min(src_x))),
            int(np.floor(np.min(src_y))),
            int(np.ceil(np.max(src_x))),
            int(np.ceil(np.max(src_y))),
        )
        if (
            src_bbox[0] > self.raster_dataset.RasterXSize
            or src_bbox[1] > self.raster_dataset.RasterYSize
            or src_bbox[2] < 0
            or src_bbox[3] < 0
        ):
            # Source bbox does not intersect the image, no tile to create
            return None

        # Clip the source to the extent of the image and then select an overview level of similar resolution to the
        # desired map tile. This will ensure we only read the minimum number of pixels necessary and warp them as
        # little as possible.
        src_bbox = (
            max(src_bbox[0], 0),
            max(src_bbox[1], 0),
            min(src_bbox[2], self.raster_dataset.RasterXSize),
            min(src_bbox[3], self.raster_dataset.RasterYSize),
        )
        logger.debug(f"After Clip to Image Bounds src_bbox = {src_bbox}")

        def find_appropriate_r_level(src_bbox, tile_width) -> int:
            src_dim = np.min([src_bbox[2] - src_bbox[0], src_bbox[3] - src_bbox[1]])
            return int(np.max([0, int(np.floor(np.log2(src_dim / tile_width)))]))

        num_overviews = self.raster_dataset.GetRasterBand(1).GetOverviewCount()
        r_level = min(find_appropriate_r_level(src_bbox, tile_size[0]), num_overviews)
        overview_bbox = tuple([int(x / 2**r_level) for x in src_bbox])
        logger.debug(f"Using r-level: {r_level}")
        logger.debug(f"overview_bbox = {overview_bbox}")
        logger.debug(f"Dataset size = {self.raster_dataset.RasterXSize},{self.raster_dataset.RasterYSize}")

        # Read pixels from the selected resolution level that match the region of the image needed to create the
        # map tile. This data becomes the "src" in the cv2.remap transformation.
        src = self._read_from_rlevel_as_array(overview_bbox, r_level)
        logger.debug(f"src.shape = {src.shape}")

        # Update the src_x and src_y coordinates because we cropped the image and pulled it from a different
        # resolution level. The original coordinates assumed the image origin at 0,0 in a full resolution
        # image
        src_x = (src_x - src_bbox[0]) / 2**r_level
        src_y = (src_y - src_bbox[1]) / 2**r_level

        # Create 2D linear interpolators that map the pixels in the map tile to x and y values in the source image.
        # This will allow us to efficiently generate the maps needed by the opencv::remap function for every pixel
        # in the destination image.
        src_x_interpolator = RectBivariateSpline(pixel_x, pixel_y, src_x, kx=1, ky=1)
        src_y_interpolator = RectBivariateSpline(pixel_x, pixel_y, src_y, kx=1, ky=1)

        # Create the map1 and map2 arrays that capture the non-linear relationship between each pixel in the map tile
        # (dst) to pixels in the original image (src). See opencv::remap documentation for definitions of these
        # parameters.
        dst_x = np.linspace(0, tile_size[0] - 1, tile_size[0])
        dst_y = np.linspace(0, tile_size[1] - 1, tile_size[1])
        map1 = src_x_interpolator(dst_x, dst_y).astype(np.float32)
        map2 = src_y_interpolator(dst_x, dst_y).astype(np.float32)

        logger.debug(
            f"Sanity check remap array sizes. " f"They should match the desired map tile size {tile_size[0]}x{tile_size[1]}"
        )
        logger.debug(f"map1.shape = {map1.shape}")
        logger.debug(f"map2.shape = {map2.shape}")

        dst = cv2.remap(src, map1, map2, cv2.INTER_LINEAR)
        output_tile_pixels = self._create_display_image(dst)

        # TODO: Formats other than PNG?
        is_success, image_bytes = cv2.imencode(".png", output_tile_pixels)
        if is_success:
            return image_bytes
        else:
            return None

    def _read_from_rlevel_as_array(
        self, scaled_bbox: Tuple[int, int, int, int], r_level: int, band_numbers: Optional[List[int]] = None
    ) -> np.array:
        """
        This method reads a region of the image from a specific image resolution level (r-level). The bounding box
        must be scaled to match the resolution level.

        :param scaled_bbox: a [minx, miny, maxx, maxy] bbox in the pixel coordinate system of the r_level
        :param r_level: the selected resolution level, r0 = full resolution image, r1 = first overview, ...
        :param band_numbers: the bands to select, if None all bands will be read. Note band numbers start at 1
        :return: a NumPy array of shape [r, c, b] for images with multiple bands or [r, c] for images with just 1 band
        """

        # If no bands are specified we will read them all.
        if not band_numbers:
            band_numbers = [n + 1 for n in range(0, self.raster_dataset.RasterCount)]

        # Loop through the bands of this image and retrieve the relevant pixels
        band_pixels = []
        for band_num in band_numbers:
            ds_band = self.raster_dataset.GetRasterBand(band_num)
            if r_level > 0:
                overview = ds_band.GetOverview(r_level - 1)
            else:
                overview = ds_band

            band_pixels.append(
                overview.ReadAsArray(
                    scaled_bbox[0],
                    scaled_bbox[1],
                    scaled_bbox[2] - scaled_bbox[0],
                    scaled_bbox[3] - scaled_bbox[1],
                )
            )

        # If the image has multiple bands then we can stack the results with the band being the 3rd dimension
        # in the array. This aligns to how OpenCV wants to work with imagery. If the image doesn't have multiple
        # bands then return a 2-dimensional grayscale image.
        if self.raster_dataset.RasterCount > 1:
            result = np.stack(band_pixels, axis=2)
        else:
            result = band_pixels[0]

        return result

    def _create_display_image(self, pixel_array: np.array) -> np.array:
        """
        This method selects the first 3 bands of a multi-band image (or a single band of grayscale) and performs a
        simple dynamic range adjustment to map those values to the 0-255 range of an 8-bit per pixel image for
        visualization.

        :param pixel_array: the input image pixels
        :return: a range adjusted 8-bit per pixel image of up to 3 bands
        """
        if pixel_array.ndim == 3 and pixel_array.shape[2] > 3:
            pixel_array = pixel_array[:, :, 0:3]

        max_channel_value = np.max(pixel_array)
        hist, bin_edges = np.histogram(pixel_array.ravel(), bins=max_channel_value + 1, range=(0, max_channel_value))
        dra_parameters = DRAParameters.from_counts(hist.tolist(), max_percentage=0.97)
        for_display = (
            255
            * (pixel_array - dra_parameters.suggested_min_value)
            / max(dra_parameters.suggested_max_value - dra_parameters.suggested_min_value, 1.0)
        )

        for_display = np.clip(for_display, 0.0, 255.0)
        for_display = for_display.astype(np.uint8)

        return for_display

    def _create_new_igeolo(self, src_window: List[int]) -> str:
        """
        Create a new 60 character string representing the corner coordinates of this tile. The string conforms to
        the geographic "G" choice for the ICORDS field as specified in MIL-STD-2500C Table A-3 NITF Image Subheader.

        The locations of the four corners of are encoded image coordinate order:
        (0,0), (Max X, 0), (Max X, Max Y), (0, Max Y) with each corner represented as a ddmmssXdddmmssY string
        representing latitude and longitude. The first half, ddmmssX, represents degrees, minutes, and seconds of
        latitude with X representing North or South (N for North, S for South). The second half, dddmmssY, represents
        degrees, minutes, and seconds of longitude with Y representing East or West (E for East, W for West),
        respectively.

        :param src_window: the [left_x, top_y, width, height] bounds of this tile
        :return: the 60 character IGEOLO geographic coordinate string
        """
        tile_corners = [
            [src_window[0], src_window[1]],
            [src_window[0] + src_window[2], src_window[1]],
            [src_window[0] + src_window[2], src_window[1] + src_window[3]],
            [src_window[0], src_window[1] + src_window[3]],
        ]
        dms_coords = []
        for coord in tile_corners:
            geodetic_tile_corner = self.sensor_model.image_to_world(ImageCoordinate(coord))
            dms_coords.append(geodetic_tile_corner.to_dms_string())
        return "".join(dms_coords)

    def _create_gdal_translate_kwargs(self) -> Dict[str, Any]:
        """
        This method creates a set of keyword arguments suitable for passing to the gdal.Translate
        function. The values for these options are derived from the region processing request and
        the raster dataset itself.

        See: https://gdal.org/python/osgeo.gdal-module.html#Translate
        See: https://gdal.org/python/osgeo.gdal-module.html#TranslateOptions

        :return: Dict[str, any] = the dictionary of translate keyword arguments
        """
        # Figure out what type of image this is and calculate a scale to map input pixels to the output type
        output_type, scale_params = get_type_and_scales(
            self.raster_dataset, desired_output_type=self.output_type, range_adjustment=self.range_adjustment
        )

        gdal_translate_kwargs = {
            "scaleParams": scale_params,
            "outputType": output_type,
            "format": self.tile_format,
        }

        creation_options = []
        if self.tile_format == GDALImageFormats.NITF:
            # Creation options specific to the NITF raster driver.
            # See: https://gdal.org/drivers/raster/nitf.html
            if self.tile_compression is None or self.tile_compression == GDALCompressionOptions.J2K:
                # Default NITF tiles to JPEG2000 compression if not specified
                # The OpenJPEG library does not currently support floating point pixel types. If J2K compression
                # was requested we need to fall back to no compression to avoid a failure. This update can be
                # removed if compression of floating points is supported or if we move to an alternate J2K library.
                if output_type in [gdalconst.GDT_Float32, gdalconst.GDT_Float64]:
                    logger.warning(
                        "OpenJPEG J2K library does not support floating point pixel values. "
                        "Outputs will not be compressed."
                    )
                    creation_options.append("IC=NC")
                else:
                    creation_options.append("IC=C8")
            elif self.tile_compression == GDALCompressionOptions.JPEG:
                creation_options.append("IC=C3")
            elif self.tile_compression == GDALCompressionOptions.NONE:
                creation_options.append("IC=NC")
            else:
                logging.warning("Invalid compress specified for NITF image defaulting to JPEG2000!")
                creation_options.append("IC=C8")
        elif self.tile_format == GDALImageFormats.GTIFF:
            # Creation options specific to the GeoTIFF raster driver.
            # See: https://gdal.org/drivers/raster/nitf.html
            if self.tile_compression is None:
                # Default GeoTiff tiles to LZQ compression if not specified
                creation_options.append("COMPRESS=LZW")
            elif self.tile_compression == GDALCompressionOptions.LZW:
                creation_options.append("COMPRESS=LZW")
            elif self.tile_compression == GDALCompressionOptions.JPEG:
                creation_options.append("COMPRESS=JPEG")
            elif self.tile_compression == GDALCompressionOptions.NONE:
                creation_options.append("COMPRESS=NONE")
            else:
                logging.warning("Invalid compress specified for GTIFF image defaulting to LZW!")
                creation_options.append("COMPRESS=LZW")

        gdal_translate_kwargs["creationOptions"] = creation_options

        return gdal_translate_kwargs
