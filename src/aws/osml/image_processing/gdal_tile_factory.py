#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import base64
import copy
import logging
from secrets import token_hex
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from osgeo import gdal, gdalconst
from scipy.interpolate import RectBivariateSpline

from aws.osml.gdal import GDALCompressionOptions, GDALImageFormats, NITFDESAccessor, RangeAdjustmentType, get_type_and_scales
from aws.osml.gdal.dynamic_range_adjustment import DRAParameters
from aws.osml.photogrammetry import GeodeticWorldCoordinate, ImageCoordinate, SensorModel

from .sar_complex_imageop import quarter_power_image
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
        # component. Note that if an external elevation model is not provided this code will
        # use a default elevation provided by the sensor model for a location at the center of
        # the image.
        center_x = self.raster_dataset.RasterXSize / 2
        center_y = self.raster_dataset.RasterYSize / 2
        geo_image_center = self.sensor_model.image_to_world(ImageCoordinate([center_y, center_x]))

        def world_to_image_func(lon, lat):
            # TODO: Assign the elevation from a DEM
            default_elevation = geo_image_center.elevation
            return self.sensor_model.world_to_image(GeodeticWorldCoordinate([lon, lat, default_elevation]))

        try:
            world_to_image_func_vectorized = np.vectorize(world_to_image_func)
            src_coords = world_to_image_func_vectorized(world_xv, world_yv)
            src_x = np.vectorize(lambda image_coord: image_coord.x)(src_coords)
            src_y = np.vectorize(lambda image_coord: image_coord.y)(src_coords)
        except Exception as e:
            # Unable to convert the map tile coordinates to image coordinates using the sensor model.
            # This usually means at least one coordinate isn't near the image and fell outside the range
            # of values the sensor model could create. No map tile can be created from this image.
            logger.debug("Unable to convert map tile coordinates to image coordinates.", e)
            return None

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
            logger.debug(
                f"Requested map tile does not overlap with image: map bbox: {src_bbox}"
                f" image bbox: {(0, 0, self.raster_dataset.RasterXSize, self.raster_dataset.RasterYSize)}"
            )
            return None

        # Select the image overview level that most closely matches the ground sample distance of the
        # requested map tile. Note that this must be done before the source bounding box is clipped to the
        # actual image extend otherwise tiles that only overlap on the edge of the image may be read from a
        # very different resolution level than the other tiles at a similar map zoom level.
        def find_appropriate_r_level(src_bbox, tile_width) -> int:
            src_dim = np.min([src_bbox[2] - src_bbox[0], src_bbox[3] - src_bbox[1]])
            return int(np.max([0, int(np.floor(np.log2(src_dim / tile_width)))]))

        num_overviews = self.raster_dataset.GetRasterBand(1).GetOverviewCount()
        r_level = min(find_appropriate_r_level(src_bbox, tile_size[0]), num_overviews)

        src_bbox = (
            max(src_bbox[0], 0),
            max(src_bbox[1], 0),
            min(src_bbox[2], self.raster_dataset.RasterXSize),
            min(src_bbox[3], self.raster_dataset.RasterYSize),
        )
        logger.debug(f"After Clip to Image Bounds src_bbox = {src_bbox}")

        overview_bbox = tuple([int(x / 2**r_level) for x in src_bbox])
        logger.debug(f"Using r-level: {r_level}")
        logger.debug(f"overview_bbox = {overview_bbox}")
        logger.debug(f"Dataset size = {self.raster_dataset.RasterXSize},{self.raster_dataset.RasterYSize}")

        # Read pixels from the selected resolution level that match the region of the image needed to create the
        # map tile. This data becomes the "src" in the cv2.remap transformation.
        src = self._read_from_rlevel_as_array(overview_bbox, r_level)
        logger.debug(f"src.shape = {src.shape}")

        # Convert the raw image pixels into a 8-bit per pixel image suitable for human review. These transformations
        # are applied before the remapping because the remapping itself may alter the distribution of pixel values
        # and the normalization itself may rely on precomputed pixel statistics/histograms that were calculated
        # based on the raw pixel values.
        src = self._normalize_image_for_display(src)

        # Add a replicate border around the source image to handle out-of-bound coordinates during remapping.
        # When using cv2.remap, some coordinates may fall outside the boundaries of the source image.
        # Using cv2.BORDER_CONSTANT or setting scalar values will assign a default color (such as black)
        # to pixels around the border, leading to unwanted uniform-colored edges.
        # By using cv2.BORDER_REPLICATE, the edge pixels of the source image are repeated in the border region,
        # allowing remap to use valid pixel values for out-of-bound coordinates, avoiding default color (black lines)
        # around the output image.
        border_size = 2
        src_bordered = cv2.copyMakeBorder(src, border_size, border_size, border_size, border_size, cv2.BORDER_REPLICATE)

        # Update the src_x and src_y coordinates because we cropped the image and pulled it from a different
        # resolution level. The original coordinates assumed the image origin at 0,0 in a full resolution
        # image and adjusted the remapping coordinates to account for the added border
        src_x = (src_x - src_bbox[0]) / 2**r_level + border_size
        src_y = (src_y - src_bbox[1]) / 2**r_level + border_size

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
            f"Sanity check remap array sizes. They should match the desired map tile size {tile_size[0]}x{tile_size[1]}"
        )
        logger.debug(f"map1.shape = {map1.shape}")
        logger.debug(f"map2.shape = {map2.shape}")

        # Transform image
        dst = cv2.remap(src_bordered, map1, map2, cv2.INTER_LINEAR)

        # Create alpha layer mask
        alpha_mask = None
        if dst.ndim == 2:  # 1-band grayscale
            all_channel_pixels_mask = dst != 0
            alpha_mask = np.zeros_like(dst, dtype=np.uint8)
            alpha_mask[all_channel_pixels_mask] = 255
        elif dst.ndim >= 3:  # multi band image e.g. RGB, complex SAR, MSI, etc.
            all_channel_pixels_mask = np.all(dst != 0, axis=2)
            alpha_mask = np.zeros_like(dst[..., 0], dtype=np.uint8)
            alpha_mask[all_channel_pixels_mask] = 255
        if alpha_mask is not None:  # arrays with zeros/zero size can be falsy so explicitly check None
            logger.debug(f"alpha_mask.shape = {alpha_mask.shape}")
        elif dst.ndim > 2:
            logger.debug(f"alpha_mask = None.  Image has {dst.ndim} dimensions and {dst.shape[2]} bands.")
        else:
            logger.debug(f"alpha_mask = None.  Image has {dst.ndim} dimensions.")

        output_tile_pixels = dst
        if alpha_mask is not None:
            # imencode does not support 2-band (grayscale + alpha) so the workaround is to convert to 3-band
            if output_tile_pixels.ndim == 2:
                output_tile_pixels = np.dstack((output_tile_pixels, output_tile_pixels, output_tile_pixels))
            # add alpha mask
            output_tile_pixels = np.dstack((output_tile_pixels, alpha_mask))

        # TODO: Formats other than PNG?
        is_success, image_bytes = cv2.imencode(".png", output_tile_pixels)
        return image_bytes if is_success else None

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

    def _normalize_image_for_display(self, pixel_array: np.array) -> np.array:
        """
        This method applies the specified range adjustment to the requested image and returns the adjusted pixels. If
        the range adjustment is set to NONE it returns the original pixels.

        :param pixel_array: the input image pixels
        :return: a range adjusted 8-bit per pixel image
        """
        if self.range_adjustment == RangeAdjustmentType.DRA:
            return self._normalize_bands(pixel_array, self._normalize_band_dra)
        elif self.range_adjustment == RangeAdjustmentType.MINMAX:
            return self._normalize_bands(pixel_array, self._normalize_band_minmax)
        else:
            return pixel_array.astype(np.uint8)

    def _normalize_bands(
        self, pixel_array: np.array, normalize_band_func: Callable[[gdal.Band, np.array], np.array]
    ) -> np.array:
        """
        This method determines if the input pixel array is grayscale or multiband.  If it is multiband it selects
        the first 3 bands to process.  The appropriate bands are passed to the normalized_band_func to perform the
        actual transform.

        :param pixel_array: the input image pixels
        :param normalize_band_func: function to use to normalize the pixels
        :return: the range adjusted 8-bit per pixel image
        """
        if pixel_array.ndim == 2:  # 1-band grayscale image
            band = self.raster_dataset.GetRasterBand(1)
            normalized_pixels = normalize_band_func(band, pixel_array)
        elif pixel_array.ndim == 3 and pixel_array.shape[2] >= 3:
            # Multiband image, select the first 3 bands
            pixel_array = pixel_array[:, :, 0:3]
            band_count = 3
            normalized_bands = []
            for band_idx in range(band_count):
                band = self.raster_dataset.GetRasterBand(band_idx + 1)
                normalized_bands.append(normalize_band_func(band, pixel_array[:, :, band_idx]))

            # Combine the normalized bands into a single image
            normalized_pixels = np.stack(normalized_bands, axis=2)
        elif pixel_array.ndim == 3 and pixel_array.shape[2] == 2:
            # TODO: Better checks to ensure this is a Complex SAR image and not an arbitrary 2-band image
            logger.debug("Complex SAR Image Pixels. Computing quarter power image for visualization")
            normalized_pixels = self._normalize_complex_sar(pixel_array)
        else:
            logger.debug("Skipping normalization.")
            normalized_pixels = pixel_array.astype(np.uint8)
        return normalized_pixels

    def _normalize_complex_sar(self, pixel_array):
        """
        This method combines the 2-band complex SAR pixels into a single power image and then adjusts the histogram
        to fit neatly into a 0-255 range.

        :param pixel_array: the input image pixels
        :return: a visualization ready quarter power image of the SAR complex values (1 band, 8-bit per pixel)
        """
        precomputed_mean = None
        approx_abs_band_means = []
        for band_num in range(1, self.raster_dataset.RasterCount + 1):
            band_stats = self.raster_dataset.GetRasterBand(band_num).GetStatistics(True, False)
            if band_stats and len(band_stats) == 4:
                min_value, max_value, mean_value, std_value = band_stats
                approx_abs_band_means.append(std_value * 50 / 68 + mean_value)
        if len(approx_abs_band_means) > 0:
            precomputed_mean = np.sqrt(np.sum(np.square(approx_abs_band_means)))

        band_first = pixel_array.transpose((2, 0, 1))
        normalized_pixels = quarter_power_image(band_first, scale_factor=3.0, precomputed_mean=precomputed_mean)
        return normalized_pixels

    @staticmethod
    def _normalize_band_minmax(band: gdal.Band, pixel_array: np.array) -> np.array:
        """
        This method applies Min-Max normalization to an individual band. It attempts to us the min-max values from the
        image but if they are none it reverts to using values from the individual tile.

        :param pixel_array: the input image pixels
        :return: a Min-Max normalized 8-bit per pixel image
        """
        # Get the minimum and maximum values from the entire GDAL dataset. If the minimum and maximum values are
        # not available, calculate them from the tile.
        min_value = band.GetMinimum() if band.GetMinimum() is not None else np.min(pixel_array)
        max_value = band.GetMaximum() if band.GetMaximum() is not None else np.max(pixel_array)

        # Apply the Min-Max normalization
        normalized_pixel = (pixel_array - min_value) * (255.0 / max(max_value, 1.0))
        normalized_pixel = np.clip(normalized_pixel, 0.0, 255.0)
        return normalized_pixel.astype(np.uint8)

    @staticmethod
    def _normalize_band_dra(band: gdal.Band, pixel_array: np.array) -> np.array:
        """
        This method performs DRA on an input pixel_array using gdal Band data. Normalization is performed with
        respect to the entire image using the GDAL histogram.

        :param pixel_array: the input image pixels
        :return: a range adjusted 8-bit per pixel image
        """
        # Get the minimum and maximum values from the entire GDAL dataset. If the minimum and maximum values are
        # not available, calculate them from the tile.
        min_value = band.GetMinimum() if band.GetMinimum() is not None else np.min(pixel_array)
        max_value = band.GetMaximum() if band.GetMaximum() is not None else np.max(pixel_array)

        hist = band.GetHistogram(min=min_value, max=max_value, buckets=256)
        dra_parameters = DRAParameters.from_counts(hist, max_percentage=0.97)
        normalized_pixel = (
            255
            * (pixel_array - dra_parameters.suggested_min_value)
            / max(dra_parameters.suggested_max_value - dra_parameters.suggested_min_value, 1.0)
        )
        normalized_pixel = np.clip(normalized_pixel, 0.0, 255.0)
        return normalized_pixel.astype(np.uint8)

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
