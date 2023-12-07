import base64
import logging
from secrets import token_hex
from typing import Any, Dict, List, Optional, Tuple

from osgeo import gdal, gdalconst

from aws.osml.gdal import GDALCompressionOptions, GDALImageFormats, NITFDESAccessor, RangeAdjustmentType, get_type_and_scales
from aws.osml.photogrammetry import ImageCoordinate, SensorModel

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
        gdal_translate_kwargs = self.default_gdal_translate_kwargs.copy()

        if output_size is not None:
            gdal_translate_kwargs["width"] = output_size[0]
            gdal_translate_kwargs["height"] = output_size[1]

        # Create a new IGEOLO value based on the corner points of this tile
        if self.sensor_model is not None and self.tile_format == GDALImageFormats.NITF:
            gdal_translate_kwargs["creationOptions"].append("ICORDS=G")
            gdal_translate_kwargs["creationOptions"].append("IGEOLO=" + self.create_new_igeolo(src_window))

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

    def create_new_igeolo(self, src_window: List[int]) -> str:
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
