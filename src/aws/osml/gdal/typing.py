from enum import Enum


class GDALCompressionOptions(str, Enum):
    """
    Enumeration defining compression algorithms for image.
    """

    NONE = "NONE"
    JPEG = "JPEG"
    J2K = "J2K"
    LZW = "LZW"


class GDALImageFormats(str, Enum):
    """
    Subset of GDAL supported image formats commonly used by this software. See
    https://gdal.org/drivers/raster/index.html for a complete listing of the formats.
    """

    NITF = "NITF"
    JPEG = "JPEG"
    PNG = "PNG"
    GTIFF = "GTiff"
