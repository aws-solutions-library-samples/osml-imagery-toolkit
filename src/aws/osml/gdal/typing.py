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


class RangeAdjustmentType(str, Enum):
    """
    Enumeration defining ways to scale raw image pixels to an output value range.

    - NONE indicates that the full range available to the input type will be used.
    - MINMAX chooses the portion of the input range that actually contains values.
    - DRA is a dynamic range adjustment that attempts to select the most important portion of the input range.
      It differs from MINMAX in that it can exclude outliers to reduce the impact of unusually bright/dark
      spots in an image.
    """

    NONE = "NONE"
    MINMAX = "MINMAX"
    DRA = "DRA"
