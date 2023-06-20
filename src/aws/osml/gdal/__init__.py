# Telling flake8 to not flag errors in this file. It is normal that these classes are imported but not used in an
# __init__.py file.
# flake8: noqa
"""
The gdal package contains utilities that assist with loading imagery and metadata using the OSGeo GDAL library.
"""

from .gdal_config import GDALConfigEnv, set_gdal_default_configuration
from .gdal_dem_tile_factory import GDALDigitalElevationModelTileFactory
from .gdal_utils import get_image_extension, get_type_and_scales, load_gdal_dataset
from .nitf_des_accessor import NITFDESAccessor
from .sensor_model_factory import ChippedImageInfoFacade, SensorModelFactory, SensorModelTypes
from .typing import GDALCompressionOptions, GDALImageFormats

__all__ = [
    "set_gdal_default_configuration",
    "load_gdal_dataset",
    "get_image_extension",
    "get_type_and_scales",
    "GDALCompressionOptions",
    "GDALConfigEnv",
    "GDALDigitalElevationModelTileFactory",
    "GDALImageFormats",
    "NITFDESAccessor",
    "ChippedImageInfoFacade",
    "SensorModelFactory",
    "SensorModelTypes",
]
