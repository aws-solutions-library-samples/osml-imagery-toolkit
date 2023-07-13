import logging
import re
from typing import Dict, List, Optional, Tuple

from defusedxml import ElementTree
from osgeo import gdal, gdalconst

from aws.osml.photogrammetry import SensorModel

from .sensor_model_factory import SensorModelFactory, SensorModelTypes

logger = logging.getLogger(__name__)


def load_gdal_dataset(image_path: str) -> Tuple[gdal.Dataset, Optional[SensorModel]]:
    """
    This function loads a GDAL raster dataset from the path provided and constructs a camera model
    abstraction used to georeference locations on this image.

    :param image_path: the path to the raster data, may be a local path or a virtual file system
        (e.g. /vsis3/...)
    :return: the raster dataset and sensor model
    """
    try:
        logger.info("GDAL attempted to load image: %s", image_path)
        ds = gdal.Open(image_path)
        if ds is None:
            raise RuntimeError("GDAL Unable to load dataset and UseExceptions is not enabled.")
    except RuntimeError:
        logger.info("Skipping: %s - GDAL Unable to Process", image_path)
        raise ValueError("GDAL Unable to Load: {}".format(image_path))

    # Get a GDAL Geo Transform and any available GCPs
    geo_transform = ds.GetGeoTransform(can_return_null=True)
    ground_control_points = ds.GetGCPs()

    # If this image has NITF TREs defined parse them
    parsed_tres = None
    xml_tres = ds.GetMetadata("xml:TRE")
    if xml_tres is not None and len(xml_tres) > 0:
        parsed_tres = ElementTree.fromstring(xml_tres[0])

    # If this image has SICD Metadata parse it
    parsed_dess = None
    xml_dess = ds.GetMetadata("xml:DES")
    if xml_dess is not None and len(xml_dess) > 0:
        parsed_dess = ElementTree.fromstring(xml_dess[0])

    selected_sensor_model_types = [
        SensorModelTypes.AFFINE,
        SensorModelTypes.PROJECTIVE,
        SensorModelTypes.RPC,
        # TODO: Enable RSM model once testing complete
        # SensorModelTypes.RSM,
    ]
    # Create the best sensor model available
    sensor_model = SensorModelFactory(
        ds.RasterXSize,
        ds.RasterYSize,
        xml_tres=parsed_tres,
        xml_dess=parsed_dess,
        geo_transform=geo_transform,
        ground_control_points=ground_control_points,
        selected_sensor_model_types=selected_sensor_model_types,
    ).build()

    return ds, sensor_model


def get_type_and_scales(raster_dataset: gdal.Dataset) -> Tuple[int, List[List[int]]]:
    """
    Get type and scales of a provided raster dataset

    :param raster_dataset: the raster dataset containing the region

    :return: a tuple containing type and scales
    """
    scale_params = []
    num_bands = raster_dataset.RasterCount
    output_type = gdalconst.GDT_Byte
    min = 0
    max = 255
    for band_num in range(1, num_bands + 1):
        band = raster_dataset.GetRasterBand(band_num)
        output_type = band.DataType
        if output_type == gdalconst.GDT_Byte:
            min = 0
            max = 2**8 - 1
        elif output_type == gdalconst.GDT_UInt16:
            min = 0
            max = 2**16 - 1
        elif output_type == gdalconst.GDT_Int16:
            min = -(2**15)
            max = 2**15 - 1
        elif output_type == gdalconst.GDT_UInt32:
            min = 0
            max = 2**32 - 1
        elif output_type == gdalconst.GDT_Int32:
            min = -(2**31)
            max = 2**31 - 1
        # TODO: Add these 64-bit cases in once GDAL is upgraded to a version that supports them
        # elif output_type == gdalconst.GDT_UInt64:
        #    min = 0
        #    max = 2**64-1
        # elif output_type == gdalconst.GDT_Int64:
        #    min = -2**63
        #    max = 2**63-1
        elif output_type == gdalconst.GDT_Float32:
            min = -3.4028235e38
            max = 3.4028235e38
        elif output_type == gdalconst.GDT_Float64:
            min = -1.7976931348623157e308
            max = 1.7976931348623157e308
        else:
            logger.warning("Image uses unsupported GDAL datatype {}. Defaulting to [0,255] range".format(output_type))

        scale_params.append([min, max, min, max])

    return output_type, scale_params


def get_image_extension(image_path: str) -> str:
    """
    Get the image extension based on the provided image path

    :param image_path: an image path

    :return: image extension
    """
    possible_extensions = get_extensions_from_driver(image_path)
    selected_extension = select_extension(image_path, possible_extensions)
    image_extension = normalize_extension(selected_extension)
    logger.info("Image extension: {}".format(image_extension))
    return image_extension


def select_extension(image_path: str, possible_extensions: List[str]) -> str:
    """
    Check to see if provided image path contains a known possible extensions

    :param image_path: an image path
    :param possible_extensions: list of possible extensions

    :return: select extension
    """
    selected_extension = "UNKNOWN"
    for i, possible_extension in enumerate(possible_extensions):
        if i == 0:
            selected_extension = possible_extension.upper()
        elif f".{possible_extension}".upper() in image_path.upper():
            selected_extension = possible_extension.upper()
    return selected_extension


def normalize_extension(unnormalized_extension: str) -> str:
    """
    Convert the extension into a proper formatted string

    :param unnormalized_extension: an unnormalized extension

    :return: normalized extension
    """
    normalized_extension = unnormalized_extension.upper()
    if re.search(r"ni?tf", normalized_extension, re.IGNORECASE):
        normalized_extension = "NITF"
    elif re.search(r"tif{1,2}", normalized_extension, re.IGNORECASE):
        normalized_extension = "TIFF"
    elif re.search(r"jpe?g", normalized_extension, re.IGNORECASE):
        normalized_extension = "JPEG"
    return normalized_extension


def get_extensions_from_driver(image_path: str) -> List[str]:
    """
    Returns a list of driver extensions

    :param image_path: an image path

    :return: driver extensions
    """
    driver_extension_lookup = get_gdal_driver_extensions()
    info = gdal.Info(image_path, format="json")
    driver_long_name = info.get("driverLongName")
    return driver_extension_lookup.get(driver_long_name, [])


def get_gdal_driver_extensions() -> Dict[str, List]:
    """
    Returns a list of gdal driver extensions

    :return: gdal driver extensions
    """
    driver_lookup = {}
    for i in range(gdal.GetDriverCount()):
        drv = gdal.GetDriver(i)
        driver_name = drv.GetMetadataItem(gdal.DMD_LONGNAME)
        driver_extensions = drv.GetMetadataItem(gdal.DMD_EXTENSIONS)
        if driver_extensions:
            extension_list = driver_extensions.strip().split(" ")
            driver_lookup[driver_name] = extension_list
    return driver_lookup
