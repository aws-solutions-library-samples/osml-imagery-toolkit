import logging
import re
from typing import Dict, List, Optional, Tuple

from defusedxml import ElementTree
from osgeo import gdal, gdalconst

from aws.osml.photogrammetry import SensorModel

from .dynamic_range_adjustment import DRAParameters
from .sensor_model_factory import SensorModelFactory, SensorModelTypes
from .typing import RangeAdjustmentType

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
    proj_wkt = ds.GetProjection()

    # If this image has NITF TREs defined parse them
    parsed_tres = None
    xml_tres = ds.GetMetadata("xml:TRE")
    if xml_tres is not None and len(xml_tres) > 0:
        parsed_tres = ElementTree.fromstring(xml_tres[0])

    # If this image has NITF DES segments read them
    xml_dess = ds.GetMetadata("xml:DES")

    selected_sensor_model_types = [
        SensorModelTypes.AFFINE,
        SensorModelTypes.PROJECTIVE,
        SensorModelTypes.RPC,
        SensorModelTypes.SICD,
        SensorModelTypes.RSM,
    ]
    # Create the best sensor model available
    sensor_model = SensorModelFactory(
        ds.RasterXSize,
        ds.RasterYSize,
        xml_tres=parsed_tres,
        xml_dess=xml_dess,
        geo_transform=geo_transform,
        proj_wkt=proj_wkt,
        ground_control_points=ground_control_points,
        selected_sensor_model_types=selected_sensor_model_types,
    ).build()

    return ds, sensor_model


def get_minmax_for_type(gdal_type: int) -> Tuple[float, float]:
    """
    This function computes the minimum and maximum values that can be stored in a given GDAL pixel type.

    :param gdal_type: the pixel type
    :return: tuple of min, max values
    """
    min_value = 0
    max_value = 255
    if gdal_type == gdalconst.GDT_Byte:
        min_value = 0
        max_value = 2**8 - 1
    elif gdal_type == gdalconst.GDT_UInt16:
        min_value = 0
        max_value = 2**16 - 1
    elif gdal_type == gdalconst.GDT_Int16:
        min_value = -(2**15)
        max_value = 2**15 - 1
    elif gdal_type == gdalconst.GDT_UInt32:
        min_value = 0
        max_value = 2**32 - 1
    elif gdal_type == gdalconst.GDT_Int32:
        min_value = -(2**31)
        max_value = 2**31 - 1
    elif gdal_type == gdalconst.GDT_UInt64:
        min_value = 0
        max_value = 2**64 - 1
    elif gdal_type == gdalconst.GDT_Int64:
        min_value = -(2**63)
        max_value = 2**63 - 1
    elif gdal_type == gdalconst.GDT_Float32:
        min_value = -3.4028235e38
        max_value = 3.4028235e38
    elif gdal_type == gdalconst.GDT_Float64:
        min_value = -1.7976931348623157e308
        max_value = 1.7976931348623157e308
    else:
        logger.warning("Image uses unsupported GDAL datatype {}. Defaulting to [0,255] range".format(gdal_type))

    return min_value, max_value


def get_type_and_scales(
    raster_dataset: gdal.Dataset,
    desired_output_type: Optional[int] = None,
    range_adjustment: RangeAdjustmentType = RangeAdjustmentType.NONE,
) -> Tuple[int, List[List[int]]]:
    """
    Get type and scales of a provided raster dataset

    :param raster_dataset: the raster dataset containing the region
    :param desired_output_type: type to be output after dynamic range adjustments
    :param range_adjustment: the type of pixel scaling effort to

    :return: a tuple containing type and scales
    """
    scale_params = []
    output_type = gdalconst.GDT_Byte
    num_bands = raster_dataset.RasterCount
    for band_num in range(1, num_bands + 1):
        band = raster_dataset.GetRasterBand(band_num)
        band_type = band.DataType
        min_value, max_value = get_minmax_for_type(band_type)

        if desired_output_type is None:
            output_type = band_type
            output_min = min_value
            output_max = max_value
        else:
            output_type = desired_output_type
            output_min, output_max = get_minmax_for_type(desired_output_type)

        # If a range adjustment is requested compute the range of source pixel values that will be mapped to the full
        # output range.
        selected_min = min_value
        selected_max = max_value
        if range_adjustment is not RangeAdjustmentType.NONE:
            # GetStatistics(1,1) means it is ok to approximate but force computation is stats not already available
            stats = band.GetStatistics(1, 1)
            min_value = stats[0]
            max_value = stats[1]

            num_buckets = int(max_value - min_value)
            if band_type == gdalconst.GDT_Float32 or band_type == gdalconst.GDT_Float64:
                num_buckets = 255

            dra = DRAParameters.from_counts(
                counts=band.GetHistogram(
                    buckets=num_buckets, max=max_value, min=min_value, include_out_of_range=1, approx_ok=1
                ),
                first_bucket_value=min_value,
                last_bucket_value=max_value,
            )

            if range_adjustment == RangeAdjustmentType.DRA:
                selected_min = dra.suggested_min_value
                selected_max = dra.suggested_max_value
            elif range_adjustment == RangeAdjustmentType.MINMAX:
                selected_min = dra.actual_min_value
                selected_max = dra.actual_max_value
            else:
                logger.warning(f"Unknown range adjustment selected {range_adjustment}. Skipping.")

        band_scale_parameters = [selected_min, selected_max, output_min, output_max]
        scale_params.append(band_scale_parameters)

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
