import logging
from typing import Dict, Optional

from osgeo import gdal

logger = logging.getLogger(__name__)


def set_gdal_default_configuration() -> None:
    """
    This function sets GDAL configuration options to support efficient reading of large raster
    datasets using the /vsis3 virtual file system.

    :return: None
    """
    # This is the maximum size of a chunk we can fetch at one time from a remote file
    # I couldn't find the value anywhere in the documentation, but it is enforced here:
    # https://github.com/OSGeo/gdal/blob/211e2430b8cda486d0e0e68446647f56cc0ca149/port/cpl_vsil_curl.cpp#L161
    max_curl_chunk_size = 10 * 1024 * 1024

    # For information on these options and their usage please see:
    # https://gdal.org/user/configoptions.html
    gdal_default_environment_options = {
        "VSI_CACHE": "YES",
        "VSI_CACHE_SIZE": "10000000000",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_CACHEMAX": "75%",
        "GDAL_NUM_THREADS": "1",
        "GDAL_GCPS_TO_GEOTRANSFORM_APPROX_OK": "YES",
        "GDAL_MAX_DATASET_POOL_SIZE": "1000",
        "BIGTIFF_OVERVIEW": "YES",
        "USE_TILE_AS_BLOCK": "YES",
        # This flag will setup verbose output for GDAL. In particular, it will show you each range
        # read for the file if using the /vsis3 virtual file system.
        "CPL_DEBUG": "OFF",
        "CPL_VSIL_CURL_CHUNK_SIZE": str(max_curl_chunk_size),
        "CPL_VSIL_CURL_CACHE_SIZE": str(max_curl_chunk_size * 500),
    }
    for key, val in gdal_default_environment_options.items():
        gdal.SetConfigOption(key, str(val))
    logger.info("Set GDAL Configuration Options: {}".format(gdal_default_environment_options))


class GDALConfigEnv:
    """
    This class provides a way to setup a temporary GDAL environment using Python's "with"
    statement. GDAL configuration options will be set inside the scope of the with statement and
    then reverted to previously set values on exit. This will commonly be used to set AWS security
    credentials (e.g. AWS_SECRET_ACCESS_KEY) for use by other GDAL operations.

    See: https://gdal.org/user/configoptions.html#gdal-configuration-file for additional options.
    """

    def __init__(self, options: Dict = None) -> None:
        if options:
            self.options = options.copy()
        else:
            self.options = {}
        self.old_options: Dict = {}

    def with_aws_credentials(self, aws_credentials: Optional[Dict[str, str]]) -> "GDALConfigEnv":
        """
        This method sets the GDAL configuration options for the AWS credentials from the
        credentials object returned by a boto3 call to sts.assume_role(...).

        :param aws_credentials: the dictionary of values from the sts.assume_role() response['Credentials']
        :return: self to facilitate a simple builder constructor pattern
        """
        if aws_credentials is not None:
            self.options.update(
                {
                    "AWS_SECRET_ACCESS_KEY": aws_credentials["SecretAccessKey"],
                    "AWS_ACCESS_KEY_ID": aws_credentials["AccessKeyId"],
                    "AWS_SESSION_TOKEN": aws_credentials["SessionToken"],
                }
            )
        return self

    def __enter__(self) -> None:
        for key, val in self.options.items():
            self.old_options[key] = gdal.GetConfigOption(key)
            gdal.SetConfigOption(key, str(val))

    def __exit__(self, exc_type: str, exc_val: str, exc_traceback: str) -> None:
        for key, val in self.options.items():
            gdal.SetConfigOption(key, self.old_options[key])
