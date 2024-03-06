#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest

import mock
from osgeo import gdal

from configuration import TEST_ENV_CONFIG


@mock.patch.dict("os.environ", TEST_ENV_CONFIG, clear=True)
class TestGDALConfig(unittest.TestCase):
    def test_gdal_config_env(self):
        from aws.osml.gdal.gdal_config import GDALConfigEnv, set_gdal_default_configuration

        set_gdal_default_configuration()  # for code coverage, setting the default values
        gdal_config = GDALConfigEnv({"Key": "Value"})  # this will get override

        gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", "DEFAULT_AWS_SECRET_ACCESS_KEY")
        gdal.SetConfigOption("AWS_ACCESS_KEY_ID", "DEFAULT_AWS_ACCESS_KEY_ID")
        gdal.SetConfigOption("AWS_SESSION_TOKEN", "DEFAULT_AWS_SESSION_TOKEN")

        with gdal_config.with_aws_credentials(
            {
                "SecretAccessKey": "TEMP_SECRET_ACCESS_KEY",
                "AccessKeyId": "TEMP_ACCESS_KEY_ID",
                "SessionToken": "TEMP_SESSION_TOKEN",
            }
        ):
            # Verify that the temporary credentials were set for the scope of this "with" statement
            assert gdal.GetConfigOption("AWS_SECRET_ACCESS_KEY") == "TEMP_SECRET_ACCESS_KEY"
            assert gdal.GetConfigOption("AWS_ACCESS_KEY_ID") == "TEMP_ACCESS_KEY_ID"
            assert gdal.GetConfigOption("AWS_SESSION_TOKEN") == "TEMP_SESSION_TOKEN"

        # Verify that the original default values are restored correctly
        assert gdal.GetConfigOption("AWS_SECRET_ACCESS_KEY") == "DEFAULT_AWS_SECRET_ACCESS_KEY"
        assert gdal.GetConfigOption("AWS_ACCESS_KEY_ID") == "DEFAULT_AWS_ACCESS_KEY_ID"
        assert gdal.GetConfigOption("AWS_SESSION_TOKEN") == "DEFAULT_AWS_SESSION_TOKEN"


if __name__ == "__main__":
    unittest.main()
