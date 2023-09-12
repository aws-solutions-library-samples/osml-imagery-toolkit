import logging
from typing import Any, Optional, Tuple

import numpy as np
from osgeo import gdal

from aws.osml.photogrammetry import (
    DigitalElevationModelTileFactory,
    ElevationRegionSummary,
    GDALAffineSensorModel,
    ImageCoordinate,
    geodetic_to_geocentric,
)


class GDALDigitalElevationModelTileFactory(DigitalElevationModelTileFactory):
    """
    This tile factory uses GDAL to load elevation data into numpy arrays. Any raster format supported by GDAL is
    fair game but the format must have sufficient metadata to populate the GDAL geo transform.
    """

    def __init__(self, tile_directory: str) -> None:
        """
        Constructor for the factory that takes in the root location of the elevation tiles. If the root starts with
        s3:/ then GDAL's VSIS3 virtual file system will be used to read rasters directly from cloud storage.

        :param tile_directory: the root tile location, may be an S3 URL

        :return: None
        """
        super().__init__()
        self.tile_directory = tile_directory

    def get_tile(
        self, tile_path: str
    ) -> Tuple[Optional[Any], Optional[GDALAffineSensorModel], Optional[ElevationRegionSummary]]:
        """
        Retrieve a numpy array of elevation values and a sensor model.

        TODO: Replace Any with numpy.typing.ArrayLike once we move to numpy >1.20

        :param tile_path: the location of the tile to load

        :return: an array of elevation values, a sensor model, and a summary or (None, None, None)
        """
        tile_location = f"{self.tile_directory}/{tile_path}"
        tile_location = tile_location.replace("s3:/", "/vsis3", 1)
        ds = gdal.Open(tile_location)

        # It isn't unusual for a DEM tile set to be missing tiles for regions (particularly over the ocean). If
        # the raster dataset can't be opened we'll return nothing so the client can proceed knowing that the
        # information isn't available.
        if not ds:
            logging.debug(f"No DEM tile available for {tile_path}. Checked {tile_location}")
            return None, None, None

        # If the raster exists but doesn't have a geo transform then it is likely invalid input data.
        geo_transform = ds.GetGeoTransform(can_return_null=True)
        if not geo_transform:
            logging.warning(f"DEM tile does not have geo transform metadata and can't be used: {tile_location}")
            return None, None, None

        raster_band = ds.GetRasterBand(1)
        band_as_array = raster_band.ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize)
        height, width = band_as_array.shape
        sensor_model = GDALAffineSensorModel(geo_transform)

        # Compute the distance in meters from the upper left to the lower right corner. Divide that by the distance
        # in pixels to get an approximate pixel size in meters
        ul_corner_ecf = geodetic_to_geocentric(sensor_model.image_to_world(ImageCoordinate([0, 0]))).coordinate
        lr_corner_ecf = geodetic_to_geocentric(sensor_model.image_to_world(ImageCoordinate([width, height]))).coordinate
        post_spacing = np.linalg.norm(ul_corner_ecf - lr_corner_ecf) / np.sqrt(width * width + height * height)

        stats = raster_band.GetStatistics(True, True)
        summary = ElevationRegionSummary(
            min_elevation=stats[0],
            max_elevation=stats[1],
            no_data_value=raster_band.GetNoDataValue(),
            post_spacing=post_spacing,
        )

        return band_as_array, sensor_model, summary
