# OversightML Imagery Core Libraries

The OversightML Imagery Core is a Python package that contains image processing and photogrammetry routines commonly
used during the analysis of imagery collected by satellites and unmanned aerial vehicles (UAVs). It builds upon GDAL
by providing additional support for images compliant with the Sensor Independent Complex Data (SICD) and National 
Imagery Transmission Format (NITF) standards.

## Installation

The intent is to vend / distribute this software through a Python Package Index. If your environment has a distribution 
you should be able to install it using pip:
```shell
pip install aws-osml-imagery-core[gdal]
```

If you are working from a source code you can build and install the package from the root directory of the
distribution.
```shell
pip install .[gdal]
```
Note that GDAL is listed as an extra dependency for this package. This is done to facilitate environments that either
don't want to use GDAL or those that have their own custom installation steps for that library. Future versions of
this package will include image IO backbones that have fewer dependencies.

## Documentation

You can find documentation for this library in the `./doc` directory. Sphinx is used to construct a searchable HTML
version of the API documents.

```shell
tox -e docs
```

## Example Usage

This library contains three core packages under the `aws.osml` namespace. 
* photogrammetry: convert locations between the image (x, y) and geodetic (lon, lat, elev) coordinate systems
* gdal: help load and manage datasets loaded by GDAL
* image_processing: common image manipulation routines

```python
from aws.osml.gdal import GDALImageFormats, GDALCompressionOptions, load_gdal_dataset
from aws.osml.image_processing import GDALTileFactory
from aws.osml.photogrammetry import ImageCoordinate, GeodeticWorldCoordinate, SensorModel
```

### Tiling with Updated Image Metadata

Many applications break large remote sensing images into smaller chips or tiles for distributed processing or 
dissemination. GDAL's Translate function provides basic capabilities but it does not correctly update geospatial
metadata to reflect the new image extent. These utilities provide those functions so tile consumers can correctly 
interpret the pixel information they have been provided.

```python
# Load the image and create a sensor model
ds, sensor_model = load_gdal_dataset("./imagery/sample.nitf")
tile_factory = GDALTileFactory(ds,
                               sensor_model,
                               GDALImageFormats.NITF, 
                               GDALCompressionOptions.NONE 
                               )

# Bounds are [left_x, top_y, width, height]
nitf_encoded_tile_bytes = tile_factory.create_encoded_tile([0, 0, 1024, 1024])
```

### More Precise Sensor Models

OversightML provides implementations of the Replacement Sensor Model (RSM) and Rational Polynomial Camera (RPC) sensor 
models to assist in geo positioning. When loading a dataset you will automatically get the most accurate sensor model
from the available image metadata. That sensor model can be used in conjunction with an optional elevation model to 
convert between image and geodetic coordinates.

```python
ds, sensor_model = load_gdal_dataset("./imagery/sample.nitf")
elevation_model = DigitalElevationModel(SRTMTileSet(version="1arc_v3"),
                                        GDALDigitalElevationModelTileFactory("./local-SRTM-tiles")
                                        )

geodetic_location_of_ul_corner = sensor_model.image_to_world(ImageCoordinate([0, 0]), elevation_model=elevation_model)

lon_degrees = -77.404453
lat_degrees = 38.954831
meters_above_ellipsoid = 100.0
image_location = sensor_model.world_to_image(GeodeticWorldCoordinate([radians(lon_degrees), 
                                                                      radians(lat_degrees), 
                                                                      meters_above_ellipsoid]))
```

## Contributing

This project welcomes contributions and suggestions. If you would like to submit a pull request, see our 
[Contribution Guide](CONTRIBUTING.md) for more information.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.