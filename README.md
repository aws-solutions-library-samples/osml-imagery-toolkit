# OversightML Imagery Toolkit

The OversightML Imagery Toolkit is a Python package that contains image processing and photogrammetry routines commonly
used during the analysis of imagery collected by satellites and unmanned aerial vehicles (UAVs). It builds upon GDAL
by providing additional support for images compliant with the National Imagery Transmission Format (NITF) and Sensor
Independent Complex Data (SICD) standards.

## Installation

The intent is to vend / distribute this software through a Python Package Index.
If your environment has a distribution,
you should be able to install it using pip:
```shell
pip install osml-imagery-toolkit[gdal]
```

If you are working from a source code, you can build and install the package from the root directory of the
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

This library contains four core packages under the `aws.osml` namespace.
* photogrammetry: convert locations between the image (x, y) and geodetic (lon, lat, elev) coordinate systems
* gdal: help load and manage datasets loaded by GDAL
* image_processing: common image manipulation routines
* formats: utilities for handling format specific information; normally not accessed directly

```python
from aws.osml.gdal import GDALImageFormats, GDALCompressionOptions, load_gdal_dataset
from aws.osml.image_processing import GDALTileFactory
from aws.osml.photogrammetry import ImageCoordinate, GeodeticWorldCoordinate, SensorModel
```

### Tiling with Updated Image Metadata

Many applications break large remote sensing images into smaller chips or tiles for distributed processing or
dissemination. GDAL's Translate function provides basic capabilities, but it does not correctly update geospatial
metadata to reflect the new image extent. These utilities provide those functions so tile consumers can correctly
interpret the pixel information they have been provided. For NITF imagery that includes the addition of a new ICHIPB
TRE. With SICD the XML ImageData elements are adjusted to identify the sub-image bounds.

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

### Tiling for Display

Some images, for example 11-bit panchromatic images or SAR imagery with floating point complex data, can not be
displayed directly without remapping the pixels into an 8-bit per pixel grayscale or RGB color model. The TileFactory
supports creation of tiles suitable for human review by setting both the output_type and range_adjustment options.

```python
viz_tile_factory = GDALTileFactory(ds,
                                   sensor_model,
                                   GDALImageFormats.PNG,
                                   GDALCompressionOptions.NONE,
                                   output_type=gdalconst.GDT_Byte,
                                   range_adjustment=RangeAdjustmentType.DRA)

viz_tile = viz_tile_factory.create_encoded_tile([0, 0, 1024, 1024])
```

### More Precise Sensor Models

OversightML provides implementations of the Replacement Sensor Model (RSM), Rational Polynomial
Camera (RPC), and Sensor Independent Complex Data (SICD) sensor models to assist in geo positioning.
When loading a dataset, the toolkit will construct the most accurate sensor model
from the available image metadata. That sensor model can be used in conjunction with an optional
elevation model to convert between image and geodetic coordinates.

```python
ds, sensor_model = load_gdal_dataset("./imagery/sample.nitf")
elevation_model = DigitalElevationModel(
    SRTMTileSet(version="1arc_v3"),
    GDALDigitalElevationModelTileFactory("./local-SRTM-tiles"))

# Note the order of ImageCoordinate is (x, y)
geodetic_location_of_ul_corner = sensor_model.image_to_world(
    ImageCoordinate([0, 0]),
    elevation_model=elevation_model)

lon_degrees = -77.404453
lat_degrees = 38.954831
meters_above_ellipsoid = 100.0

image_location = sensor_model.world_to_image(
    GeodeticWorldCoordinate([radians(lon_degrees),
                             radians(lat_degrees),
                             meters_above_ellipsoid]))
```

## Contributing

This project welcomes contributions and suggestions. If you would like to submit a pull request, see our
[Contribution Guide](CONTRIBUTING.md) for more information.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
