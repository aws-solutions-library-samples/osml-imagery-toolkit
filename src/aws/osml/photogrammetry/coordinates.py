import numpy as np
import numpy.typing as npt
import pyproj
from pyproj.enums import TransformDirection


class WorldCoordinate:
    """
    A world coordinate is a vector representing a position in 3D space. The ground coordinate system specified is
    either Geodetic (latitude, longitude, and height above the WGS 84 reference ellipsoid), or Rectangular (cartesian
    coordinates in reference to a local tangent plane). Regardless whether the coordinate system is specified as
    Geodetic or Rectangular, associated ground point locations are represented as a triple – x, y, and z.

    A Rectangular system should be specified when the image footprint is near the earth’s North or South Pole. Either
    a Rectangular or Geodetic system can be specified when the footprint is near 180 degrees East longitude. However,
    if Geodetic, the range for longitude is then specified as (0,2pi) radians instead of the usual (-pi, +pi) radians.
    """

    def __init__(self, coordinate: npt.ArrayLike = None) -> None:
        """
        Constructs a world coordinate from an x, y, z triple. The triple can be expressed as a List or any other
        structure that can be used to construct a Numpy array.

        :param coordinate: the x,y,z components

        :return: None
        """
        if coordinate is None:
            coordinate = [0.0, 0.0, 0.0]

        if len(coordinate) != 3:
            raise ValueError("WorldCoordinates must have 3 components (x,y,z)")

        self.coordinate = np.array(coordinate, dtype=np.float64)

    @property
    def x(self) -> float:
        return self.coordinate[0]

    @x.setter
    def x(self, value: float) -> None:
        self.coordinate[0] = value

    @property
    def y(self) -> float:
        return self.coordinate[1]

    @y.setter
    def y(self, value: float) -> None:
        self.coordinate[1] = value

    @property
    def z(self) -> float:
        return self.coordinate[2]

    @z.setter
    def z(self, value: float) -> None:
        self.coordinate[2] = value


class GeodeticWorldCoordinate(WorldCoordinate):
    """
    A GeodeticWorldCoordinate is an WorldCoordinate where the x,y,z components can be interpreted as longitude,
    latitude, and elevation.
    """

    def __init__(self, coordinate: npt.ArrayLike = None) -> None:
        """
        Constructs a geodetic world coordinate from a longitude, latitude, elevation triple. The longitude and
        latitude components are in radians. The triple can be expressed as a List or any other structure that can
        be used to construct a Numpy array.

        :param coordinate: the longitude, latitude, elevation components

        :return: None
        """
        super().__init__(coordinate)

    @property
    def longitude(self) -> float:
        return self.x

    @longitude.setter
    def longitude(self, value: float) -> None:
        self.x = value

    @property
    def latitude(self) -> float:
        return self.y

    @latitude.setter
    def latitude(self, value: float) -> None:
        self.y = value

    @property
    def elevation(self) -> float:
        return self.z

    @elevation.setter
    def elevation(self, value: float) -> None:
        self.z = value

    def to_dms_string(self) -> str:
        """
        Outputs this coordinate in a format ddmmssXdddmmssY. The first half, ddmmssX, represents degrees, minutes, and
        seconds of latitude with X representing North or South (N for North, S for South). The second half, dddmmssY,
        represents degrees, minutes, and seconds of longitude with Y representing East or West (E for East, W for West),
        respectively.

        :return: the formatted coordinate string
        """
        result_parts = []
        lat_degrees = np.degrees(self.latitude)
        direction = "N"
        if lat_degrees < 0:
            lat_degrees *= -1.0
            direction = "S"
        d = int(lat_degrees)
        m = int(round(lat_degrees - d, 6) * 60)
        s = int(round(lat_degrees - d - m / 60, 6) * 3600)
        result_parts.append(format(d, "02d"))
        result_parts.append(format(m, "02d"))
        result_parts.append(format(s, "02d"))
        result_parts.append(direction)

        lon_degrees = np.degrees(self.longitude)
        direction = "E"
        if lon_degrees < 0:
            lon_degrees *= -1.0
            direction = "W"
        d = int(lon_degrees)
        m = int(round(lon_degrees - d, 6) * 60)
        s = int(round(lon_degrees - d - m / 60, 6) * 3600)
        result_parts.append(format(d, "03d"))
        result_parts.append(format(m, "02d"))
        result_parts.append(format(s, "02d"))
        result_parts.append(direction)

        return "".join(result_parts)


# These are common definitions of projections used by Pyproj. They are used when converting between an Earth Centered
# Earth Fixed (ECEF or geocentric) coordinate system that uses cartesian coordinates and a longitude, latitude based
# geographic coordinate system. Both of these systems use the WGS84 datum which is a widely used standard among our
# customers
ECEF_PROJ = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
LLA_PROJ = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
GEODETIC_TO_GEOCENTRIC_TRANSFORM = pyproj.Transformer.from_proj(LLA_PROJ, ECEF_PROJ)


def geocentric_to_geodetic(ecef_world_coordinate: WorldCoordinate) -> GeodeticWorldCoordinate:
    """
    Converts a ECEF world coordinate (x, y, z) in meters into a (longitude, latitude, elevation) geodetic coordinate
    with units of radians, radians, meters.

    :param ecef_world_coordinate: the geocentric coordinate

    :return: the geodetic coordinate
    """
    return GeodeticWorldCoordinate(
        GEODETIC_TO_GEOCENTRIC_TRANSFORM.transform(
            ecef_world_coordinate.x,
            ecef_world_coordinate.y,
            ecef_world_coordinate.z,
            radians=True,
            direction=TransformDirection.INVERSE,
        )
    )


def geodetic_to_geocentric(geodetic_coordinate: GeodeticWorldCoordinate) -> WorldCoordinate:
    """
    Converts a geodetic world coordinate (longitude, latitude, elevation) with units of radians, radians, meters into
    a ECEF / geocentric world coordinate (x, y, z) in meters.

    :param geodetic_coordinate: the geodetic coordinate

    :return: the geocentric coordinate
    """
    return WorldCoordinate(
        GEODETIC_TO_GEOCENTRIC_TRANSFORM.transform(
            geodetic_coordinate.longitude,
            geodetic_coordinate.latitude,
            geodetic_coordinate.elevation,
            radians=True,
            direction=TransformDirection.FORWARD,
        )
    )


class ImageCoordinate:
    """
    This image coordinate system convention is defined as follows. The upper left corner of the upper left pixel
    of the original full image has continuous image coordinates (pixel position) (r, c) = (0.0,0.0) , and the center
    of the upper left pixel has continuous image coordinates (r, c) = (0.5,0.5) . The first row of the original full
    image has discrete image row coordinate R = 0 , and corresponds to a range of continuous image row coordinates of
    r = [0,1) . The first column of the original full image has discrete image column coordinate C = 0 , and
    corresponds to a range of continuous image column coordinates of c = [0,1) . Thus, for example, continuous image
    coordinates (r, c) = (5.6,8.3) correspond to the sixth row and ninth column of the original full image, and
    discrete image coordinates (R,C) = (5,8).
    """

    def __init__(self, coordinate: npt.ArrayLike = None) -> None:
        """
        Constructs an image coordinate from an x, y tuple. The tuple can be expressed as a List or any other
        structure that can be used to construct a Numpy array.

        :param coordinate: the x, y components

        :return: None
        """
        if coordinate is None:
            coordinate = [0.0, 0.0]

        if len(coordinate) != 2:
            raise ValueError("ImageCoordinates must have 2 components (x,y)")

        self.coordinate = np.array(coordinate, dtype=np.float64)

    @property
    def c(self) -> float:
        return self.coordinate[0]

    @c.setter
    def c(self, value: float) -> None:
        self.coordinate[0] = value

    @property
    def r(self) -> float:
        return self.coordinate[1]

    @r.setter
    def r(self, value: float) -> None:
        self.coordinate[1] = value

    @property
    def x(self) -> float:
        return self.c

    @x.setter
    def x(self, value: float) -> None:
        self.c = value

    @property
    def y(self) -> float:
        return self.r

    @y.setter
    def y(self, value: float) -> None:
        self.r = value
