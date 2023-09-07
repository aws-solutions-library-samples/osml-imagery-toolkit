import unittest
from math import radians
from pathlib import Path

import numpy as np
from xsdata.formats.dataclass.parsers import XmlParser

import aws.osml.formats.sicd.models.sicd_v1_2_1 as sicd121
from aws.osml.gdal.sicd_sensor_model_builder import poly1d_to_native, poly2d_to_native, xyzpoly_to_native, xyztype_to_ndarray
from aws.osml.photogrammetry import (
    GeodeticWorldCoordinate,
    ImageCoordinate,
    INCAProjectionSet,
    PFAProjectionSet,
    PlaneProjectionSet,
    SARImageCoordConverter,
    SICDSensorModel,
    WorldCoordinate,
    geodetic_to_geocentric,
)


class TestSICDSensorModel(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_xrgycr(self):
        sicd: sicd121.SICD = XmlParser().from_path(Path("./test/data/sicd/example.sicd121.rma.xml"))

        scp_ecf = WorldCoordinate(xyztype_to_ndarray(sicd.geo_data.scp.ecf))
        scp_pixel = ImageCoordinate([sicd.image_data.scppixel.col, sicd.image_data.scppixel.row])

        time_coa_poly = poly2d_to_native(sicd.grid.time_coapoly)
        arp_poly = xyzpoly_to_native(sicd.position.arppoly)

        first_pixel = ImageCoordinate([sicd.image_data.first_col, sicd.image_data.first_row])

        coord_converter = SARImageCoordConverter(
            scp_pixel=scp_pixel,
            scp_ecf=scp_ecf,
            u_row=xyztype_to_ndarray(sicd.grid.row.uvect_ecf),
            u_col=xyztype_to_ndarray(sicd.grid.col.uvect_ecf),
            row_ss=sicd.grid.row.ss,
            col_ss=sicd.grid.col.ss,
            first_pixel=first_pixel,
        )

        projection_set = PlaneProjectionSet(
            scp_ecf=scp_ecf,
            image_plane_urow=xyztype_to_ndarray(sicd.grid.row.uvect_ecf),
            image_plane_ucol=xyztype_to_ndarray(sicd.grid.col.uvect_ecf),
            coa_time_poly=time_coa_poly,
            arp_poly=arp_poly,
        )

        sicd_sensor_model = SICDSensorModel(
            coord_converter=coord_converter,
            coa_projection_set=projection_set,
            scp_arp=xyztype_to_ndarray(sicd.scpcoa.arppos),
            scp_varp=xyztype_to_ndarray(sicd.scpcoa.arpvel),
            side_of_track=str(sicd.scpcoa.side_of_track.value),
        )

        geodetic_world_coordinate = sicd_sensor_model.image_to_world(
            ImageCoordinate([sicd.image_data.scppixel.col, sicd.image_data.scppixel.row])
        )
        ecf_world_coordinate = geodetic_to_geocentric(geodetic_world_coordinate)

        assert np.allclose(ecf_world_coordinate.coordinate, scp_ecf.coordinate)

        geo_scp_world_coordinate = GeodeticWorldCoordinate(
            [radians(sicd.geo_data.scp.llh.lon), radians(sicd.geo_data.scp.llh.lat), sicd.geo_data.scp.llh.hae]
        )

        assert np.allclose(geo_scp_world_coordinate.coordinate, geodetic_world_coordinate.coordinate)

        calculated_image_scp = sicd_sensor_model.world_to_image(geo_scp_world_coordinate)

        assert np.allclose(calculated_image_scp.coordinate, scp_pixel.coordinate)

    def test_rgzero_inca(self):
        sicd: sicd121.SICD = XmlParser().from_path(Path("./test/data/sicd/example.sicd121.capella.xml"))

        scp_ecf = WorldCoordinate(xyztype_to_ndarray(sicd.geo_data.scp.ecf))
        scp_pixel = ImageCoordinate([sicd.image_data.scppixel.col, sicd.image_data.scppixel.row])

        time_coa_poly = poly2d_to_native(sicd.grid.time_coapoly)
        arp_poly = xyzpoly_to_native(sicd.position.arppoly)

        first_pixel = ImageCoordinate([sicd.image_data.first_col, sicd.image_data.first_row])

        image_plane = SARImageCoordConverter(
            scp_pixel=scp_pixel,
            scp_ecf=scp_ecf,
            u_row=xyztype_to_ndarray(sicd.grid.row.uvect_ecf),
            u_col=xyztype_to_ndarray(sicd.grid.col.uvect_ecf),
            row_ss=sicd.grid.row.ss,
            col_ss=sicd.grid.col.ss,
            first_pixel=first_pixel,
        )

        projection_set = INCAProjectionSet(
            r_ca_scp=sicd.rma.inca.r_ca_scp,
            inca_time_coa_poly=poly1d_to_native(sicd.rma.inca.time_capoly),
            drate_sf_poly=poly2d_to_native(sicd.rma.inca.drate_sfpoly),
            coa_time_poly=time_coa_poly,
            arp_poly=arp_poly,
        )

        sicd_sensor_model = SICDSensorModel(
            coord_converter=image_plane,
            coa_projection_set=projection_set,
            scp_arp=xyztype_to_ndarray(sicd.scpcoa.arppos),
            scp_varp=xyztype_to_ndarray(sicd.scpcoa.arpvel),
            side_of_track=str(sicd.scpcoa.side_of_track.value),
        )

        geodetic_world_coordinate = sicd_sensor_model.image_to_world(
            ImageCoordinate([sicd.image_data.scppixel.col, sicd.image_data.scppixel.row])
        )
        ecf_world_coordinate = geodetic_to_geocentric(geodetic_world_coordinate)

        assert np.allclose(ecf_world_coordinate.coordinate, scp_ecf.coordinate)

        geo_scp_world_coordinate = GeodeticWorldCoordinate(
            [radians(sicd.geo_data.scp.llh.lon), radians(sicd.geo_data.scp.llh.lat), sicd.geo_data.scp.llh.hae]
        )

        assert np.allclose(geo_scp_world_coordinate.coordinate, geodetic_world_coordinate.coordinate)

        calculated_image_scp = sicd_sensor_model.world_to_image(geo_scp_world_coordinate)

        assert np.allclose(calculated_image_scp.coordinate, scp_pixel.coordinate)

    def test_rgazim_pfa(self):
        sicd: sicd121.SICD = XmlParser().from_path(Path("./test/data/sicd/example.sicd121.pfa.xml"))

        scp_ecf = WorldCoordinate(xyztype_to_ndarray(sicd.geo_data.scp.ecf))
        scp_pixel = ImageCoordinate([sicd.image_data.scppixel.col, sicd.image_data.scppixel.row])

        polar_ang_poly = poly1d_to_native(sicd.pfa.polar_ang_poly)
        spatial_freq_sf_poly = poly1d_to_native(sicd.pfa.spatial_freq_sfpoly)
        time_coa_poly = poly2d_to_native(sicd.grid.time_coapoly)
        arp_poly = xyzpoly_to_native(sicd.position.arppoly)

        first_pixel = ImageCoordinate([sicd.image_data.first_col, sicd.image_data.first_row])

        image_plane = SARImageCoordConverter(
            scp_pixel=scp_pixel,
            scp_ecf=scp_ecf,
            u_row=xyztype_to_ndarray(sicd.grid.row.uvect_ecf),
            u_col=xyztype_to_ndarray(sicd.grid.col.uvect_ecf),
            row_ss=sicd.grid.row.ss,
            col_ss=sicd.grid.col.ss,
            first_pixel=first_pixel,
        )

        projection_set = PFAProjectionSet(
            scp_ecf=scp_ecf,
            polar_ang_poly=polar_ang_poly,
            spatial_freq_sf_poly=spatial_freq_sf_poly,
            coa_time_poly=time_coa_poly,
            arp_poly=arp_poly,
        )

        # FPN is the default ground plane normal for a PFA projection otherwise we calculate it as a normal
        # from WGS84 ellipsoid
        ugpn = xyztype_to_ndarray(sicd.pfa.fpn)

        sicd_sensor_model = SICDSensorModel(
            coord_converter=image_plane,
            coa_projection_set=projection_set,
            scp_arp=xyztype_to_ndarray(sicd.scpcoa.arppos),
            scp_varp=xyztype_to_ndarray(sicd.scpcoa.arpvel),
            side_of_track=str(sicd.scpcoa.side_of_track.value),
            u_gpn=ugpn,
        )

        geodetic_world_coordinate = sicd_sensor_model.image_to_world(
            ImageCoordinate([sicd.image_data.scppixel.col, sicd.image_data.scppixel.row])
        )
        ecf_world_coordinate = geodetic_to_geocentric(geodetic_world_coordinate)

        assert np.allclose(ecf_world_coordinate.coordinate, scp_ecf.coordinate)

        geo_scp_world_coordinate = GeodeticWorldCoordinate(
            [radians(sicd.geo_data.scp.llh.lon), radians(sicd.geo_data.scp.llh.lat), sicd.geo_data.scp.llh.hae]
        )

        assert np.allclose(geo_scp_world_coordinate.coordinate, geodetic_world_coordinate.coordinate)

        calculated_image_scp = sicd_sensor_model.world_to_image(geo_scp_world_coordinate)

        assert np.allclose(calculated_image_scp.coordinate, scp_pixel.coordinate)
