import base64
import logging
from enum import Enum
from typing import List, Optional
from xml.etree import ElementTree as ET

from osgeo import gdal

from aws.osml.photogrammetry import ChippedImageSensorModel, CompositeSensorModel, ImageCoordinate, SensorModel

from .gdal_sensor_model_builder import GDALAffineSensorModelBuilder, GDALGCPSensorModelBuilder
from .nitf_des_accessor import NITFDESAccessor
from .projective_sensor_model_builder import ProjectiveSensorModelBuilder
from .rpc_sensor_model_builder import RPCSensorModelBuilder
from .rsm_sensor_model_builder import RSMSensorModelBuilder
from .sicd_sensor_model_builder import SICDSensorModelBuilder
from .sidd_sensor_model_builder import SIDDSensorModelBuilder
from .xmltre_utils import get_tre_field_value


class ChippedImageInfoFacade:
    """
    This is a facade class that can be initialized with an ICHIPB TRE. It provides accessors for the values
    so that they can easily be used to create an ChippedImageSensorModel
    """

    def __init__(self, ichipb_tre: ET.Element) -> None:
        """
        Constructor initializes the properties from values in the TRE.

        :param ichipb_tre: the GDAL XML for the ICHIPB TRE

        :return: None
        """
        try:
            # Loop through the Output Product (OP) and Full Image (FI) fields in the ICHIPB TRE and construct
            # the corresponding image coordinates needed to create a chipped sensor model.
            self.full_image_coordinates = []
            self.chipped_image_coordinates = []
            for grid_point in ["11", "12", "21", "22"]:
                op_col = get_tre_field_value(ichipb_tre, f"OP_COL_{grid_point}", float)
                op_row = get_tre_field_value(ichipb_tre, f"OP_ROW_{grid_point}", float)
                fi_col = get_tre_field_value(ichipb_tre, f"FI_COL_{grid_point}", float)
                fi_row = get_tre_field_value(ichipb_tre, f"FI_ROW_{grid_point}", float)
                self.full_image_coordinates.append(ImageCoordinate([fi_col, fi_row]))
                self.chipped_image_coordinates.append(ImageCoordinate([op_col, op_row]))

            self.full_image_width = get_tre_field_value(ichipb_tre, "FI_COL", int)
            self.full_image_height = get_tre_field_value(ichipb_tre, "FI_ROW", int)
        except ValueError as ve:
            logging.warning("Unable to parse ICHIPB TRE found in XML metadata. SensorModel is unchanged.")
            logging.warning(str(ve))


class SensorModelTypes(Enum):
    """
    This enumeration defines the various sensor model types this factory can build.
    """

    AFFINE = "AFFINE"
    PROJECTIVE = "PROJECTIVE"
    RPC = "RPC"
    RSM = "RSM"
    SICD = "SICD"


ALL_SENSOR_MODEL_TYPES = [item for item in SensorModelTypes]


class SensorModelFactory:
    """
    This class encapsulates the logic necessary to construct SensorModels from imagery metadata parsed using GDAL.
    Users initialize the builder by providing whatever metadata is available and this class will decide how to create
    the most accurate SensorModel from the available information.
    """

    def __init__(
        self,
        actual_image_width: int,
        actual_image_height: int,
        xml_tres: Optional[ET.Element] = None,
        xml_dess: Optional[List[str]] = None,
        geo_transform: Optional[List[float]] = None,
        proj_wkt: Optional[str] = None,
        ground_control_points: Optional[List[gdal.GCP]] = None,
        selected_sensor_model_types: Optional[List[SensorModelTypes]] = None,
    ) -> None:
        """
        Construct a builder providing whatever metadata is available from the image. All of the parameters are named and
        optional allowing users to provide whatever they can and trusting that this builder will make use of as much of
        the information as possible.

        :param actual_image_width: width of the current image
        :param actual_image_height: height of the current image
        :param xml_tres: XML representing metadata in the tagged record extensions(TRE)
        :param xml_dess: XML representing data contained in the data extension segments (DES)
        :param geo_transform: a GDAL affine transform
        :param proj_wkt: the well known text string of the CRS used by the image
        :param ground_control_points: a list of GDAL GCPs that identify correspondences in the image
        :param selected_sensor_model_types: a list of sensor models that should be attempted by this factory

        :return: None
        """
        if selected_sensor_model_types is None:
            selected_sensor_model_types = ALL_SENSOR_MODEL_TYPES
        self.actual_image_width = actual_image_width
        self.actual_image_height = actual_image_height
        self.xml_tres = xml_tres
        self.xml_dess = xml_dess
        self.geo_transform = geo_transform
        self.proj_wkt = proj_wkt
        self.ground_control_points = ground_control_points
        self.selected_sensor_model_types = selected_sensor_model_types

    def build(self) -> Optional[SensorModel]:
        """
        Constructs the sensor model from the available information. Note that in cases where not enough information is
        available to provide any solution this method will return None.

        :return: the highest quality sensor model available given the information provided
        """

        approximate_sensor_model = None
        precision_sensor_model = None

        if SensorModelTypes.AFFINE in self.selected_sensor_model_types:
            if self.geo_transform is not None:
                approximate_sensor_model = GDALAffineSensorModelBuilder(self.geo_transform, self.proj_wkt).build()

        if SensorModelTypes.PROJECTIVE in self.selected_sensor_model_types:
            if self.ground_control_points is not None and len(self.ground_control_points) > 3:
                approximate_sensor_model = GDALGCPSensorModelBuilder(self.ground_control_points).build()

        if self.xml_tres is not None:
            # Start with the assumption that the raster we have is the full image. We will update this later if
            # it turns out we're working with an image chip.
            full_image_width = self.actual_image_width
            full_image_height = self.actual_image_height

            # Check to see if this image is a chip from a larger image and if so extract the chip corner information
            # from the ICHIPB TRE.
            chipped_image_info = None
            ichipb_tre = self.xml_tres.find("./tre[@name='ICHIPB']")
            if ichipb_tre is not None:
                chipped_image_info = ChippedImageInfoFacade(ichipb_tre)
                full_image_width = chipped_image_info.full_image_width
                full_image_height = chipped_image_info.full_image_height

            # Attempt to build a robust sensor model from either RSM or RPC metadata in the TREs. These
            # sensor models always reference the full image so if this is a chip we wrap the resulting sensor
            # model using information taken from ICHIPB. Note that in the unlikely event that an image has both
            # RSM and RPC metadata the RSM will be used because it has been developed as a replacement for RPC.
            precision_sensor_model = None
            if SensorModelTypes.RSM in self.selected_sensor_model_types:
                precision_sensor_model = RSMSensorModelBuilder(self.xml_tres).build()
            if precision_sensor_model is None and SensorModelTypes.RPC in self.selected_sensor_model_types:
                precision_sensor_model = RPCSensorModelBuilder(self.xml_tres).build()
            if precision_sensor_model is not None and chipped_image_info is not None:
                precision_sensor_model = ChippedImageSensorModel(
                    chipped_image_info.full_image_coordinates,
                    chipped_image_info.chipped_image_coordinates,
                    precision_sensor_model,
                )

            # Attempt to build an approximate sensor model from information in a corner coordinate TRE. The CSCRNA
            # TRE is considered more precise than IGEOLO so we will used whenever possible.
            if SensorModelTypes.PROJECTIVE in self.selected_sensor_model_types:
                cscrna_tre = self.xml_tres.find("./tre[@name='CSCRNA']")
                if cscrna_tre is not None:
                    approximate_sensor_model = ProjectiveSensorModelBuilder(
                        self.xml_tres, full_image_width, full_image_height
                    ).build()
                    if approximate_sensor_model is not None and chipped_image_info is not None:
                        approximate_sensor_model = ChippedImageSensorModel(
                            chipped_image_info.full_image_coordinates,
                            chipped_image_info.chipped_image_coordinates,
                            approximate_sensor_model,
                        )

                # TODO: Maybe create a projective sensor model from corner locations derived from the precision model
                # TODO: Consider using the rough corners from IGEOLO

        if self.xml_dess is not None and len(self.xml_dess) > 0:
            des_accessor = NITFDESAccessor(self.xml_dess)

            xml_data_content_segments = des_accessor.get_segments_by_name("XML_DATA_CONTENT")
            if xml_data_content_segments is not None:
                for xml_data_segment in xml_data_content_segments:
                    xml_bytes = des_accessor.parse_field_value(xml_data_segment, "DESDATA", base64.b64decode)
                    xml_str = xml_bytes.decode("utf-8")
                    if "SIDD" in xml_str:
                        # SIDD images will often contain SICD XML metadata as well but the SIDD should come first
                        # so we can stop processing other XML data segments
                        precision_sensor_model = SIDDSensorModelBuilder(sidd_xml=xml_str).build()
                        break
                    elif "SICD" in xml_str and SensorModelTypes.SICD in self.selected_sensor_model_types:
                        precision_sensor_model = SICDSensorModelBuilder(sicd_xml=xml_str).build()
                        break

        # If we have both an approximate and a precision sensor model return them as a composite so applications
        # can choose which model best meets their needs. If we were only able to construct one or the other then
        # return what we were able to build.
        if approximate_sensor_model is not None and precision_sensor_model is not None:
            return CompositeSensorModel(
                approximate_sensor_model=approximate_sensor_model,
                precision_sensor_model=precision_sensor_model,
            )
        elif precision_sensor_model is not None:
            return precision_sensor_model
        else:
            return approximate_sensor_model
