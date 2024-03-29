#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""This file was generated by xsdata, v23.8, on 2023-10-05 09:59:45

Generator: DataclassGenerator
See: https://xsdata.readthedocs.io/
"""
from enum import Enum

__NAMESPACE__ = "urn:us:gov:ic:cvenum:ism:dissem"


class CVEnumISMDissemValues(Enum):
    """(U) All currently valid Dissemination controls from the published register
    PERMISSIBLE VALUES
    The permissible values for this simple type are defined in the Controlled Value Enumeration:
    CVEnumISMDissem.xml

    :cvar RS: RISK SENSITIVE
    :cvar FOUO: FOR OFFICIAL USE ONLY
    :cvar OC: ORIGINATOR CONTROLLED
    :cvar OC_USGOV: ORIGINATOR CONTROLLED US GOVERNMENT
    :cvar IMC: CONTROLLED IMAGERY
    :cvar NF: NOT RELEASABLE TO FOREIGN NATIONALS
    :cvar PR: CAUTION-PROPRIETARY INFORMATION INVOLVED
    :cvar REL: AUTHORIZED FOR RELEASE TO
    :cvar RELIDO: RELEASABLE BY INFORMATION DISCLOSURE OFFICIAL
    :cvar EYES: EYES ONLY
    :cvar DSEN: DEA SENSITIVE
    :cvar FISA: FOREIGN INTELLIGENCE SURVEILLANCE ACT
    :cvar DISPLAYONLY: AUTHORIZED FOR DISPLAY BUT NOT RELEASE TO
    """

    RS = "RS"
    FOUO = "FOUO"
    OC = "OC"
    OC_USGOV = "OC-USGOV"
    IMC = "IMC"
    NF = "NF"
    PR = "PR"
    REL = "REL"
    RELIDO = "RELIDO"
    EYES = "EYES"
    DSEN = "DSEN"
    FISA = "FISA"
    DISPLAYONLY = "DISPLAYONLY"
