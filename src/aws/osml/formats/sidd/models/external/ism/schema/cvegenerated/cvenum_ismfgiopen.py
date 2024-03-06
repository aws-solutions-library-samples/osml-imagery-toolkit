#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""This file was generated by xsdata, v23.8, on 2023-10-05 09:59:45

Generator: DataclassGenerator
See: https://xsdata.readthedocs.io/
"""
from enum import Enum

__NAMESPACE__ = "urn:us:gov:ic:ism-cvenum"


class CVEnumISMFGIOpenValues(Enum):
    """(U) UNKNOWN followed by all currently valid ISO Trigraphs except USA in
    alphabetical order by Trigraph, followed by all currently valid CAPCO Coalition
    tetragraphs in alphabetical order by tetragraph.

    PERMISSIBLE VALUES
    The permissible values for this simple type are defined in the Controlled Value Enumeration:
    CVEnumISMFGIOpen.xml

    :cvar UNKNOWN: Unknown
    :cvar ABW: Trigraph for Aruba
    :cvar AFG: Trigraph for Afghanistan
    :cvar AGO: Trigraph for Angola
    :cvar AIA: Trigraph for Anguilla
    :cvar ALA: Trigraph for Åland Islands
    :cvar ALB: Trigraph for Albania
    :cvar AND: Trigraph for Andorra
    :cvar ANT: Trigraph for Netherlands Antilles
    :cvar ARE: Trigraph for United Arab Emirates
    :cvar ARG: Trigraph for Argentina
    :cvar ARM: Trigraph for Armenia
    :cvar ASM: Trigraph for American Samoa
    :cvar ATA: Trigraph for Antarctica
    :cvar ATF: Trigraph for French Southern Territories
    :cvar ATG: Trigraph for Antigua and Barbuda
    :cvar AUS: Trigraph for Australia
    :cvar AUT: Trigraph for Austria
    :cvar AZE: Trigraph for Azerbaijan
    :cvar BDI: Trigraph for Burundi
    :cvar BEL: Trigraph for Belgium
    :cvar BEN: Trigraph for Benin
    :cvar BFA: Trigraph for Burkina Faso
    :cvar BGD: Trigraph for Bangladesh
    :cvar BGR: Trigraph for Bulgaria
    :cvar BHR: Trigraph for Bahrain
    :cvar BHS: Trigraph for Bahamas
    :cvar BIH: Trigraph for Bosnia and Herzegovina
    :cvar BLM: Trigraph for Saint Barthélemy
    :cvar BLR: Trigraph for Belarus
    :cvar BLZ: Trigraph for Belize
    :cvar BMU: Trigraph for Bermuda
    :cvar BOL: Trigraph for Bolivia
    :cvar BRA: Trigraph for Brazil
    :cvar BRB: Trigraph for Barbados
    :cvar BRN: Trigraph for Brunei Darussalam
    :cvar BTN: Trigraph for Bhutan
    :cvar BVT: Trigraph for Bouvet Island
    :cvar BWA: Trigraph for Botswana
    :cvar CAF: Trigraph for Central African Republic
    :cvar CAN: Trigraph for Canada
    :cvar CCK: Trigraph for Cocos (Keeling) Islands
    :cvar CHE: Trigraph for Switzerland
    :cvar CHL: Trigraph for Chile
    :cvar CHN: Trigraph for China
    :cvar CIV: Trigraph for Côte d'Ivoire
    :cvar CMR: Trigraph for Cameroon
    :cvar COD: Trigraph for Congo, The Democratic Republic of the
    :cvar COG: Trigraph for Congo
    :cvar COK: Trigraph for Cook Islands
    :cvar COL: Trigraph for Colombia
    :cvar COM: Trigraph for Comoros
    :cvar CPV: Trigraph for Cape Verde
    :cvar CRI: Trigraph for Costa Rica
    :cvar CUB: Trigraph for Cuba
    :cvar CXR: Trigraph for Christmas Island
    :cvar CYM: Trigraph for Cayman Islands
    :cvar CYP: Trigraph for Cyprus
    :cvar CZE: Trigraph for Czech Republic
    :cvar DEU: Trigraph for Germany
    :cvar DJI: Trigraph for Djibouti
    :cvar DMA: Trigraph for Dominica
    :cvar DNK: Trigraph for Denmark
    :cvar DOM: Trigraph for Dominican Republic
    :cvar DZA: Trigraph for Algeria
    :cvar ECU: Trigraph for Eucador
    :cvar EGY: Trigraph for Egypt
    :cvar ERI: Trigraph for Eritrea
    :cvar ESH: Trigraph for Western Sahara
    :cvar ESP: Trigraph for Spain
    :cvar EST: Trigraph for Estonia
    :cvar ETH: Trigraph for Ethiopia
    :cvar FIN: Trigraph for Finland
    :cvar FJI: Trigraph for Fiji
    :cvar FLK: Trigraph for Falkland Islands (Malvinas)
    :cvar FRA: Trigraph for France
    :cvar FRO: Trigraph for Faroe Islands
    :cvar FSM: Trigraph for Micronesia, Federated States of
    :cvar GAB: Trigraph for Gabon
    :cvar GBR: Trigraph for United Kingdom
    :cvar GEO: Trigraph for Georgia
    :cvar GGY: Trigraph for Guernsey
    :cvar GHA: Trigraph for Ghana
    :cvar GIB: Trigraph for Gibraltar
    :cvar GIN: Trigraph for Guinea
    :cvar GLP: Trigraph for Guadeloupe
    :cvar GMB: Trigraph for Gambia
    :cvar GNB: Trigraph for Guinea-Bissau
    :cvar GNQ: Trigraph for Equatorial Guinea
    :cvar GRC: Trigraph for Greece
    :cvar GRD: Trigraph for Grenada
    :cvar GRL: Trigraph for Greenland
    :cvar GTM: Trigraph for Guatemala
    :cvar GUF: Trigraph for French Guiana
    :cvar GUM: Trigraph for Guam
    :cvar GUY: Trigraph for Guyana
    :cvar HKG: Trigraph for Hong Kong
    :cvar HMD: Trigraph for Heard Island and McDonald Islands
    :cvar HND: Trigraph for Honduras
    :cvar HRV: Trigraph for Croatia
    :cvar HTI: Trigraph for Haiti
    :cvar HUN: Trigraph for Hungary
    :cvar IDN: Trigraph for Indonesia
    :cvar IMN: Trigraph for Isle of Man
    :cvar IND: Trigraph for India
    :cvar IOT: Trigraph for British Indian Ocean Territory
    :cvar IRL: Trigraph for Ireland
    :cvar IRN: Trigraph for Iran, Islamic Republic of
    :cvar IRQ: Trigraph for Iraq
    :cvar ISL: Trigraph for Iceland
    :cvar ISR: Trigraph for Israel
    :cvar ITA: Trigraph for Italy
    :cvar JAM: Trigraph for Jamaica
    :cvar JEY: Trigraph for Jersey
    :cvar JOR: Trigraph for Jordan
    :cvar JPN: Trigraph for Japan
    :cvar KAZ: Trigraph for Kazakhstan
    :cvar KEN: Trigraph for Kenya
    :cvar KGZ: Trigraph for Kyrgyzstan
    :cvar KHM: Trigraph for Cambodia
    :cvar KIR: Trigraph for Kiribati
    :cvar KNA: Trigraph for Saint Kitts and Nevis
    :cvar KOR: Trigraph for Korea, Republic of
    :cvar KWT: Trigraph for Kuwait
    :cvar LAO: Trigraph for Lao People's Democratic Republic
    :cvar LBN: Trigraph for Lebanon
    :cvar LBR: Trigraph for Liberia
    :cvar LBY: Trigraph for Libyan Arab Jamahiriya
    :cvar LCA: Trigraph for Saint Lucia
    :cvar LIE: Trigraph for Liechtenstein
    :cvar LKA: Trigraph for Sri Lanka
    :cvar LSO: Trigraph for Lesotho
    :cvar LTU: Trigraph for Lithuania
    :cvar LUX: Trigraph for Luxembourg
    :cvar LVA: Trigraph for Latvia
    :cvar MAC: Trigraph for Macao
    :cvar MAF: Trigraph for Saint Martin (French part)
    :cvar MAR: Trigraph for Morocco
    :cvar MCO: Trigraph for Monaco
    :cvar MDA: Trigraph for Moldova (the Republic of)
    :cvar MDG: Trigraph for Madagascar
    :cvar MDV: Trigraph for Maldives
    :cvar MEX: Trigraph for Mexico
    :cvar MHL: Trigraph for Marshall Islands
    :cvar MKD: Trigraph for Macedonia, The former Yugoslav Republic of
    :cvar MLI: Trigraph for Mali
    :cvar MLT: Trigraph for Malta
    :cvar MMR: Trigraph for Myanmar
    :cvar MNE: Trigraph for Montenegro
    :cvar MNG: Trigraph for Mongolia
    :cvar MNP: Trigraph for Northern Mariana Islands
    :cvar MOZ: Trigraph for Mozambique
    :cvar MRT: Trigraph for Mauritania
    :cvar MSR: Trigraph for Montserrat
    :cvar MTQ: Trigraph for Martinique
    :cvar MUS: Trigraph for Mauritius
    :cvar MWI: Trigraph for Malawi
    :cvar MYS: Trigraph for Malaysia
    :cvar MYT: Trigraph for Mayotte
    :cvar NAM: Trigraph for Namibia
    :cvar NCL: Trigraph for New Caledonia
    :cvar NER: Trigraph for Niger
    :cvar NFK: Trigraph for Norfolk Island
    :cvar NGA: Trigraph for Nigeria
    :cvar NIC: Trigraph for Nicaragua
    :cvar NIU: Trigraph for Niue
    :cvar NLD: Trigraph for Netherlands
    :cvar NOR: Trigraph for Norway
    :cvar NPL: Trigraph for Nepal
    :cvar NRU: Trigraph for Nauru
    :cvar NZL: Trigraph for New Zealand
    :cvar OMN: Trigraph for Oman
    :cvar PAK: Trigraph for Pakistan
    :cvar PAN: Trigraph for Panama
    :cvar PCN: Trigraph for Pitcairn
    :cvar PER: Trigraph for Peru
    :cvar PHL: Trigraph for Philippines
    :cvar PLW: Trigraph for Palau
    :cvar PNG: Trigraph for Papua New Guinea
    :cvar POL: Trigraph for Poland
    :cvar PRI: Trigraph for Puerto Rico
    :cvar PRK: Trigraph for Korea, Democratic People's Republic of
    :cvar PRT: Trigraph for Portugal
    :cvar PRY: Trigraph for Paraguay
    :cvar PSE: Trigraph for Palestinian Territory, Occupied
    :cvar PYF: Trigraph for French Polynesia
    :cvar QAT: Trigraph for Qatar
    :cvar REU: Trigraph for Réunion
    :cvar ROU: Trigraph for Romania
    :cvar RUS: Trigraph for Russian Federation
    :cvar RWA: Trigraph for Rwanda
    :cvar SAU: Trigraph for Saudi Arabia
    :cvar SDN: Trigraph for Sudan
    :cvar SEN: Trigraph for Senegal
    :cvar SGP: Trigraph for Singapore
    :cvar SGS: Trigraph for South Georgia and the South Sandwich Islands
    :cvar SHN: Trigraph for Saint Helena
    :cvar SJM: Trigraph for Svalbard and Jan Mayen
    :cvar SLB: Trigraph for Solomon Islands
    :cvar SLE: Trigraph for Sierra Leone
    :cvar SLV: Trigraph for El Salvador
    :cvar SMR: Trigraph for San Marino
    :cvar SOM: Trigraph for Somalia
    :cvar SPM: Trigraph for Saint Pierre and Miquelon
    :cvar SRB: Trigraph for Serbia
    :cvar STP: Trigraph for Sao Tome and Principe
    :cvar SUR: Trigraph for Suriname
    :cvar SVK: Trigraph for Slovakia
    :cvar SVN: Trigraph for Slovenia
    :cvar SWE: Trigraph for Sweden
    :cvar SWZ: Trigraph for Swaziland
    :cvar SYC: Trigraph for Seychelles
    :cvar SYR: Trigraph for Syrian Arab Republic
    :cvar TCA: Trigraph for Turks and Caicos Islands
    :cvar TCD: Trigraph for Chad
    :cvar TGO: Trigraph for Togo
    :cvar THA: Trigraph for Thailand
    :cvar TJK: Trigraph for Tajikistan
    :cvar TKL: Trigraph for Tokelau
    :cvar TKM: Trigraph for Turkmenistan
    :cvar TLS: Trigraph for Timor-Leste
    :cvar TON: Trigraph for Tonga
    :cvar TTO: Trigraph for Trinidad and Tobago
    :cvar TUN: Trigraph for Tunisia
    :cvar TUR: Trigraph for Turkey
    :cvar TUV: Trigraph for Tuvalu
    :cvar TWN: Trigraph for Taiwan, Province of China
    :cvar TZA: Trigraph for Tanzania, United Republic of
    :cvar UGA: Trigraph for Uganda
    :cvar UKR: Trigraph for Ukraine
    :cvar UMI: Trigraph for United States Minor Outlying Islands
    :cvar URY: Trigraph for Uruguay
    :cvar UZB: Trigraph for Uzbekistan
    :cvar VAT: Trigraph for Holy See (Vatican City State)
    :cvar VCT: Trigraph for Saint Vincent and the Grenadines
    :cvar VEN: Trigraph for Venezuela
    :cvar VGB: Trigraph for Virgin Islands, British
    :cvar VIR: Trigraph for Virgin Islands, U.S.
    :cvar VNM: Trigraph for Viet Nam
    :cvar VUT: Trigraph for Vanuatu
    :cvar WLF: Trigraph for Wallis and Futuna
    :cvar WSM: Trigraph for Samoa
    :cvar YEM: Trigraph for Yemen
    :cvar ZAF: Trigraph for South Africa
    :cvar ZMB: Trigraph for Zambia
    :cvar ZWE: Trigraph for Zimbabwe
    :cvar ACGU: Tetragraph for FOUR EYES
    :cvar APFS: Suppressed
    :cvar BWCS: Tetragraph for Biological Weapons Convention States
    :cvar CFCK: Tetragraph for ROK/US Combined Forces Command, Korea
    :cvar CMFC: Tetragraph for Combined Maritime Forces
    :cvar CMFP: Tetragraph for Cooperative Maritime Forces Pacific
    :cvar CPMT: Tetragraph for Civilian Protection Monitoring Team for Sudan
    :cvar CWCS: Tetragraph for Chemical Weapons Convention States
    :cvar EFOR: Tetragraph for European Union Stabilization Forces in Bosnia
    :cvar EUDA: Tetragraph for European Union DARFUR
    :cvar FVEY: Tetragraph for FIVE EYES
    :cvar GCTF: Tetragraph for Global Counter-Terrorism Forces
    :cvar GMIF: Tetragraph for Global Maritime Interception Forces
    :cvar IESC: Tetragraph for International Events Security Coalition
    :cvar ISAF: Tetragraph for International Security Assistance Force for Afghanistan
    :cvar KFOR: Tetragraph for Stabilization Forces in Kosovo
    :cvar MCFI: Tetragraph for Multinational Coalition Forces - Iraq
    :cvar MIFH: Tetragraph for Multinational Interim Force Haiti
    :cvar MLEC: Tetragraph for Multi-Lateral Enduring Contingency
    :cvar NACT: Tetragraph for North African Counter-Terrorism Forces
    :cvar NATO: Tetragraph for North Atlantic Treaty Organization
    :cvar SPAA: Suppressed
    :cvar TEYE: Tetragraph for THREE EYES
    :cvar UNCK: Tetragraph for United Nations Command, Korea
    """

    UNKNOWN = "UNKNOWN"
    ABW = "ABW"
    AFG = "AFG"
    AGO = "AGO"
    AIA = "AIA"
    ALA = "ALA"
    ALB = "ALB"
    AND = "AND"
    ANT = "ANT"
    ARE = "ARE"
    ARG = "ARG"
    ARM = "ARM"
    ASM = "ASM"
    ATA = "ATA"
    ATF = "ATF"
    ATG = "ATG"
    AUS = "AUS"
    AUT = "AUT"
    AZE = "AZE"
    BDI = "BDI"
    BEL = "BEL"
    BEN = "BEN"
    BFA = "BFA"
    BGD = "BGD"
    BGR = "BGR"
    BHR = "BHR"
    BHS = "BHS"
    BIH = "BIH"
    BLM = "BLM"
    BLR = "BLR"
    BLZ = "BLZ"
    BMU = "BMU"
    BOL = "BOL"
    BRA = "BRA"
    BRB = "BRB"
    BRN = "BRN"
    BTN = "BTN"
    BVT = "BVT"
    BWA = "BWA"
    CAF = "CAF"
    CAN = "CAN"
    CCK = "CCK"
    CHE = "CHE"
    CHL = "CHL"
    CHN = "CHN"
    CIV = "CIV"
    CMR = "CMR"
    COD = "COD"
    COG = "COG"
    COK = "COK"
    COL = "COL"
    COM = "COM"
    CPV = "CPV"
    CRI = "CRI"
    CUB = "CUB"
    CXR = "CXR"
    CYM = "CYM"
    CYP = "CYP"
    CZE = "CZE"
    DEU = "DEU"
    DJI = "DJI"
    DMA = "DMA"
    DNK = "DNK"
    DOM = "DOM"
    DZA = "DZA"
    ECU = "ECU"
    EGY = "EGY"
    ERI = "ERI"
    ESH = "ESH"
    ESP = "ESP"
    EST = "EST"
    ETH = "ETH"
    FIN = "FIN"
    FJI = "FJI"
    FLK = "FLK"
    FRA = "FRA"
    FRO = "FRO"
    FSM = "FSM"
    GAB = "GAB"
    GBR = "GBR"
    GEO = "GEO"
    GGY = "GGY"
    GHA = "GHA"
    GIB = "GIB"
    GIN = "GIN"
    GLP = "GLP"
    GMB = "GMB"
    GNB = "GNB"
    GNQ = "GNQ"
    GRC = "GRC"
    GRD = "GRD"
    GRL = "GRL"
    GTM = "GTM"
    GUF = "GUF"
    GUM = "GUM"
    GUY = "GUY"
    HKG = "HKG"
    HMD = "HMD"
    HND = "HND"
    HRV = "HRV"
    HTI = "HTI"
    HUN = "HUN"
    IDN = "IDN"
    IMN = "IMN"
    IND = "IND"
    IOT = "IOT"
    IRL = "IRL"
    IRN = "IRN"
    IRQ = "IRQ"
    ISL = "ISL"
    ISR = "ISR"
    ITA = "ITA"
    JAM = "JAM"
    JEY = "JEY"
    JOR = "JOR"
    JPN = "JPN"
    KAZ = "KAZ"
    KEN = "KEN"
    KGZ = "KGZ"
    KHM = "KHM"
    KIR = "KIR"
    KNA = "KNA"
    KOR = "KOR"
    KWT = "KWT"
    LAO = "LAO"
    LBN = "LBN"
    LBR = "LBR"
    LBY = "LBY"
    LCA = "LCA"
    LIE = "LIE"
    LKA = "LKA"
    LSO = "LSO"
    LTU = "LTU"
    LUX = "LUX"
    LVA = "LVA"
    MAC = "MAC"
    MAF = "MAF"
    MAR = "MAR"
    MCO = "MCO"
    MDA = "MDA"
    MDG = "MDG"
    MDV = "MDV"
    MEX = "MEX"
    MHL = "MHL"
    MKD = "MKD"
    MLI = "MLI"
    MLT = "MLT"
    MMR = "MMR"
    MNE = "MNE"
    MNG = "MNG"
    MNP = "MNP"
    MOZ = "MOZ"
    MRT = "MRT"
    MSR = "MSR"
    MTQ = "MTQ"
    MUS = "MUS"
    MWI = "MWI"
    MYS = "MYS"
    MYT = "MYT"
    NAM = "NAM"
    NCL = "NCL"
    NER = "NER"
    NFK = "NFK"
    NGA = "NGA"
    NIC = "NIC"
    NIU = "NIU"
    NLD = "NLD"
    NOR = "NOR"
    NPL = "NPL"
    NRU = "NRU"
    NZL = "NZL"
    OMN = "OMN"
    PAK = "PAK"
    PAN = "PAN"
    PCN = "PCN"
    PER = "PER"
    PHL = "PHL"
    PLW = "PLW"
    PNG = "PNG"
    POL = "POL"
    PRI = "PRI"
    PRK = "PRK"
    PRT = "PRT"
    PRY = "PRY"
    PSE = "PSE"
    PYF = "PYF"
    QAT = "QAT"
    REU = "REU"
    ROU = "ROU"
    RUS = "RUS"
    RWA = "RWA"
    SAU = "SAU"
    SDN = "SDN"
    SEN = "SEN"
    SGP = "SGP"
    SGS = "SGS"
    SHN = "SHN"
    SJM = "SJM"
    SLB = "SLB"
    SLE = "SLE"
    SLV = "SLV"
    SMR = "SMR"
    SOM = "SOM"
    SPM = "SPM"
    SRB = "SRB"
    STP = "STP"
    SUR = "SUR"
    SVK = "SVK"
    SVN = "SVN"
    SWE = "SWE"
    SWZ = "SWZ"
    SYC = "SYC"
    SYR = "SYR"
    TCA = "TCA"
    TCD = "TCD"
    TGO = "TGO"
    THA = "THA"
    TJK = "TJK"
    TKL = "TKL"
    TKM = "TKM"
    TLS = "TLS"
    TON = "TON"
    TTO = "TTO"
    TUN = "TUN"
    TUR = "TUR"
    TUV = "TUV"
    TWN = "TWN"
    TZA = "TZA"
    UGA = "UGA"
    UKR = "UKR"
    UMI = "UMI"
    URY = "URY"
    UZB = "UZB"
    VAT = "VAT"
    VCT = "VCT"
    VEN = "VEN"
    VGB = "VGB"
    VIR = "VIR"
    VNM = "VNM"
    VUT = "VUT"
    WLF = "WLF"
    WSM = "WSM"
    YEM = "YEM"
    ZAF = "ZAF"
    ZMB = "ZMB"
    ZWE = "ZWE"
    ACGU = "ACGU"
    APFS = "APFS"
    BWCS = "BWCS"
    CFCK = "CFCK"
    CMFC = "CMFC"
    CMFP = "CMFP"
    CPMT = "CPMT"
    CWCS = "CWCS"
    EFOR = "EFOR"
    EUDA = "EUDA"
    FVEY = "FVEY"
    GCTF = "GCTF"
    GMIF = "GMIF"
    IESC = "IESC"
    ISAF = "ISAF"
    KFOR = "KFOR"
    MCFI = "MCFI"
    MIFH = "MIFH"
    MLEC = "MLEC"
    NACT = "NACT"
    NATO = "NATO"
    SPAA = "SPAA"
    TEYE = "TEYE"
    UNCK = "UNCK"
