from dataclasses import dataclass
from models import GenderType
from models import LanguageSubCatType


@dataclass
class PatientInfo:
    id: str
    samplingRate: int
    sex: GenderType
    age: int
    audioFileRoot: str
    transcript: str
    recordingEnvironment: str
    recordingDevice: str
    havingNoise: bool
    startPos: float
    endPos: float
    playTime: float
    subType: LanguageSubCatType
