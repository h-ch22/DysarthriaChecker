import glob
import json
import os

from models.PatientInfo import PatientInfo
from models.GenderType import GenderType
from models.LanguageSubCatType import LanguageSubCatType


class IOHelper:
    def load_file(self, LABEL_PATH, SOURCE_PATH, CLASSES):
        global LABELED_FILES
        patients = []

        if os.path.exists(LABEL_PATH):
            for CLASS in CLASSES:
                print(CLASSES[CLASS])
                LABELED_FILES = glob.glob(LABEL_PATH + r'\%s' % CLASSES[CLASS] + r'\*.json')
                print(LABELED_FILES)

                for file in LABELED_FILES:
                    jsonFile = open(file, 'rt', encoding='UTF8')
                    data = json.load(jsonFile)

                    diseaseType = data['Disease_info']['Type']
                    samplingRate = data['Meta_info']['SamplingRate']
                    ID = data['File_id']
                    sex = data['Patient_info']['Sex']
                    age = data['Patient_info']['Age']
                    subCategory = data['Disease_info']['Subcategory' + str(int(diseaseType))]
                    audioFile = SOURCE_PATH + r'\%s' % CLASSES[int(subCategory)] + r'\%s' % ID
                    script = data['Transcript']
                    recordingEnvironment = data['Meta_info']['RecordingEnviron']
                    recordingDevice = data['Meta_info']['RecordingDevice']
                    noise = str(ID).split('-')[3]
                    havingNoise = True if noise == 'Y' else False
                    endPos = data['Meta_info']['EndPos']
                    startPos = data['Meta_info']['StartPos']
                    playTime = data['Meta_info']['PlayTime']

                    if os.path.exists(audioFile):
                        patients.append(
                            PatientInfo(
                                id=ID, samplingRate=int(samplingRate),
                                sex=GenderType.MALE if sex == 'M' else GenderType.FEMALE,
                                age=int(age), audioFileRoot=audioFile, transcript=script,
                                recordingEnvironment=recordingEnvironment, recordingDevice=recordingDevice,
                                havingNoise=havingNoise, startPos=float(startPos), endPos=float(endPos),
                                playTime=float(playTime),
                                subType=LanguageSubCatType(int(subCategory))
                            )
                        )

                    else:
                        raise Exception("FILE NOT FOUND EXCEPTION\nFILE : %s" % audioFile)

            return patients

        else:
            raise Exception("DIRECTORY NOT FOUND EXCEPTION\nDIR : %s" % LABEL_PATH)
