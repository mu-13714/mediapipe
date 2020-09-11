from mediapipe.framework.formats.landmark_pb2 import LandmarkList
from google.protobuf.json_format import MessageToJson

import glob
import re
from pprint import pprint
import json
import sys

if len(sys.argv) > 1:
    args = sys.argv
    targetDir = args[1]
else:
    targetDir = "./result/blazepose/00691.mp4/"

outputFiles = glob.glob(targetDir + "/" + "*.txt")

landmarkFiles = [(re.findall(
    f"iLoop=(\d+)_landmark.txt", outputFile), outputFile) for outputFile in outputFiles]
landmarkFilesFiltered = [
        (landmarkFile[0], landmarkFile[1].replace("\\", "/")) for landmarkFile in landmarkFiles if landmarkFile[0]]

for landmarkFile in landmarkFilesFiltered:
    with open(landmarkFile[1], "rb") as f:
        content = f.read()

    landmark = LandmarkList()
    landmark.ParseFromString(content)
    jsonObj = MessageToJson(landmark)

    landmarkFileOutput = landmarkFile[1].replace("txt", "json")
    with open(landmarkFileOutput, "w") as f:
        f.write(jsonObj)
