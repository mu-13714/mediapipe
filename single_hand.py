import sys
from os import path
import os
import subprocess

import sys
import shutil

INPUTVIDEOPATHS = [
    "/mnt/c/Users/u-ma1/Videos/WLASL/00335.mp4",
    "/mnt/c/Users/u-ma1/Videos/WLASL/00378.mp4",
]

def doMultiHandTracking(inputVideoPath):
    dirname = path.dirname(inputVideoPath)
    basename = path.basename(inputVideoPath)
    outputDir = dirname + '/SHT'

    if os.path.exists(f"{outputDir}/{basename}"):
        os.remove(f"{outputDir}/{basename}")

    if not path.exists(outputDir):
        os.makedirs(outputDir)

    if os.path.exists(f"./result/MHT/{basename}"):
        shutil.rmtree(f"./result/MHT/{basename}")

    command = " ".join([
        f'GLOG_logtostderr=1',
        f'./bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu',
        f'--calculator_graph_config_file="./mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt"',
        f'--input_video_path="{inputVideoPath}"',
        f'--output_video_path="{outputDir}/{basename}"',
    ])
    res = subprocess.run(command, stderr=subprocess.STDOUT, shell=True)

if __name__=="__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            inputVideoPaths = [s.strip() for s in f.readlines()]
    else:
        inputVideoPaths = INPUTVIDEOPATHS

    for inputVideoPath in inputVideoPaths:
        print(inputVideoPath)
        doMultiHandTracking(inputVideoPath)