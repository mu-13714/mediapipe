import sys
from os import path
import os
import subprocess

import sys
import shutil

INPUTVIDEOPATHS = [
    # "/mnt/c/Users/u-ma1/Videos/WLASL/00335.mp4",
    #"/mnt/c/Users/u-ma1/Videos/WLASL/00378.mp4",
    # "/mnt/c/Users/u-ma1/Videos/WLASL/00430.mp4",
    #"/mnt/c/Users/u-ma1/Videos/WLASL/00626.mp4",
    #"/mnt/c/Users/u-ma1/Videos/WLASL/00665.mp4",
    "/mnt/c/Users/u-ma1/Videos/WLASL/00634.mp4"
]

def doMultiHandTracking(inputVideoPath):
    dirname = path.dirname(inputVideoPath)
    basename = path.basename(inputVideoPath)
    #outputDir = inputVideoPath.replace(".mp4", "-mp4")
    outputDir = dirname + '/MHT'

    if os.path.exists(f"{outputDir}/{basename}"):
        os.remove(f"{outputDir}/{basename}")

    if not path.exists(outputDir):
        os.makedirs(outputDir)

    if os.path.exists(f"./result/MHT/{basename}"):
        shutil.rmtree(f"./result/MHT/{basename}")

    command = " ".join([
        f'GLOG_logtostderr=1',
        f'./bazel-bin/mediapipe/examples/desktop/multi_hand_tracking/multi_hand_tracking_cpu',
        f'--calculator_graph_config_file="./mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live.pbtxt"',
        f'--input_video_path="{inputVideoPath}"',
        f'--output_video_path="{outputDir}/{basename}"',
    ])
    res = subprocess.run(command, stderr=subprocess.STDOUT, shell=True)

    command2 = " ".join([
        f'python3 convertHandProtoToJson.py ./result/MHT/{basename}',
    ])
    res = subprocess.run(command2, stderr=subprocess.STDOUT, shell=True)

if __name__=="__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            inputVideoPaths = [s.strip() for s in f.readlines()]
    else:
        inputVideoPaths = INPUTVIDEOPATHS

    # command = " ".join([
    #     f'bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_cpu',
    # ])
    # res = subprocess.run(command, stderr=subprocess.STDOUT, shell=True)

    for inputVideoPath in inputVideoPaths:
        print(inputVideoPath)
        doMultiHandTracking(inputVideoPath)