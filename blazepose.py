import sys
from os import path
import os
import subprocess

import sys
import shutil

INPUTVIDEOPATHS = [
    "/mnt/c/Users/u-ma1/Videos/WLASL/00335.mp4",
    "/mnt/c/Users/u-ma1/Videos/WLASL/00378.mp4",
    "/mnt/c/Users/u-ma1/Videos/WLASL/00430.mp4",
    "/mnt/c/Users/u-ma1/Videos/WLASL/00626.mp4",
]

def doBlazePose(inputVideoPath):
    dirname = path.dirname(inputVideoPath)
    basename = path.basename(inputVideoPath)
    outputDir = dirname + '/BlazePose'

    if os.path.exists(f"{outputDir}/{basename}"):
        os.remove(f"{outputDir}/{basename}")

    if not path.exists(outputDir):
        os.makedirs(outputDir)

    if os.path.exists(f"./result/BlazePose/{basename}"):
        shutil.rmtree(f"./result/BlazePose/{basename}")

    command = " ".join([
        f'GLOG_logtostderr=1',
        f'./bazel-bin/mediapipe/examples/desktop/upper_body_pose_tracking/upper_body_pose_tracking_cpu',
        f'--calculator_graph_config_file="./mediapipe/graphs/pose_tracking/upper_body_pose_tracking_cpu.pbtxt"',
        f'--input_video_path="{inputVideoPath}"',
        f'--output_video_path="{outputDir}/{basename}"',
    ])
    res = subprocess.run(command, stderr=subprocess.STDOUT, shell=True)

    command2 = " ".join([
        f'python3 convertPoseProtoToJson.py ./result/blazepose/{basename}',
    ])
    res = subprocess.run(command2, stderr=subprocess.STDOUT, shell=True)

if __name__=="__main__":
    if len(sys.argv) > 1:
        inputVideoPaths = [sys.argv[1]]
    else:
        inputVideoPaths = INPUTVIDEOPATHS

    # command = " ".join([
    #     f'bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/upper_body_pose_tracking:upper_body_pose_tracking_gpu',
    # ])
    # res = subprocess.run(command, stderr=subprocess.STDOUT, shell=True)

    for inputVideoPath in inputVideoPaths:
        print(inputVideoPath)
        doBlazePose(inputVideoPath)