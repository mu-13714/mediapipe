import sys
from os import path
import os
import json
import math
import matplotlib.pyplot as plt
import glob
import re

INPUTJSONPATH = [
    "./result/MHT/00626.mp4"
]

def convertJson(target_dir):
    output_files = glob.glob(target_dir + "/" + "*.json")

    print('get json file')
    target_files = [(re.findall(
        f"iLoop=(\\d+)_landmarkRaw_j=0.json", output_file), output_file) for output_file in output_files]
    target_files_filetered = [
        (target_file[0], target_file[1].replace("\\", "/")) for target_file in target_files if target_file[0]]
    
    print('make graph')
    fig = plt.figure()
    DIPax = fig.add_subplot(221)
    PIPax = fig.add_subplot(222)
    MCPax = fig.add_subplot(223)

    print('make yvals list')
    # 人差し指, 中指, 薬指, 小指
    DIPyvals = [[], [], [], []]
    PIPyvals = [[], [], [], []]
    MCPyvals = [[], [], [], []]
    frame_number = 0
    print('convert start')
    for target_file in target_files_filetered:
        with open(target_file[1], 'rb') as f:
            json_load = json.load(f)
        
        while f"{frame_number}" not in target_file[0]:
            for i in range(0, 4):
                DIPyvals[i].append(0)
                PIPyvals[i].append(0)
                MCPyvals[i].append(0)
            frame_number += 1
            
        Landmarks = getLandmarkCoord(json_load)
        calcDIPAngle(Landmarks, DIPyvals)
        calcPIPAngle(Landmarks, PIPyvals)
        calcMCPAngle(Landmarks, MCPyvals)
        frame_number += 1

    xvals = list(range(0, frame_number))
    # DIP
    DIPax.plot(xvals, DIPyvals[0], label='index')
    DIPax.plot(xvals, DIPyvals[1], label='middle')
    DIPax.plot(xvals, DIPyvals[2], label='ring')
    DIPax.plot(xvals, DIPyvals[3], label='little')
    # PIP
    PIPax.plot(xvals, PIPyvals[0], label='index')
    PIPax.plot(xvals, PIPyvals[1], label='middle')
    PIPax.plot(xvals, PIPyvals[2], label='ring')
    PIPax.plot(xvals, PIPyvals[3], label='little')
    # MCP
    MCPax.plot(xvals, MCPyvals[0], label='index')
    MCPax.plot(xvals, MCPyvals[0], label='middle')
    MCPax.plot(xvals, MCPyvals[0], label='ring')
    MCPax.plot(xvals, MCPyvals[0], label='little')

    plt.show()
            

# jsonからlandmarkの座標をリストとして取得
def getLandmarkCoord(json_load):
    Landmarks = []
    for i in range(21):
        x = json_load["landmark"][i]["x"]
        y = json_load["landmark"][i]["y"]
        z = json_load["landmark"][i]["z"]
        Landmarks.append((x, y, z))
    
    return Landmarks
        
# ラジアン
def calcAngle(p0, p1, p2):
    dot = ((p0[0] - p1[0]) * (p2[0] - p1[0]) + (p0[1] - p1[1]) * (p2[1] - p1[1]) + (p0[2] - p1[2]) * (p2[2] - p1[2]))
    norm01 = math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2] - p1[2])**2)
    norm21 = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
    cos = dot / (norm01 * norm21)

    return math.acos(cos)

def calcDIPAngle(landmarks, yvals):
    j = 0
    for i in range(6, 19, 4):
        rad = calcAngle(landmarks[i], landmarks[i+1], landmarks[i+2])
        yvals[j].append(rad)
        j += 1

def calcPIPAngle(landmarks, yvals):
    j = 0
    for i in range(5, 18, 4):
        rad = calcAngle(landmarks[i], landmarks[i+1], landmarks[i+2])
        yvals[j].append(rad)
        j += 1

def calcMCPAngle(landmarks, yvals):
    j = 0
    for i in range(5, 18, 4):
        rad = calcAngle(landmarks[0], landmarks[i], landmarks[i+1])
        yvals[j].append(rad)
        j += 1

if __name__ == "__main__":
    for path in INPUTJSONPATH:
        convertJson(path)