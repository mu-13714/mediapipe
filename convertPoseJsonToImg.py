import json
import cv2
import numpy as np

# json
json_path = './result/blazepose/00691.mp4/iLoop=24_landmark.json'
json_open = open(json_path, 'r')
json_load = json.load(json_open)

# image
width = 640
height = 360
img = np.zeros((height, width, 3))

for i in range(31):
    x = json_load['landmark'][i]['x'] * 640
    y = json_load['landmark'][i]['y'] * 360

    #cv2.circle(img, (int(x), int(y)), 3, (255, 0, 0), thickness=-1)
    if i >= 25:
        cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), lineType=cv2.LINE_AA)
    else:
        cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), lineType=cv2.LINE_AA)

cv2.imwrite('./result/blazepose/00691.mp4/iLoop=24_landmark2.png', img)
