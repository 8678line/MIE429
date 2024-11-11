# Video Analysis

import cv2
import easyocr
import numpy as np
from time import perf_counter
import os
import matplotlib.pyplot as plt

reader = easyocr.Reader(['en']) # specify the language 

# cam = cv2.VideoCapture("SANMH13-SANMH14_Street G_221025_080504.mp4") 
#cam = cv2.VideoCapture("SAN-6_SAN-7_U_111821.mpg") 
cam = cv2.VideoCapture("SAP 34514 SAP 34513_SAP 34513_202310311440.mp4")

amount_of_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cam.get(cv2.CAP_PROP_FPS)


print(amount_of_frames)
for i in range(amount_of_frames):


    ret, frame = cam.read()

    frame = frame[int(len(frame)*3/4):]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (len(frame[0])*4, len(frame)*4))


    result = reader.readtext(frame, allowlist='0123456789.:m')

    j = 0
    for (bbox, text, prob) in result:
        if(text == 'm' and j != 0):
            result[j-1] = (result[j-1][0], result[j-1][1] + "m", result[j-1][2])
        j += 1
    x_len = 0
    for (bbox, text, prob) in result:
        if(text[-1] == 'm' and any(ch in text for ch in ["00", ".0", "0."])):
            x_len = bbox[1][0] - bbox[0][0]
            bounds = [bbox[0][0] - x_len, bbox[1][0], bbox[0][1], bbox[2][1]]
            num = i
            break
    if(x_len != 0):
        break

arr = []
dot_loc = [0,0,0,0,0,0,0]
while True:
    num += 1
    ret, frame = cam.read()

    if(not ret or i == 15000):
        break
    if(num % 10 != 0):
        continue
    # if(num % 1000 == 0):
    #     break

    frame = frame[int(len(frame)*3/4):]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (len(frame[0])*4, len(frame)*4))
    frame = frame[bounds[2]:bounds[3], bounds[0]:bounds[1]]
    result = reader.readtext(frame, allowlist='0123456789.m')
    if(len(result) == 0):
        continue
    arr.append([result[0][1], num])
    if("." in result[0][1]):
        dot_loc[result[0][1].index(".")] += 1

    if(num % 100 == 0):
        print(num)

best_dot_pos = dot_loc.index(max(dot_loc))
numeric_readings = []
frame_nums = []
for i in range(len(arr)):
    if('m' in arr[i][0][:-1]):
        continue
    elif('m' in arr[i][0]):
        arr[i][0] = arr[i][0][:-1]
    
    string = arr[i][0].replace(".", "")
    if(len(string) > best_dot_pos):
        string = string[:best_dot_pos] + "." + string[best_dot_pos:]
    
    if(string == ""):
        continue

    numeric_readings.append(float(string))
    frame_nums.append(arr[i][1])

plt.plot(frame_nums, numeric_readings)
plt.xlabel("Frame Number")
plt.ylabel("Distance Reading (m)")
plt.title("Colgan SAN-13 - SAN-14 OCR Readings")
plt.show()
check = 0
