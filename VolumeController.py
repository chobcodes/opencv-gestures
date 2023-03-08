import cv2
import numpy as np
import time
import HandTrackingBase as htb
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)

detector = htb.handDetector(detection_confidence=0.8)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

vol = 0
volBar = 400

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPos(img, draw=False)

    if len(lmList):
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        length = math.hypot(x2-x1,y2-y1)
        
        vol = np.interp(length, [50,300], [minVol, maxVol])
        volBar = np.interp(length, [50,300], [400, 150])
        volume.SetMasterVolumeLevel(vol, None)

    cv2.rectangle(img, (50,150), (85,400), (255,0,0), 2)
    cv2.rectangle(img, (50,int(volBar)), (85,400), (255,0,0), cv2.FILLED)

    cv2.imshow("Img", img)
    cv2.waitKey(1)