import cv2
import mediapipe as mp
import time
import numpy as np
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/Volume")

def Vol():
    cap=cv2.VideoCapture(0)

    mphands=mp.solutions.hands
    hands=mphands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.7,min_tracking_confidence=0.5)#these are default that are already defines so no need to define it like this unless we want change
    mpDraw=mp.solutions.drawing_utils

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
    volumerange=volume.GetVolumeRange()
    minvol=volumerange[0]
    maxvol=volumerange[1]

    def calculate_distance(point1, point2):
        return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

    ptime=0
    ctime=0
    cap.set(3,1080)
    cap.set(4,720)
    flag=0
    term_flag=0
    while True:
        rel,frame=cap.read()

        framergb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=hands.process(framergb)
        
        #hand detection code:
        if results.multi_hand_landmarks and flag==0:
            for handlandmark in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame,handlandmark,mphands.HAND_CONNECTIONS)
                for id,lm in enumerate(handlandmark.landmark):
                    h,w,c=frame.shape
                    cx,cy=int(lm.x*w),int(lm.y*h)
                    lmlist=[id,cx,cy]
                    if(id==4):
                        x1=cx
                        y1=cy
                    if(id==8):
                        x2=cx
                        y2=cy
                    if(id==4 or id==8):
                        cv2.circle(frame,(cx,cy),10,(255,0,255),cv2.FILLED)
                        if(id==8):
                            cv2.line(frame,(x1,y1),(x2,y2),(255,0,255),3)
                            center1,center2=(x1+x2)//2,(y1+y2)//2
                            cv2.circle(frame,(center1,center2),7,(255,0,255),cv2.FILLED)

                            length=math.hypot(x2-x1,y2-y1)
                            #print(length)

                            if(length<70):
                                cv2.circle(frame,(center1,center2),7,(0,255,0),cv2.FILLED)
                            if(length>130):
                                cv2.circle(frame,(center1,center2),7,(0,0,255),cv2.FILLED)
                            
                            vol=np.interp(length,[20,150],[minvol,maxvol])
                            #print(vol)
                            volume.SetMasterVolumeLevel(vol, None)
                    #print(lmlist)
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        cv2.putText(frame,str(int(fps)),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        cv2.imshow("frame",frame)

        #termination code:
        if results.multi_hand_landmarks:
            for handlandmark in results.multi_hand_landmarks:
                index_finger = handlandmark.landmark[8]
                middle_finger_tip = handlandmark.landmark[12]
                thumb=handlandmark.landmark[4]
                pause=handlandmark.landmark[0]

                pause_dist = calculate_distance((pause.x * w, pause.y * h), (middle_finger_tip.x * w, middle_finger_tip.y * h))
                termination = calculate_distance((pause.x * w, pause.y * h), (index_finger.x * w, index_finger.y * h))
                #print(termination)
                if(pause_dist<150):
                    flag=0
                else:
                    flag=1
                if(termination<100):
                    term_flag=1

        if cv2.waitKey(1)==ord("x") or (term_flag==1 and flag==1):
            break
    cv2.destroyAllWindows()