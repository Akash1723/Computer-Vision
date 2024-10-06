import cv2
import numpy as np
import time
import autopy
import mediapipe as mp
import math
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/vmouse")
def vmouse():
    ##########################
    wCam, hCam = 640, 480
    wScr,hScr=autopy.screen.size()
    frameR = 100 # Frame Reduction
    smoothening = 10
    frameR=100
    smoothening=5
    #########################

    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    wScr, hScr = autopy.screen.size()
    # print(wScr, hScr)

    mpDraw=mp.solutions.drawing_utils
    mphands=mp.solutions.hands
    hands=mphands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.7,min_tracking_confidence=0.7)
    tipIds = [4, 8, 12, 16, 20]

    #########################################
    def findDistance(p1, p2, img, lmList,draw=True,r=15, t=3):
        x1, y1 = lmList[p1][1:]
        x2, y2 = lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    ########################################

    while True:
        # 1. Find hand Landmarks
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image.")
            break
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results=hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if True:
                    mpDraw.draw_landmarks(img, handLms,mphands.HAND_CONNECTIONS)
        
        xList = []
        yList = []
        bbox = []
        lmList = []
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if True:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if xList and yList:
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),(0, 255, 0), 2)
        
        fingers = [0,0,0,0]


        if len(lmList)!=0:
            x1,y1=lmList[8][1:]
            x2,y2=lmList[12][1:]

            #print(x1,y1,x2,y2)
            finger=[]
            
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                finger.append(1)
            else:
                finger.append(0)
            
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    finger.append(1)
                else:
                    finger.append(0)
            fingers=finger
        #print(fingers)
        
        cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)
        if fingers[1]==1 and fingers[2]==0:
            x3=np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3=np.interp(y1,(frameR,hCam-frameR),(0,hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            autopy.mouse.move(wScr-clocX,clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocX, plocY = clocX, clocY
        
        if fingers[1]==1 and fingers[2]==1:
            length,img,lineInfo=findDistance(8,12,img,lmList)
            print(length)
            if length<40:
                cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()

        
        # 11. Frame Rate
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
        (255, 0, 0), 3)
        # 12. Display
        cv2.imshow("img", img)
        if cv2.waitKey(1)==ord('x'):
            break

    cap.release()