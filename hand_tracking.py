import cv2
import mediapipe as mp
import time
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/handtrack")
def Handtrack():

    cap=cv2.VideoCapture(0)

    mphands=mp.solutions.hands
    hands=mphands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)#these are default that are already defines so no need to define it like this unless we want change
    mpDraw=mp.solutions.drawing_utils
    ptime=0
    ctime=0
    cap.set(3,1080)
    cap.set(4,720)

    while True:
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

       ctime=time.time()
       fps=1/(ctime-ptime)
       ptime=ctime

       cv2.putText(img,str(int(fps)),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
       cv2.imshow("frame",img)
       if cv2.waitKey(1)==ord("x"):
        break
    cv2.destroyAllWindows()