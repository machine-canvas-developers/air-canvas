from typing import Counter
import cv2 as cv
import mediapipe as mp
from mediapipe.python.packet_creator import create_image_frame
import numpy as np
import math as m
from sklearn.metrics import pairwise
import os
# import matplotlib as plt

mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    os.makedirs("data/train/palm")
    os.makedirs("data/train/twoF") #two fingers (peace sign)
    # os.makedirs("data/train/fiveF")
    os.makedirs("data/train/broFist")
    # os.makedirs("data/train/Lshape")
    # os.makedirs("data/train/okay")

    os.makedirs("data/test/palm")
    os.makedirs("data/test/twoF") 
    # os.makedirs("data/test/fiveF")
    os.makedirs("data/test/broFist")
    # os.makedirs("data/test/Lshape")
    # os.makedirs("data/test/okay")

# mode = train or test
mode = 'train' # for testing set to test
directory = 'data/' + mode + '/'  


canvas = None
x1,y1 = 0,0

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv.VideoCapture(0)

# frame_count = 0

# prev_frame = img


def circles(x1, y1, x2, y2, r1, r2):
    distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2); 
    radSumSq = (r1 + r2) * (r1 + r2); 
    if (distSq == radSumSq):
        return 1 
    elif (distSq > radSumSq):
        return -1 
    else:
        return 0 

def lmList(img, handNo = 0):
    lmLst = []
    results = hands.process(img)

    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
            # print(id, lm)
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            # print(id, cx, cy)
            lmLst.append([id, cx, cy])

    return lmLst

def drawHands(img):
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    return img

def main():
    global canvas,x1,y1
    cap = cv.VideoCapture(0)

    # For FPS
    prevTime = 0
    curTime = 0

    while True:
        success, img = cap.read()

        if not success:
            print("Please check your webcam.")
            continue

        if canvas is None:
            canvas = np.zeros_like(img)

        img = cv.flip(img, 1)
        img = drawHands(img)

        lmLst = lmList(img)

        if len(lmLst) != 0:
                
            ix, iy = lmLst[8][1], lmLst[8][2]

            # Area of hand:
            # Middle Finger (Top)
            #mx, my = lmLst[8][1], lmLst[8][2]
            mx, my = ix, iy
            # Wrist (Bottom)
            wx, wy = lmLst[0][1] , lmLst[0][2]

            # Pinky finger (right)
            px, py = lmLst[20][1], lmLst[20][2]

            # Thumb (left) 
            tx , ty = lmLst[4][1], lmLst[4][2]

            fingers = []
            tipIds = [8, 12, 16, 20]

            # # Thumb
            # if lmLst[tipIds[0]][1] > lmLst[tipIds[0] - 1][1]:
            #     fingers.append(1)
            # else:
            #     fingers.append(0)

            # 4 Fingers
            for id in range(0, 4):
                if lmLst[tipIds[id]][2] < lmLst[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Fingers Opened

            # print(fingers)

            if fingers[1] == 1:   # Middle Finger
                mx, my = lmLst[12][1], lmLst[12][2]
            elif fingers[0] == 1: # Index Finger
                mx, my = lmLst[8][1], lmLst[8][2]
            elif fingers[2] == 1: # Ring Finger
                mx, my = lmLst[16][1], lmLst[16][2]
            elif fingers[3] == 1: # Pinky Finger
                mx, my = lmLst[20][1], lmLst[20][2]


            top    = (mx,my)
            bottom = (wx,wy)
            right  = (px,py)
            left   = (tx,ty)

            # In theory, the center of the hand is half way between the top and bottom and halfway between left and right
            cX = (left[0] + right[0]) // 2
            cY = (top[1] + bottom[1]) // 2
            
            distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
            
            # Grab the largest distance
            max_distance = distance.max()
            print(max_distance)
            
            # Create a circle with 1% radius of the max euclidean distance
            radius = int(max_distance)
            circumference = (2 * np.pi * radius)

            # Now grab an ROI of only that circle
            #circular_roi = np.zeros(img.shape, dtype="uint8")

            x1, y1 = (cX - radius), (cY + radius)
            x2, y2 = (cX + radius), (cY - radius) 
            
            roi = img[y2:y1,x1:x2]
            roi = cv.resize(roi, (128,128))

            print("shape ",roi.shape)

            cv.imshow("ROI", roi)

            count = {
                'palm' : len(os.listdir(directory +"/palm")),
                'twoF' : len(os.listdir(directory +"/twoF")),
                # 'fiveF' : len(os.listdir(directory +"/fiveF")),
                'broFist' : len(os.listdir(directory +"/broFist"))
                # 'Lshape' : len(os.listdir(directory +"/Lshape")),
                # 'okay' : len(os.listdir(directory +"/okay"))
            }

            # Prints that count on screen
            cv.putText(img, f"Mode: {mode}", (9,50),cv.FONT_HERSHEY_PLAIN,1,(26,2,181),2)
            cv.putText(img, f"img count ->", (9,100),cv.FONT_HERSHEY_PLAIN,1,(26,2,181),2)
            cv.putText(img, f"Palm: {str(count['palm'])}", (9,120),cv.FONT_HERSHEY_PLAIN,1,(26,2,181),2)
            cv.putText(img, f"Two Fingers: {str(count['twoF'])}", (9,140),cv.FONT_HERSHEY_PLAIN,1,(26,2,181),2)
            # cv.putText(img, f"Five Fingers: {str(count['fiveF'])}", (9,160),cv.FONT_HERSHEY_PLAIN,1,(26,2,181),2)
            cv.putText(img, f"Fist: {str(count['broFist'])}", (9,180),cv.FONT_HERSHEY_PLAIN,1,(26,2,181),2)
            # cv.putText(img, f"Lshape: {str(count['Lshape'])}", (9,200),cv.FONT_HERSHEY_PLAIN,1,(26,2,181),2)
            # cv.putText(img, f"Okay: {str(count['okay'])}", (9,220),cv.FONT_HERSHEY_PLAIN,1,(26,2,181),2)
            
            #Hand ROI
            # roi = cv.circle(circular_roi, (cX, cY), radius, 255, 10)
            # roi = cv.resize(roi, (128,128))d
            # cv.imshow("ROI1",roi)
            # print(roi.shape)
            # Converting the ROI into black-white img (Thresholding)
            gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)[1]
            pixels = cv.countNonZero(thresh)
            #_, roi = cv.threshold(roi, 120, 255, cv.THRESH_BINARY)
            cv.imshow("ROI2", thresh)

            cv.imshow("Pixels", pixels)


            cv.imshow("Frame",img)

            key = cv.waitKey(1) & 0xFF

            if key == ord('p'):
                cv.imwrite(directory + 'palm/' + str(count['palm']) + '.jpg', thresh)
            if key == ord('t'):
                cv.imwrite(directory + 'twoF/' + str(count['twoF']) + '.jpg', thresh)
            # if key == ord('f'):
            #     cv.imwrite(directory + 'fiveF/' + str(count['fiveF']) + '.jpg', thresh)
            if key == ord('b'):
                cv.imwrite(directory + 'broFist/' + str(count['broFist']) + '.jpg', thresh)
            # if key == ord('l'):
            #     cv.imwrite(directory + 'Lshape/' + str(count['Lshape']) + '.jpg', thresh)
            # if key == ord('o'):
            #     cv.imwrite(directory + 'okay/' + str(count['okay']) + '.jpg', thresh)
        
        # img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img = cv.add(img,canvas) #to see the result in img too 
        # stacked = np.hstack((img,canvas))
        # cv.imshow('Beautiful hands', cv.resize(stacked,None,fx=1.2,fy=1.2))

        # frame_count += 1
        # prev_frame = img.copy()
        # success, img = cap.read() # img = crurent frame

        key = cv.waitKey(1) & 0xFF
        if key == ord('d'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()