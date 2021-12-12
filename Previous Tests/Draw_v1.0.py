import cv2 as cv
import mediapipe as mp
import numpy as np
from sklearn.metrics import pairwise
import time

# Canvas
canvas = None
x1,y1 = 0,0

# Global variables
mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def circles(x1, y1, x2, y2, r1, r2):
    distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    radSumSq = (r1 + r2) * (r1 + r2)
    if (distSq == radSumSq):
        return 1 
    elif (distSq > radSumSq):
        return -1 
    else:
        return 0 

def drawHands(img):
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    return img

def getPoints(img, handNo = 0):
    landmkLst = []
    results = hands.process(img)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[handNo]
        height, width, _ = img.shape
        for index, lm in enumerate(hand.landmark):
            cx, cy = int(lm.x * width), int(lm.y * height)
            landmkLst.append([index, cx, cy])

    return landmkLst

def distance(img,cX, cY,left, right, top, bottom):
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
    max_distance = distance.max()
    radius = int(0.2 * max_distance)
    circumference = (2 * np.pi * radius)

    circular_roi = np.zeros(img.shape[:2], dtype="uint8")

    cir = cv.circle(circular_roi, (cX, cY), radius, 255, 10)
    return radius, circular_roi

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
        lmLst = getPoints(img)

        if len(lmLst) != 0:
            # Index finger coordinates
            ix, iy = lmLst[8][1], lmLst[8][2]

            # Area of hand:
            # Middle finger (Top)
            mx, my = lmLst[12][1], lmLst[12][2]

            # Wrist (Bottom)
            wx, wy = lmLst[0][1] , lmLst[0][2]

            # Pinky finger (right)
            px, py = lmLst[20][1], lmLst[20][2]

            # Thumb (left) 
            tx , ty = lmLst[4][1], lmLst[4][2]

            top    = (mx,my)
            bottom = (wx,wy)
            right  = (px,py)
            left   = (tx,ty)

            # In theory, the center of the hand is half way between the top and bottom and halfway between left and right
            cX = (left[0] + right[0]) // 2
            cY = (top[1] + bottom[1]) // 2

            rad, circular_roi = distance(img,cX,cY,left,right,top,bottom)

            # Index finger circle
            iX = ix
            iY = iy

            cv.circle(circular_roi, (iX, iY), rad, 255, 10)
            # cv.imshow("Index finger circle", IndexCircle)

            # Thumb circle
            tX = tx
            tY = ty

            cv.circle(circular_roi, (tX, tY), rad, 255, 10)
            # cv.imshow("Thumb finger circle", ThumbCircle)
            
            t = circles(tX,tY,iX,iY,rad,rad)

            if (t < 0):
                pass

            else:
                avgX = (tX + iX ) // 2
                avgY = (tY + iY ) // 2

                if x1 == 0 and y1 == 0:
                    x1,y1 = avgX,avgY
            
                else:
                    canvas = cv.line(canvas, (x1,y1),(avgX,avgY), [72, 139, 247], 3)
        
                x1,y1 = avgX,avgY
        

        curTime = time.time()
        FPS = 1 / (curTime - prevTime)
        prevTime = curTime

        cv.putText(img, f'FPS: {int(FPS)}', (5, 40),
                   cv.FONT_HERSHEY_PLAIN, 2, (66, 96, 245), 4)


        img = cv.add(img,canvas) #to see the result in image too 
        stacked = np.hstack((img,canvas))
        cv.imshow('Beautiful hands', cv.resize(stacked,None,fx=1.1,fy=1.1))

        key = cv.waitKey(1) & 0xFF
        if key == ord('d'): #exit
            break
        if key == ord('c'): #clear canvas
            canvas = None

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
