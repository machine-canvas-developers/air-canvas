import cv2 as cv
import numpy as np
import mediapipe as mp
from sklearn.metrics import pairwise
import os 
import time

# Canvas
canvas = None
xx1,yy1 = 0,0

current_file_path = os.path.dirname(os.path.realpath(__file__))
directory = current_file_path
os.chdir(directory)

# Global variables
mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 1, 0.5, 0.5)#min_detection_confidence=0.5, min_tracking_confidence=0.5)


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

def circles(x1, y1, x2, y2, r1, r2):
    distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    radSumSq = (r1 + r2) * (r1 + r2)
    if (distSq == radSumSq):
        return 1 
    elif (distSq > radSumSq):
        return -1 
    else:
        return 0 

def main():
    color = (0,0,255)
    global canvas,xx1,yy1

    brushList = os.listdir("brushes")
    brLists = list()
    for brush in brushList:
        image = cv.imread(f"brushes/{brush}")
        brLists.append(image)

    header = brLists[0]

    cap = cv.VideoCapture(0)
    cap.set(3, cv.CAP_PROP_FRAME_WIDTH)
    cap.set(4, cv.CAP_PROP_FRAME_HEIGHT)

    while True:
        success, img = cap.read()
        img = cv.flip(img,1)
        img = cv.resize(img,(1280,720))

        if not success:
            print("Please check your webcam.")
            continue

        if canvas is None:
            canvas = np.zeros_like(img)


        img = drawHands(img)
        lmLst = getPoints(img)

        if len(lmLst) != 0:
            x1,y1 = lmLst[8][1:] #index

            #Selecting brushes
            # cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), color, cv.FILLED)
            if y1 < 125:
                if 216 < x1 < 308:
                    header = brLists[0]
                    color = (0,0,255)  #red
                elif 364 < x1 < 457:
                    header = brLists[1]
                    color = (0,127,255) # orange
                elif 507 < x1 < 599:
                    header = brLists[2]
                    color = (0,255,127)  # greeen
                elif 659 < x1 < 752:
                    header = brLists[3]
                    color = (255,182,56)  # blue
                elif 806 < x1 < 899:
                    header = brLists[4]
                    color = (0,255,255)     # yellow
                elif 946 < x1 < 1039:
                    header = brLists[5]
                    color = (84,84,84) # black
                elif 1097 < x1 < 1218:
                    header = brLists[6]
                    color = (0,0,0)
            else:
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
                    xx1,yy1 = 0,0 # whenever hand is detected make it org
                    pass

                else:
                    avgX = (tX + iX ) // 2
                    avgY = (tY + iY ) // 2

                    if xx1 == 0 and yy1 == 0:
                        xx1,yy1 = avgX,avgY
                    if color == (0,0,0):
                        cv.line(img, (xx1,yy1),(x1,y1), color, 50)
                        cv.line(canvas, (xx1,yy1),(x1,y1), color, 50)

                    cv.line(img, (xx1,yy1),(x1,y1), color, 15)
                    cv.line(canvas, (xx1,yy1),(x1,y1), color, 15)
            
                    xx1,yy1 = avgX,avgY


        imgGray = cv.cvtColor(canvas,cv.COLOR_BGR2GRAY)
        _, imgInv = cv.threshold(imgGray,50,255,cv.THRESH_BINARY_INV)
        imgInv = cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
        img = cv.bitwise_and(img,imgInv)
        img = cv.bitwise_or(img,canvas)

        
        img[0:125,0:1280] = header
        # img = cv.addWeighted(img,0.9,canvas,0.9,0) #not actual, but both iamges are blended
        # cv.imshow("canvas",canvas)
        cv.imshow("img",img)


        key = cv.waitKey(1) & 0xFF
        if key == ord('d'): #exit
            break
        if key == ord('c'): #clear canvas
            canvas = None

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
