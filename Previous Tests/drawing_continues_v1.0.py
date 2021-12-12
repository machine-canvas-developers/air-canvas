from typing import Counter
import cv2 as cv
import mediapipe as mp
from mediapipe.python.packet_creator import create_image_frame
import numpy as np
import math as m
from sklearn.metrics import pairwise
# import matplotlib as plt

canvas = None
x1,y1 = 0,0

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv.VideoCapture(0)

# frame_count = 0

# prev_frame = image


def circles(x1, y1, x2, y2, r1, r2):
    distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2); 
    radSumSq = (r1 + r2) * (r1 + r2); 
    if (distSq == radSumSq):
        return 1 
    elif (distSq > radSumSq):
        return -1 
    else:
        return 0 

with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read() # image = crurent frame
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        
        if canvas is None:
            canvas = np.zeros_like(image)

        # frame_diff = cv.absdiff(image,prev_frame)
        # cv.imshow("frame diff",frame_diff)


        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        

        image.flags.writeable = True
        results = hands.process(image)

        image_height, image_width, _ = image.shape

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS) #to see the skeleton 

                image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
                h, s, v = image[:, :, :], image[:, :,:], image[:, :, 0]


                # print(
                #     f'Index finger tip coordinates: (',
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                # )
                # contours = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
                # lst.append(contours)

                # image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
                x2 = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
                y2 = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)


                x3 = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)
                y3 = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                # min_dst = m.sqrt(((x3-x2) * (x3-x2)) + ((y3 - y1) * (y3 - y1)))
                # print(min_dst)

                # Area of hand:
                #Top
                midX = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)
                midY = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)
                avgMid = ( midX + midY ) / 2

                #Down
                WristX = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)
                WristY = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                avgWrist = ( WristX + WristY ) / 2

                # mid_point = (avgMid + avgWrist) / 2
                
                #Right
                pinX = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)
                pinY = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)
                majorAxis = ( pinX + pinY ) / 2

                #Left   
                thumbX = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)
                thumbY = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)
                minorAxis = ( thumbX + thumbY ) / 2

                top    = (midX,midY)
                bottom = (WristX,WristY)
                left   = (thumbX,thumbY)
                right  = (pinX,pinY)

                # In theory, the center of the hand is half way between the top and bottom and halfway between left and right
                cX = (left[0] + right[0]) // 2
                cY = (top[1] + bottom[1]) // 2
                # print(cX)
                # print(cY)

                # find the maximum euclidean distance between the center of the palm
                
                
                # Calculate the Euclidean Distance between the center of the hand and the left, right, top, and bottom.
                distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
                
                # Grab the largest distance
                max_distance = distance.max()
                # print(max_distance)
                
                # Create a circle with 1% radius of the max euclidean distance
                radius = int(0.2 * max_distance)
                circumference = (2 * np.pi * radius)

                # Now grab an ROI of only that circle
                circular_roi = np.zeros(image.shape[:2], dtype="uint8")
                
                # draw the circular ROI
                cir = cv.circle(circular_roi, (cX, cY), radius, 255, 10)

                # Index finger circle
                iX = x2
                iY = y2

                IndexCircle = cv.circle(circular_roi, (iX, iY), radius, 255, 10)
                # cv.imshow("Index finger circle", IndexCircle)

                # Thumb circle
                tX = thumbX
                tY = thumbY

                ThumbCircle = cv.circle(circular_roi, (tX, tY), radius, 255, 10)
                # cv.imshow("Thumb finger circle", ThumbCircle)
                
                t = circles(tX,tY,iX,iY,radius,radius)
                # if(t == 1):
                #     print("circles touch")
                # elif (t < 0):
                #     print("Circle not touch to each other.") 
                # else:
                #     print("Circle intersect to each other.") 

                # if (t < 0):
                #     print("Circle not touch to each other.") 

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



                # x = (100,100)
                # y = (200, 200)

                # contours, _ = cv.findContours(cnt,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                # cv.drawContours(image,contours,-1,(0,255,0), 3)
                # print(contours)
                # cv.line(canvas,(x1,y1),(x2,y2),(0,255,0),9)

        # image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        image = cv.add(image,canvas) #to see the result in image too 
        '''
        This ^^ sometimes causes error:
        error: (-209:Sizes of input arguments do not 
        match) The operation is neither 'array op array' (where arrays have the same size and the same number of channels), nor 'array op scalar', nor 'scalar op array' in function 'cv::arithm_op'
        '''
        stacked = np.hstack((image,canvas))
        cv.imshow('Beautiful hands', cv.resize(stacked,None,fx=1.2,fy=1.2))

        # frame_count += 1
        # prev_frame = image.copy()
        # success, image = cap.read() # image = crurent frame

        key = cv.waitKey(1) & 0xFF
        if key == ord('d'):
            # print(f"Frames found: {frame_count}")
            break
        if key == ord('c'): #clear canvas
            canvas = None

cap.release()
cv.destroyAllWindows()
