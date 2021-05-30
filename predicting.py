from typing import Counter
import cv2 as cv
import mediapipe as mp
from mediapipe.python.packet_creator import create_image_frame
import numpy as np
import math as m
from sklearn.metrics import pairwise
import os
from keras.models import model_from_json
import operator
import pyautogui as pg
# import matplotlib as plt

mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 1, 0.65, 0.5)


# Loading the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

# Category dictionary
# categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE'}

canvas = None
x1,y1 = 0,0

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv.VideoCapture(0)

# frame_count = 0

# prev_frame = img


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

        img = cv.flip(img, 1)
        img = drawHands(img)

        lmLst = lmList(img)

        height, width = img.shape[:2]
        
        if height == 0 or width == 0:
            continue

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

            if fingers[1] == 1 :   # Middle Finger
                mx, my = lmLst[12][1], lmLst[12][2]
            elif fingers[0] == 1 : # Index Finger
                mx, my = lmLst[8][1], lmLst[8][2]
            elif fingers[2] == 1 : # Ring Finger
                mx, my = lmLst[16][1], lmLst[16][2]
            elif fingers[3] == 1 : # Pinky Finger
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
            
            # Create a circle with 1% radius of the max euclidean distance
            radius = int(max_distance)

            # Now grab an ROI of only that circle
            #circular_roi = np.zeros(img.shape, dtype="uint8")

            x1, y1 = (cX - radius), (cY + radius)
            x2, y2 = (cX + radius), (cY - radius) 
            
            roi = img[y2:y1,x1:x2]
            height, width = roi.shape[:2]
        
            if height <= 0 or width <= 0:
                continue
            roi = cv.resize(roi, (128,128))


            print("shape ",roi.shape)

            cv.imshow("ROI", roi)

            roi = cv.resize(roi, (64, 64)) 
            roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

            _, test_image = cv.threshold(roi, 120, 255, cv.THRESH_BINARY)
            cv.imshow("test", test_image)

            result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
            # print(type(result))
            print("result", result)

                # {'Lshape': 0, 'broFist': 1, 'fiveF': 2, 'okay': 3, 'palm': 4, 'twoF': 5}

            prediction = {#'none': result[0][0], 
                        # 'fiveF': result[0][1], 
                        #'hands': result[0][2],
                        'broFist': result[0][0],
                        'palm': result[0][1],
                        'twoF': result[0][2],
                        #'Lshape': result[0][2]}
            }
                        
            # Do whatever u wanna do here
            # if prediction.get('Lshape'):
            #     cv.putText(img,f"L-shaped hand",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)

            if prediction.get('broFist'):
                cv.putText(img,f"Playing current track",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)
                pg.press("playpause")

            # if prediction.get('fiveF'):
            #     cv.putText(img,f"Playing current track",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)
                # pg.press("playpause")

            # if prediction.get('okay'):
            #     cv.putText(img,f"Okay",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)

            if prediction.get('palm'):
                cv.putText(img,f"Pausing current track",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)
                pg.press("pause")

            # if prediction.get('twoF'):
                # cv.putText(img,f"Two Fingers",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)

           



            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            
            cv.putText(img, prediction[0][0], (10, 120), cv.FONT_HERSHEY_PLAIN, 2, (31,240,31), 2)    
            cv.imshow("img", img)
                
                
        # img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # img = cv.add(img,canvas) #to see the result in img too 
        # stacked = np.hstack((img,canvas))
        # cv.imshow('Beautiful Gestures', cv.resize(stacked,None,fx=1.2,fy=1.2))

        # frame_count += 1
        # prev_frame = img.copy()
        # success, img = cap.read() # img = crurent img

        key = cv.waitKey(1) & 0xFF
        if key == ord('d'):
            # print(f"Frames found: {frame_count}")
            break
        if key == ord('c'): #clear canvas
            canvas = None


    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()











# < TEST 2 > #


# from typing import Counter
# import cv2 as cv
# import mediapipe as mp
# from mediapipe.python.packet_creator import create_image_frame
# import numpy as np
# import math as m
# from sklearn.metrics import pairwise
# import os
# from keras.models import model_from_json
# import operator
# import pyautogui as pg
# # import matplotlib as plt

# # Loading the model
# json_file = open("model-bw.json", "r")
# model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(model_json)
# # load weights into new model
# loaded_model.load_weights("model-bw.h5")
# print("Loaded model from disk")

# # Category dictionary
# # categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE'}

# canvas = None
# x1,y1 = 0,0

# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands

# cap = cv.VideoCapture(0)

# # frame_count = 0

# # prev_frame = image


# def circles(x1, y1, x2, y2, r1, r2):
#     distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2); 
#     radSumSq = (r1 + r2) * (r1 + r2); 
#     if (distSq == radSumSq):
#         return 1 
#     elif (distSq > radSumSq):
#         return -1 
#     else:
#         return 0 

# with mp_hands.Hands(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as hands:
#     while cap.isOpened():
#         success, image = cap.read() # image = crurent frame
#         if not success:
#             print("Ignoring empty camera image.")
#             # If loading a video, use 'break' instead of 'continue'.
#             continue
        
#         if canvas is None:
#             canvas = np.zeros_like(image)



#         # Flip the image horizontally for a later selfie-view display, and convert
#         # the BGR image to RGB.

#         image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)

#         # To improve performance, optionally mark the image as not writeable to
#         # pass by reference.


#         #image = cv.cvtColor(image, cv.COLOR_HSV2RGB)

#         image.flags.writeable = True
#         results = hands.process(image)

#         image_height, image_width, _ = image.shape

#         # Draw the hand annotations on the image.
#         image.flags.writeable = True
#         image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image, hand_landmarks, mp_hands.HAND_CONNECTIONS) #to see the skeleton 

  
#                 x2 = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
#                 y2 = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)


#                 x3 = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)
#                 y3 = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

#                 # min_dst = m.sqrt(((x3-x2) * (x3-x2)) + ((y3 - y1) * (y3 - y1)))
#                 # print(min_dst)

#                 # Area of hand:
#                 #Top
#                 midX = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)
#                 midY = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)
#                 avgMid = ( midX + midY ) / 2

#                 #Down
#                 WristX = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)
#                 WristY = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
#                 avgWrist = ( WristX + WristY ) / 2

#                 # mid_point = (avgMid + avgWrist) / 2
                
#                 #Right
#                 pinX = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)
#                 pinY = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)
#                 majorAxis = ( pinX + pinY ) / 2

#                 #Left   
#                 thumbX = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)
#                 thumbY = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)
#                 minorAxis = ( thumbX + thumbY ) / 2

#                 top    = (midX,midY)
#                 bottom = (WristX,WristY)
#                 left   = (thumbX,thumbY)
#                 right  = (pinX,pinY)

#                 # In theory, the center of the hand is half way between the top and bottom and halfway between left and right
#                 cX = (left[0] + right[0]) // 2
#                 cY = (top[1] + bottom[1]) // 2
                
#                 distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
                
#                 # Grab the largest distance
#                 max_distance = distance.max()
#                 print(max_distance)
                
#                 # Create a circle with 1% radius of the max euclidean distance
#                 radius = int(max_distance)
#                 circumference = (2 * np.pi * radius)

#                 # Now grab an ROI of only that circle
#                 #circular_roi = np.zeros(image.shape, dtype="uint8")

#                 iX = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
#                 iY = (int)(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)


#                 x1, y1 = (cX - radius), (cY + radius)
#                 x2, y2 = (cX + radius), (cY - radius) 
                
#                 roi = image[y2:y1,x1:x2]
#                 roi = cv.resize(roi, (128,128))

#                 # print("shape ",roi.shape)

#                 cv.imshow("ROI", roi)

#                 roi = cv.resize(roi, (64, 64)) 
#                 roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

#                 _, test_image = cv.threshold(roi, 120, 255, cv.THRESH_BINARY)
#                 cv.imshow("test", test_image)

#                 result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
#                 # print(type(result))
#                 print(result)

#                  # {'Lshape': 0, 'broFist': 1, 'fiveF': 2, 'okay': 3, 'palm': 4, 'twoF': 5}

#                 prediction = {'Lshape': result[0][0], 
#                             'broFist': result[0][1], 
#                             'fiveF': result[0][2],
#                             'okay': result[0][3],
#                             'palm': result[0][4],
#                             'twoF': result[0][5]}
                
#                 # Do whatever u wanna do here
#                 if prediction.get('Lshape'):
#                     cv.putText(image,f"L-shaped hand",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)
#                 if prediction.get('broFist'):
#                     cv.putText(image,f"Bro Fist",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)
#                 if prediction.get('fiveF'):
#                     cv.putText(image,f"Playing current track",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)
#                     pg.press("playpause")

#                 if prediction.get('okay'):
#                     cv.putText(image,f"Okay",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)
#                 if prediction.get('palm'):
#                     cv.putText(image,f"Pausing current track",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)
#                     pg.press("pause")

#                 if prediction.get('twoF'):
#                     cv.putText(image,f"Two Fingers",(10,150),cv.FONT_HERSHEY_PLAIN,1,(0,240,31), 1)



#                 prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
                
#                 cv.putText(image, prediction[0][0], (10, 120), cv.FONT_HERSHEY_PLAIN, 2, (31,240,31), 2)    
#                 cv.imshow("image", image)
                
                
#         # image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
#         # image = cv.add(image,canvas) #to see the result in image too 
#         # '''
#         # This ^^ sometimes causes error:
#         # error: (-209:Sizes of input arguments do not 
#         # match) The operation is neither 'array op array' (where arrays have the same size and the same number of channels), nor 'array op scalar', nor 'scalar op array' in function 'cv::arithm_op'
#         # '''
#         # stacked = np.hstack((image,canvas))
#         # cv.imshow('Beautiful hands', cv.resize(stacked,None,fx=1.2,fy=1.2))


#         key = cv.waitKey(1) & 0xFF
#         if key == ord('d'):
#             # print(f"Frames found: {frame_count}")
#             break

# cap.release()
# cv.destroyAllWindows()





# < TEST 1 > #

# from keras.models import model_from_json
# import numpy as np
# import operator
# import cv2 as cv

# # Loading the model
# json_file = open("model-bw.json", "r")
# model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(model_json)
# # load weights into new model
# loaded_model.load_weights("model-bw.h5")
# print("Loaded model from disk")

# cap = cv.VideoCapture(0)

# while True:
#     _, frame = cap.read()
#     # Simulating mirror image
#     frame = cv.flip(frame, 1)
    
#     # Got this from collect-data.py
#     # Coordinates of the ROI
#     x1 = int(0.5*frame.shape[1])
#     y1 = 10
#     x2 = frame.shape[1]-10
#     y2 = int(0.5*frame.shape[1])
#     # Drawing the ROI
#     # The increment/decrement by 1 is to compensate for the bounding box
#     cv.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
#     # Extracting the ROI
#     roi = frame[y1:y2, x1:x2]
    
#     # Resizing the ROI so it can be fed to the model for prediction
#     roi = cv.resize(roi, (64, 64)) 
#     roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
#     _, test_image = cv.threshold(roi, 120, 255, cv.THRESH_BINARY)
#     cv.imshow("test", test_image)
#     # Batch of 1
#     result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
#     prediction = {'broFist': result[0][0], 
#                   'palm': result[0][1], 
#                   'twoF': result[0][2]}
#     # Sorting based on top prediction
#     prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
#     # Displaying the predictions
#     cv.putText(frame, prediction[0][0], (10, 120), cv.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)    
#     cv.imshow("Frame", frame)
    
#     key = cv.waitKey(1) & 0xFF
#     if key == ord('d'):
#         break
        
 
# cap.release()
# cv.destroyAllWindows()





