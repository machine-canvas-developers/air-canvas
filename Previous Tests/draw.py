import cv2 as cv
import numpy as np
# import time

#This variable determines if we want to load color range from memory or use the ones defined here.
load_from_disk = True
# If true then load color range from memory
if load_from_disk:
    hsv_value = np.load('values.npy')

# <=== Video ===> #
cap = cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# kernel = np.ones((5,5),np.uint8) #define the shape of array (uint8 => dtype of image)

# Initializing the canvas on which we will draw upon
canvas = None
# Initilize x1,y1 points
x1,y1=0,0
# Threshold for noise
noiseth = 500

while True:
    _, frame = cap.read()
    frame = cv.flip( frame, 1 ) #recommended to flip the video else ur left hand will become ur right hand in feed
    
    # Initialize the canvas as a black image of the same size as the frame.
    if canvas is None:
        canvas = np.zeros_like(frame)
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # If you're reading from memory then load the upper and lower ranges from there
    if load_from_disk:
        lower_range = hsv_value[0]
        upper_range = hsv_value[1]
            
    # Otherwise define your own custom values for upper and lower range.
    else:           
        # lower_range  = np.array([134, 20, 204])
        # upper_range = np.array([179, 255, 255])
        pass
    
    mask = cv.inRange(hsv, lower_range, upper_range)
    
    # Operations to get rid of the noise
    # canny = cv.Canny(frame,123,175)

    mask = cv.GaussianBlur(mask,(7,7),cv.BORDER_DEFAULT) #blurred the image
    mask = cv.erode(mask,(7,7),iterations=1) # eroded the image
    mask = cv.dilate(mask,(7,7),iterations=1) # and dilated the image


    # <===== Find Contours ====> #
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Make sure there is a contour present and also its size is bigger than the noise threshold.
    if contours and cv.contourArea(max(contours, key = cv.contourArea)) > noiseth:
        c = max(contours, key = cv.contourArea)    
        x2,y2,w,h = cv.boundingRect(c)
        
        # If there were no previous points then save the detected x2,y2 coordinates as x1,y1. 
        # This is true when we writing for the first time or when writing again when the pen had disappeared from view.
        if x1 == 0 and y1 == 0:
            x1,y1= x2,y2
            
        else:
            # Draw the line on the canvas
            canvas = cv.line(canvas, (x1,y1),(x2,y2), [56,0,255], 4)
        
        # After the line is drawn the new points become the previous points.
        x1,y1= x2,y2
    else:
        # If there were no contours detected then make x1,y1 = 0
        x1,y1 =0,0
    
    # Merge the canvas and the frame.
    frame = cv.add(frame,canvas)
    
    # Optionally stack both frames and show it.
    stacked = np.hstack((canvas,frame))
    cv.imshow('Air-Canvas',cv.resize(stacked,None,fx=0.6,fy=0.6)) #stacked and resized the frame to 60% of size 
    k = cv.waitKey(1) & 0xFF

    if k == ord('d'): # when d key is pressed exit out of the window
        break
    if k == ord('c'): # when c key is pressed, clear the canvas
        canvas = None
cv.destroyAllWindows()
cap.release()
