"""
Python code to blur russian car plates automatically. 
This technique was introduced during an lecture at https://www.udemy.com/course/python-for-computer-vision-with-opencv-and-deep-learning/.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


#Using OpenCV built-in classifier for russian license plates
plate_cascade = cv2.CascadeClassifier('../data/haarcascade_russian_plate_number.xml')

def blur_plate(img):
    blurred_img = img.copy()
    #Detecting plates' vertices using detectMultiscale 
    plate_vertex = plate_cascade.detectMultiScale(blurred_img, scaleFactor=1.3, minNeighbors=3) 
    
    for (x,y,w,h) in plate_vertex: 
        roi = blurred_img.copy() #Remaking the roi in case of mutiple plates in a single frame
        roi = roi[y:y+h,x:x+w]
        blurred_roi = cv2.medianBlur(roi, 7) #Blurring with medianBlur
        
        #Re-drawing blurred plates on our frame
        blurred_img[y:y+h, x:x+w] = blurred_roi
        
    return blurred_img


#Initializing camera (0 = default) and its parameters
cap = cv2.VideoCapture(0)

while True:
    #Capture each frame (image)
    ret, frame = cap.read()
    if not ret:
        continue

    #Applying our funcion to every image and showing
    cv2.imshow('frame', blur_plate(frame))
    
    #Quitting when letter 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Releasing the capture and destroying the windows created
cap.release()
cv2.destroyAllWindows()