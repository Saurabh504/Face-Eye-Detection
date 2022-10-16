# There is a variety of computer applications that identify the eye in digital images, 
# like- security systems, criminal identification, healthcare, and so on.
# The eye-detection algorithms focus on the detection of the frontal human eye. 
# The Python OpenCV library functions are mainly aimed at real-time computer vision. 
# It is mainly used to do all the operations for image processing as well as detect objects.
# OpenCV already contains many pre-trained classifiers for faces, eyes, smiles, etc.

# Importing the required libraries

import cv2
import numpy as np

# creating a cascade
face_cascade = cv2.CascadeClassifier('E:\Projects\haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('E:\Projects\haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

# Simulating the process
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.4, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,1), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(40) & 0xff
    if k== 37:
        break

cap.release()
cv2.destroyAllWindows()
