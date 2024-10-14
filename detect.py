import cv2 
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from playsound import playsound


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
  
cap = cv2.VideoCapture(-1)


# fontScale
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1   
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

# #Load model
model = load_model("/home/sky/Desktop/driver_drowsiness/cnn_m_dd.h5")

# loop runs if capturing has been initialized.
while 1: 
  
    # reads frames from a camera
    ret, img = cap.read() 
      
    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        im1 = cv2.resize(roi_color, (128, 128),interpolation = cv2.INTER_LINEAR)
        im1 = np.expand_dims(im1, axis=0)
        class_ = np.argmax(model.predict(im1,verbose=0))
        
        eyes = eye_cascade.detectMultiScale(roi_gray) 
        
        if class_ == 1:
            PREDICTIONS = "level 1"
        elif class_ == 2:            
            PREDICTIONS = "level 2"
        elif len(eyes) <= 1:
            PREDICTIONS = "Level 3"
            # cv2.putText(img, PREDICTIONS,  (10, 50), font, fontScale, color, thickness, cv2.LINE_AA)
            playsound("/home/sky/Desktop/driver_drowsiness/detection/alert.mp3")
        else:
            PREDICTIONS = "level 0"
        
        cv2.putText(img,PREDICTIONS ,  (10, 50), font, fontScale, color, thickness, cv2.LINE_AA)
            
    # Display an image in a window
    cv2.imshow('img',img)
  
    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
  
# Close the window
cap.release()
# De-allocate any associated memory usage
cv2.destroyAllWindows() 
