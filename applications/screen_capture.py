from imutils.video import WebcamVideoStream
import argparse
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from PIL import Image
from mss import mss

# mon = {'left': 160, 'top': 160, 'width': 1000, 'height': 1000}

# with mss() as sct:
#     while True:
#         screenShot = sct.grab(mon)
#         img = Image.frombytes(
#             'RGB', 
#             (screenShot.width, screenShot.height), 
#             screenShot.rgb, 
#         )
#         cv2.imshow('Screen Capture', np.array(img))
#         if cv2.waitKey(33) & 0xFF in (
#             ord('q'), 
#             27, 
#         ):
#             break

classifier = load_model('./resnet50_78339.h5')
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
class_labels = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

color = (0, 255, 0) # bounding box color.

# This defines the area on the screen.
mon = {'top' : 10, 'left' : 10, 'width' : 1500, 'height' : 1000}
sct = mss()

while True :
    screenshot = sct.grab(mon)
    frame = Image.frombytes( 'RGB', (screenshot.width, screenshot.height), screenshot.rgb )
    frame = np.array(frame)
    # image = image[ ::2, ::2, : ] # can be used to downgrade the input
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(frame,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=frame[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)
            preds=classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    
    cv2.imshow ('Emotion Detector', frame)
    if cv2.waitKey ( 1 ) & 0xff == ord( 'q' ) :
        cv2.destroyAllWindows()
        break