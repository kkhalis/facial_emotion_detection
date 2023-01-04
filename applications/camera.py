import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

print('Loading classifier here...')
classifier = load_model('./resnet50_78339.h5')
class_labels = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE,2)

if not cap.isOpened():
    print('Camera not open')
    exit(0)

while True:
    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        cv2.imshow('Emotion Detector',frame)
        if cv2.waitKey(1) == ord('q'): 
            break
cap.release()
cv2.destroyAllWindows()
