from flask import Flask, render_template, Response
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2

face_c = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
model = load_model('./models/resnet50_78627.h5')
class_labels = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
font = cv2.FONT_HERSHEY_SIMPLEX
app = Flask(__name__)

cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_c.detectMultiScale(gray_fr, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray=frame[y:y+h,x:x+w]
                roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                if np.sum([roi_gray])!=0:
                    roi=roi_gray.astype('float')/255.0
                    roi=img_to_array(roi)
                    roi=np.expand_dims(roi,axis=0)
                    preds=model.predict(roi)[0]
                    label=class_labels[preds.argmax()]
                    label_position=(x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                else:
                    cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)