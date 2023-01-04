# Guide to using sample applications
This guide is meant to demonstrate on using the fitted model in the applications listed below to run facial emotion detection:

* Web Camera
* Screen Capture
* Hosting on Flask Server

# Types of Application
There are 3 different types of test applications here to use and play around. This guide will show the python commands required to launch each application.

## Prequisites
In order to run these applications, please ensure your python3 (<=3.8.8) environment has the following python packages installed:
* numpy
* opencv (opencv-python)
* tensorflow
* PIL
* mss (Required for screen capture)
* Flask (Required for server hosting via Flask)

## Web Camera
Navigate to the `applications` directory and run the following command in your terminal:
```
python camera.py
```
This will launch a python application and activate the primary web camera, indicated as `cv2.VideoCapture(0)` in code. Should you have additional cameras, you may need to configure this portion to utilise the correct camera.

## Screen Capture
Navigate to the `applications` directory and run the following command in your terminal:
```
python screen_capture.py
```
This will capture a region of space where:
* x = 10
* y = 10

with a width of 1500 pixels and height of 1000 pixels. The screen capture application will detect any faces, and use the model to predict the emotion shown in the faces on screen.

## Hosting on Flask
Navigate to the `applications` directory and run the following command in your terminal:
```
python server.py
```

This will host a Flask server instance on your computer, and render the `index.html` inside the template folder. `index.html` will call the function in `server.py` to render and detect frames captured by the webcam to detect the face, and then predict the emotion. Do note that this is not tested on mobile devices yet.

# Feedback and Findings
The applications are computationally intensive. There is a short lag when there suddenly are faces detected on the web camera or screen. This is likely due to the CPU/GPU attempting to detect and predict face emotions in real-time. When running Flask on a minimal AWS instance, a minimum of 2GB memory is required in order for the server run without issues. This application is not developed for asynchronous usage, and will definitely require extensive computational power to cater to multiple hosts.