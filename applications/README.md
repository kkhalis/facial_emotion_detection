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
```
python camera.py
```

## Screen Capture
```
python screen_capture.py
```

## Hosting on Flask
```
python server.py
```