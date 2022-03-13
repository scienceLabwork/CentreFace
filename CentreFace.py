import cv2
import mediapipe as mp
import math
import time

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    height, width, _ = image.shape
    if not success:
        continue    
    image = cv2.flip(image, 1)
    resizedcropped = image
    cv2.imshow('Centre Face', resizedcropped)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        l = int(detection.location_data.relative_bounding_box.xmin*width) - 220
        r = int(detection.location_data.relative_bounding_box.ymin*height) - 200
        l1 = int(detection.location_data.relative_bounding_box.width*width) + 420
        r1 = int(detection.location_data.relative_bounding_box.height*height) + 330

        if(l<0):
            l=0
        if(l>width):
            l=width

        if(r<0):
            r=0
        if(r>height):
            r=height

        cropped = image[r:r+r1, l:l+l1]
        resizedcropped = cv2.resize(cropped, (width-200, height))
        cv2.imshow('Centre Face', resizedcropped)

        # TO SHOW CORDNATE STRUCTURE OF FACE
        # cv2.circle(image, (l,r), 10, (0,0,255), -1)
        # cv2.circle(image, (l+l1,r), 10, (0,0,255), -1)

        # cv2.line(image, (l,r), (l+l1,r), (225,255,0), 2)
        # cv2.line(image, (l,r), (l,r+r1), (0,255,0), 2)

        # cv2.circle(image, (l,r+r1), 10, (0,0,255), -1)
        # cv2.circle(image, (l+l1,r+r1), 10, (0,0,255), -1)

        # cv2.line(image, (l+l1,r), (l+l1,r+r1), (0,255,0), 2)
        # cv2.line(image, (l,r+r1), (l+l1,r+r1), (0,255,0), 2)

    cv2.imshow('Normal View', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()