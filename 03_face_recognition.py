import cv2
import numpy as np
import os
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/model.yml")

face_detector = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
cv2.startWindowThread()

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
# map ids to names
names = ["Unknown", "Nicholas", "Lani", "Rodolfo"]

while True:
    im = picam2.capture_array()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0))
    
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        user = names[id]
        
        conf = " {0}%".format(round(100-confidence))
        cv2.putText(im, user, (x+5, y-5), font, 1, (255,255,255), 2)
        cv2.putText(im, conf, (x+5, y+h-5), font, 1, (255,255,255), 2)

    cv2.imshow("Camera", im)
    k = cv2.waitKey(100) % 256
    # Press ESC to exit
    if k == 27:
        print("ESC pressed, exiting...")
        break

cv2.destroyAllWindows()