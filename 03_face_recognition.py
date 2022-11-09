import cv2
import numpy as np
import os

MIN_CONFIDENCE = 50

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set width
cam.set(4, 480) # set height

# min size for face detection
minWidth = 0.1 * cam.get(3)
minHeight = 0.1 * cam.get(4)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/model.yml")

face_detector = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
cv2.startWindowThread()

font = cv2.FONT_HERSHEY_SIMPLEX

# map ids to names (0 - Unknown, 1 - Daniel, ...)
names = ["Unknown", "Rudy", "Jerry", "Lani"]
exitLoop = False

while not exitLoop:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minWidth),int(minHeight)))
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0))
    
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # the confidence returned by the recognizer is inverted 0 => 100 (perfect match)
        confidence = 100 - confidence
        # check if confidence is above our min level
        if confidence > MIN_CONFIDENCE:
            user = names[id]
            print(f"{user} successfully identified with a confidence level of {round(confidence)}%")
            #exitLoop = True
        else:
            user = "Unknown"
            
        conf = " {0}%".format(round(confidence))
        cv2.putText(im, user, (x+5, y-5), font, 1, (255,255,255), 2)
        cv2.putText(im, conf, (x+5, y+h-5), font, 1, (255,255,255), 2)

    cv2.imshow("Camera", im)
    k = cv2.waitKey(100) % 256
    # Press ESC to exit
    if k == 27:
        print("ESC pressed, exiting...")
        break

cam.release()
cv2.destroyAllWindows()