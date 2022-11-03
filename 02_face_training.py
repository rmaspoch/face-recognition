import os
import cv2
import numpy as np
from PIL import Image

USERS_DIR = "users/"

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")

with os.scandir(USERS_DIR) as it:
    for entry in it:
        if entry.is_dir():
            print(entry.name)
            for f in os.listdir(USERS_DIR + entry.name):
                print("\t" + f)

    