import os
import cv2
import numpy as np
from PIL import Image

USERS_DIR = "users/"

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")

# gets images and labels for training the face recognition model
def getTrainingData(userDirPath):
    imagePaths = [os.path.join(userDirPath, f) for f in os.listdir(userDirPath)]
    faceSamples = []
    userIds = []
    for imagePath in imagePaths:
        # convert image to grayscale
        PIL_image = Image.open(imagePath).convert("L")
        # convert grayscale image to array
        numpy_image = np.array(PIL_image, "uint8")
        user_id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = face_detector.detectMultiScale(numpy_image, 1.1, 5)
        for (x, y, w, h) in faces:
            faceSamples.append(numpy_image[y:y+h,x:x+w])
            userIds.append(user_id)
            
    return faceSamples, userIds

print("\n[INFO] Training face recognition model. This might take a few seconds. Please wait...")
faces, userIds = getTrainingData(USERS_DIR) 
recognizer.train(faces, np.array(userIds))
# save model
recognizer.write("models/model.yml")

print(f"\n[INFO] {len(np.unique(userIds))} faces trained")