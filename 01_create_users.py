import os
import cv2

MAX_CAPTURES = 30
USERS_DIR = "users/"

def validate_input(message, input_type=str):
    while True:
        try:
            return input_type(input(message))
        except:
            pass
        
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set width
cam.set(4, 480) # set height

# min size for face detection
minWidth = 0.1 * cam.get(3)
minHeight = 0.1 * cam.get(4)

# get user id and name
user_id = validate_input("Enter user id and press <return> ", int)
user_name = validate_input("Enter user name and press <return> ", str)
print("\n[INFO] Initializing face capture. Please look at the camera and wait for images to be saved to disk...")

face_detector = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
cv2.startWindowThread()

# capture user images up to MAX_CAPTURES
count = 0
while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minWidth),int(minHeight)))

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0))
        count += 1
        # save image
        imgName = f"{USERS_DIR}{user_name}.{user_id}.{count}.jpg"
        cv2.imwrite(imgName, gray[y:y+h,x:x+w])
        print(f"{imgName} saved")
        
    cv2.imshow("Camera", im)
    k = cv2.waitKey(100) % 256
    # Press ESC to exit
    if k == 27 or count >= MAX_CAPTURES:
        print("Exiting...")
        break

cam.release()    
cv2.destroyAllWindows()