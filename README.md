# Face recognition using Python and OpenCV
You need to manually create 2 folders: "users" and "models" before running any of the Python files.

## Capturing a user face
Run the 01_create_users.py file and enter a name for the user. The user name will be used to create a folder under the "users" directory with the user images. The process captures 30 images by default, but this number can be changed using the MAX_CAPTURES constant.

## Training the face recognition model
Run the 02_face_training.py file. 
This process will output a "model.yml" file in the "models" folder.

## Performing face recognition
Run the 03_face_recognition.py file. 
The process will attempt to recognize the face using the camera, if the face was recorded in the model before.
