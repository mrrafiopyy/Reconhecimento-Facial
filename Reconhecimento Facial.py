import face_recognition
import cv2
import numpy as np

global acesso

from time import sleep

RED = "\033[1;31m"
CYAN = "\033[1;96m"
GREEN = "\033[1;92m"

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(1)

# Load a sample picture and learn how to recognize it.
rafael1_image = face_recognition.load_image_file("Me.jpg")
rafael1_face_encoding = face_recognition.face_encodings(rafael1_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    rafael1_face_encoding,

    ]
known_face_names = [
    "Rafael Felipe",

]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    if 'Rafael Felipe' in face_names:
        print(GREEN + "Acesso Autorizado." + CYAN + "Bem vindo de volta Rafael!")
        break
    else:
        print(RED + 'Acesso negado.')
        sleep(3)

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
