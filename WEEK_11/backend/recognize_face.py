import face_recognition
import pickle
import cv2
import numpy as np
import csv
from datetime import datetime

with open("models/face_model.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

cap = cv2.VideoCapture(0)

attendance_file = "attendance/attendance.csv"

marked = set()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)

    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for encoding, location in zip(face_encodings, face_locations):

        distances = face_recognition.face_distance(known_encodings, encoding)

        best_match_index = np.argmin(distances)

        name = "Unknown"

        if distances[best_match_index] < 0.45:
            name = known_names[best_match_index]

            if name not in marked:

                now = datetime.now().strftime("%H:%M:%S")

                with open(attendance_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, now])

                marked.add(name)

        top, right, bottom, left = location

        cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)

        cv2.putText(frame,name,(left,top-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,(0,255,0),2)

    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()