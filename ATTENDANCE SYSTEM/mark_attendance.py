import cv2
import numpy as np
import pickle
import csv
from datetime import datetime

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.yml")

with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

cap = cv2.VideoCapture(0)
marked = set()

with open('attendance.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))
            label_id, confidence = model.predict(roi)
            if confidence < 70:
                name = labels[label_id]
                if name not in marked:
                    now = datetime.now()
                    writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
                    marked.add(name)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) == 13:
            break

cap.release()
cv2.destroyAllWindows()
