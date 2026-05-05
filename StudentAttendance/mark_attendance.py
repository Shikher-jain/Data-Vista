import csv
from datetime import datetime
from pathlib import Path

import cv2
import pickle

BASE_DIR = Path(__file__).resolve().parent

face_cascade = cv2.CascadeClassifier(str(BASE_DIR / "haarcascade_frontalface_default.xml"))
if face_cascade.empty():
    print("Error: Haarcascade XML file not loaded properly.")
    raise SystemExit(1)

model = cv2.face.LBPHFaceRecognizer_create()
model.read(str(BASE_DIR / "face_model.yml"))

with open(BASE_DIR / "labels.pkl", "rb") as f:
    labels = pickle.load(f)

cap = cv2.VideoCapture(0)
marked = set()

with open(BASE_DIR / "attendance.csv", "a", newline="") as f:
    writer = csv.writer(f)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(
            frame,
            "Press Q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y : y + h, x : x + w]
            roi = cv2.resize(roi, (200, 200))
            label_id, confidence = model.predict(roi)

            if confidence < 70:
                name = labels[label_id]
                if name not in marked:
                    now = datetime.now()
                    writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
                    f.flush()
                    marked.add(name)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Attendance System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, ord("q"), ord("Q")):
            break

cap.release()
cv2.destroyAllWindows()
