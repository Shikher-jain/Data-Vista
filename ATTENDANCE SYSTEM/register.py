import cv2
import os

# Load Haar Cascade
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_classifier.empty():
    print("Error: Haarcascade XML file not loaded properly.")
    exit()

def capture_face(name):
    cam = cv2.VideoCapture(0)
    count = 0
    path = 'student_images/' + name
    os.makedirs(path, exist_ok=True)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Skip small detections (optional)
            # if w * h < 10000:
            #     continue

            count += 1
            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            cv2.imwrite(f"{path}/{count}.jpg", face)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Images Captured: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Registering Face", frame)

        if cv2.waitKey(1) == 13 or count >= 20:  # press Enter key or 20 faces
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f" Face capture complete for {name}.")
    print(" Now run: python train.py to train your recognizer.")

if __name__ == "__main__":
    username = input("Enter student name: ")

    capture_face(username)