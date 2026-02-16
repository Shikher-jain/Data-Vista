import cv2
import os

# Load Haar Cascade
# face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_classifier = cv2.CascadeClassifier(os.path.abspath('haarcascade_frontalface_default.xml'))
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

        # Default: red border for capturing
        border_color = (0, 0, 255)  # Red
        face_captured = False

        for (x, y, w, h) in faces:
            count += 1
            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            cv2.imwrite(f"{path}/{count}.jpg", face)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green rectangle for face
            cv2.putText(frame, f"Images Captured: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            face_captured = True

        if face_captured:
            cv2.putText(frame, "Face Captured!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            border_color = (0, 255, 0)  # Green
        else:
            cv2.putText(frame, "Capturing...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw border around the whole frame
        thickness = 4
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w-1, h-1), border_color, thickness)

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