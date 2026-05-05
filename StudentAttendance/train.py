import cv2
import numpy as np
from os import listdir
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / 'student_images'
dirs = [d for d in listdir(data_path)]
faces = []
labels = []
label_map = {}

label_id = 0
for folder in dirs:
    label_map[label_id] = folder
    folder_path = data_path / folder
    image_paths = [f for f in listdir(folder_path) if (folder_path / f).is_file()]
    for img_name in image_paths:
        img_path = folder_path / img_name
        img = cv2.imread(str(img_path), 0)
        faces.append(np.asarray(img, dtype=np.uint8))
        labels.append(label_id)
    label_id += 1

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(faces), np.asarray(labels))
model.save(str(BASE_DIR / "face_model.yml"))

# Save label map
import pickle
with open(BASE_DIR / "labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("Model trained and saved.")
