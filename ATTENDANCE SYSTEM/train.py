import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'student_images/'
dirs = [d for d in listdir(data_path)]
faces = []
labels = []
label_map = {}

label_id = 0
for folder in dirs:
    label_map[label_id] = folder
    image_paths = [f for f in listdir(data_path + folder) if isfile(join(data_path + folder, f))]
    for img_name in image_paths:
        img_path = data_path + folder + '/' + img_name
        img = cv2.imread(img_path, 0)
        faces.append(np.asarray(img, dtype=np.uint8))
        labels.append(label_id)
    label_id += 1

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(faces), np.asarray(labels))
model.save("face_model.yml")

# Save label map
import pickle
with open("labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("Model trained and saved.")
