import cv2
import numpy as np
import os
import random

names = ['None', 'Rid', 'Nail', 'Arsyad']
data_path = "dataset/"
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# List image paths, grouped sesuai person id
files_by_id = {}
for fname in os.listdir(data_path):
    pid = int(fname.split("-")[1])
    files_by_id.setdefault(pid, []).append(data_path + fname)

# Split 10 random images per person untuk tes, sisanya buat training
train_files, test_files = [], []
for pid, files in files_by_id.items():
    random.shuffle(files)
    test_files += [(f, pid) for f in files[:10]]
    train_files += [(f, pid) for f in files[10:]]

def loadFace(fpath):
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img)
    return cv2.resize(img, (200,200))

# Train modelnya
recognizer = cv2.face.LBPHFaceRecognizer.create()
train_faces = [loadFace(f) for f, pid in train_files]
train_ids = np.array([pid for f, pid in train_files])
recognizer.train(train_faces, train_ids)

# Testing
correct = 0
for fpath, true_id in test_files:
    face = loadFace(fpath)
    pred_id, confidence = recognizer.predict(face)
    is_correct = (pred_id == true_id)
    correct += is_correct
    print(f"{fpath}: true={names[true_id]} pred={names[pred_id]} conf={confidence:.0f} {'OK' if is_correct else 'WRONG'}")

print(f"\nAccuracy: {correct}/{len(test_files)}")
