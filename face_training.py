import cv2
import numpy as np
import os

def checkDataset(path="dataset/"):
    if os.path.exists(path) and len(os.listdir(path)) != 0:
        return True
    return False

def organizeDataset(path="dataset/"):
    imagePath = [os.path.join(path, p) for p in os.listdir(path)]
    faces = []
    ids = np.array([], dtype='int')
    for i in imagePath:
        img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(i)[-1].split("-")[1])
        face = faceCascade.detectMultiScale(img)
        for (x, y, w, h) in face:
            #Resize and equalize brightness
            face_img = cv2.equalizeHist(img[y:y+h, x:x+w])
            face_img = cv2.resize(face_img, (200,200))
            faces.append(face_img)
            ids = np.append(ids, id)
    return faces, ids

if not checkDataset():
    print("Dataset gaada")
else:
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    print("Training faces")
    faces, ids = organizeDataset()
    recognizer.train(faces,ids)
    print("Finished")
    recognizer.write('faces-model.yml')
