import cv2

recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read("faces-model.yml")
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_COMPLEX

id = 0
names = ['None', 'Rid', 'Nail', 'Arsyad'] # ini sesuai index, Rid itu index 1, Nail index 2, Arsyad index 3
colors = [(128,128,128), (0,0,255), (0,255,0), (255,0,0)]  # ini buat pemetaan warna (sesuai index)
cap = cv2.VideoCapture(1)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    for (x,y,w,h) in faces:
        # equalize and resize
        face_img = cv2.equalizeHist(gray[y:y+h, x:x+w])
        face_img = cv2.resize(face_img, (200,200))
        id, confidence = recognizer.predict(face_img)

        if confidence < 100:
            color = colors[id]
            id = names[id]
        else:
            color = colors[0]
            id = 'unknown'
            
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        # confidence dari LBPH itu jarak (distance): 0 = cocok banget, makin besar makin jelek.
        # Kita balik jadi persentase, jarak 0 -> 100%, jarak >=100 -> 0%.
        match_percent = max(0, round(100 - confidence))
        confident = '{}%'.format(match_percent)

        cv2.putText(frame, str(id), (x+5, y-5), font, 1, color, 1)
        cv2.putText(frame, confident, (x+5, y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
