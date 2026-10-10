import cv2
import os

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(1)
data_path = "dataset/"

if not os.path.exists(data_path):
    os.mkdir(data_path)

person_id = 1 # kalo mau run, ini IDnya ganti dulu
count = 0 # kalo mau nambah, ini countnya jg diubah dulu ke penomoran terakhir

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(frame, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(30, 30))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        count+=1
        cv2.imwrite(data_path+"Person-"+str(person_id)
                    +"-"+str(count)+".jpg", gray[y:y+h, x:x+w])

    cv2.imshow("Camera",frame)

    if cv2.waitKey(1) == ord('q'):
        break
    elif count == 500: # Set di 50 agar dataset lebih banyak
        break

cap.release()
cv2.destroyAllWindows()