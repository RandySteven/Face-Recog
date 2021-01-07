import cv2, os, numpy as np
wajahDir = 'datawajah'
readDir = 'readwajah'
cam = cv2.VideoCapture(0)
cam.set(3, 640) #ubah lebar cam
cam.set(4, 480) #ubah tinggi cam
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

faceRecognizer.read(readDir+'/training.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['Toilet', 'Creator', 'Siapa tu??']

minWidth = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)

while True:
    retV, frame = cam.read()
    frame = cv2.flip(frame, 1) #vertical flip
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.2, 5, minSize=(round(minWidth), round(minHeight)),) #frame, #factor scale, #minneighbor

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, confidance = faceRecognizer.predict(abuAbu[y:y+h, x:x+w]) #confidance = 0 artinya cocok
        if confidance <= 50:
            nameId = names[1]
            confidanceTxt = "{0}%".format(round(100-confidance))
        elif confidance >= 100:
            nameId = names[2]
            confidanceTxt = "{0}%".format(round(100-confidance))
        else :
            nameId = names[0]
            confidanceTxt = "{0}%".format(round(100-confidance))
        cv2.putText(frame, str(nameId), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(confidanceTxt), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

    cv2.imshow("Recognize Wajah", frame)  
    
    k = cv2.waitKey(1) & 0xFF 
    if k == 27 or k == ord('q'):
        break

print("Pengambilan wajah selesai")
cam.release()
cv2.destroyAllWindows()

