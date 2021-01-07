#read wajah
import cv2, os
import numpy as np
from PIL import Image
wajahDir = 'datawajah'
readDir = 'readwajah'
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getImageLabel(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    faceIDs = []
    for imagePath in imagePaths :
        PILImg = Image.open(imagePath).convert('L') #convert ke dalam grey
        imgNum = np.array(PILImg, 'uint8')
        faceId = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = faceDetector.detectMultiScale(imgNum)
        for (x, y, w, h) in faces :
            faceSamples.append(imgNum[y:y+h, x:x+w])
            faceIDs.append(faceId)
        return faceSamples, faceIDs
        
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
print("Mesin sedang melakukan training data wajah")
faces, IDs = getImageLabel(wajahDir)
print(IDs)
faceRecognizer.train(faces, np.array(IDs))
#simpan 
faceRecognizer.write(readDir+'/training.xml')
print('Sebanyak {0} data wajah telah ditraining ke mesin.', format(len(np.unique(IDs))))