import numpy as np
import cv2
from datetime import datetime
import face_recognition as fr
import cv2 as cv
import os
path = "C:/image/"
images = []
names = []
myList = os.listdir(path)
print(myList)

for imgnames in myList:
    curImg = cv2.imread(f"{path}/{imgnames}")
    images.append(curImg)
    names.append(os.path.splitext(imgnames)[0])
print(names)


def findencodings(images):
    encodedlist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodedlist.append(encode)
    return encodedlist


def markattendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S,%m/%d/%Y')
            f.writelines(f'\n{name},{dtString}')


encodedlistknown = findencodings(images)
print('encodings complete')
print(len(encodedlistknown))
cap = cv2.VideoCapture(0)

while True:
    success, webcam = cap.read()
    imgResized = cv2.resize(webcam, (0, 0), None, 0.25, 0.25)
    imgResized = cv2.cvtColor(imgResized, cv2.COLOR_BGR2RGB)

    faceCurFrame = fr.face_locations(imgResized)
    encodeFaceCurFrame = fr.face_encodings(imgResized, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeFaceCurFrame, faceCurFrame):
        matches = fr.compare_faces(encodedlistknown, encodeFace)
        faceDis = fr.face_distance(encodedlistknown, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = names[matchIndex].upper()
            #print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(webcam, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(webcam, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(webcam, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, )
            markattendance(name)
    cv2.imshow("webcam", webcam)
    cv2.waitKey(1)


