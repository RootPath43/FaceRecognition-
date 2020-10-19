import os
from datetime import time, datetime, timedelta
from firebase import firebase
import cv2
import face_recognition
import numpy as np
import _thread
import time

path='Images'#fotoğrafların konumu
images = []
classNames = []
timeList=[]
alreadyExecuted=True
myList=os.listdir(path)
print(myList)
firebase = firebase.FirebaseApplication("https://raspberrypie-67356.firebaseio.com/", None)#firebase bağlantı linki
for cl in myList:
    currentImage=cv2.imread(f'{path}/{cl}')
    images.append(currentImage)
    classNames.append(os.path.splitext(cl)[0])#fotoğrafların isimlerini alma
print(classNames)
def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#renkleri
        encodeEmirhan = face_recognition.face_encodings(img)[0]#encoding
        encodeList.append(encodeEmirhan)#encoded fotoğrafları list e aktarma
    return encodeList
def markEntarence(name ,timeString):
    nameList =[]
    time.sleep(10)#programın beklemesi için
    enteringTime = timeList[0]#giriş saati kişiyi ilk gördüğü andaki saat
    exittingTime = timeList.pop()#programın kişiyi en son gördüğü zman
    enter = datetime.strptime(enteringTime, '%H:%M:%S')#giriş saatini stringten datetime a çevirme
    exit = datetime.strptime(exittingTime, '%H:%M:%S')#çıkış saatini stringren datetime a çevirme
    duration=exit-enter#içeride bulunduğu süreyi hesaplama
    Duration =str( duration)#içerde bulunduğu süreyi string e çevirme
    if name not in nameList:
            data={'Name': name,'Entering Time': enteringTime,'Exitting Time': exittingTime  ,'Duration': Duration}
            #database için kaydedilecek veriler
    if alreadyExecuted:
                result = firebase.post('/raspberrypie-67356/Usage', data)#database e  veriyi kaydetme methodu
                print(result)
                return alreadyExecuted ==False


encodeListKnown=findEncodings(images)
print('Encoding Complete')
cap=cv2.VideoCapture(0)#kameraya erişim
runBefore=True
while True:
    success, img=cap.read()#görüntüyü okuma
    imgSmall=cv2.resize(img,(0,0),None,0.25,0.25)#alının görüntüyü küçültme
    imgSmall=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#renklerin dönüştürülmesi
    faceCurrentFrame = face_recognition.face_locations(imgSmall)#yüzün lokasyonuu belirleme
    encodeCurrentFrame = face_recognition.face_encodings(imgSmall, faceCurrentFrame)#fotoğrafın kod haline dönüştürülmesi

    for encodeFace, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
        matches= face_recognition.compare_faces(encodeListKnown,encodeFace)# yüz eşleştirme
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)#yüz benzeme oranı
        print(faceDistance)
        matchIndex=np.argmin(faceDistance)
        if matches[matchIndex]:
            name=classNames[matchIndex].upper()

            now=datetime.now()#yüz tanımlandığı zmaanı alma
            timeString=now.strftime('%H:%M:%S')#zamanı stringe çevirme
            timeList.append(timeString)#lis'e ekleöe

            print(name,timeString )
            y1,x2,y2,x1 =faceLoc
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)#yüzü çerçeveleme
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2,)#çerçeve altına metin yazılması
            if runBefore:

                _thread.start_new_thread(markEntarence, (name, timeString))#giriş çıkış zamanınını database e aktaran methodun çalıştıırlması
                runBefore=False




    cv2.imshow('Webcam', img)#görüntüleme
    cv2.waitKey(1)#bekleme





