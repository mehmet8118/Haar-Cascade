import numpy as np
from PIL import ImageGrab
import cv2
import time



def resim(imgs):

    img = imgs # screen_record'dan gelen resmimizi img değişkenine atadık
    yuz_casc = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    # Haar Cascade xml dosyası.
    # türkçesi: basamaklı sınıflandırıcı
    griton=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Resmimizi gri ton yapıyoruz
    yuzler=yuz_casc.detectMultiScale(griton,1.1,4)
    """:arg
    minNeighbors tespit ettiği nesnenin minimum kaç farklı çerçevede aranması gerektiğini belirttiğimiz 
    parametre .minsize ve maxsize da çerçevenin boyutu﻿ yani bu çerçeve resimi tararken farklı boyutlarda 
    tarıyor örneğin ilk tarama (2,2) sonra (3,3) eğer minNeighbors=2 ise bu iki çerçevede de nesne bulmuşsa 
    işaretliyor, beş olsaydı çerçevelerin boyutunu değiştirecekti beş defa ve beşinde de bulursa nesneyi öyle
     işaret koyacaktı.
    """
    for(x,y,w,h) in yuzler: # Burda resimde bulduğu .... kordinatlarını alıyor
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3) # Aldığı kordinatlar ile çerçeve çiziyoruz

    cv2.imshow('yuzler',img) #Resmimizi gösterdik


def screen_record():
    last_time = time.time()
    while(True):
        printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640))) #Anlık olarak ekran görüntüsünü aldık
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        resim(printscreen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
screen_record()