# -*- coding: utf-8 -*-
import os
import cv2
from hyperlpr_py3 import  pipline as  pp
import time
import numpy as np

def cv_imread(img_path):
    path =np.fromfile(img_path,dtype=np.uint8)
    #print(u'vgygy'+path)
    cv_img=cv2.imdecode(path,-1)
    return cv_img

if __name__ == '__main__':

    FindPath = 'F://NI//Materials//license//“青”牌-36'

    FileNames = os.listdir(FindPath)
    count=0
    success=0

    for file_name in FileNames:

        t0 = time.time()
        fullfilename=os.path.join(FindPath,file_name)
        #print(file_name)
        image= cv_imread(fullfilename)
        sp = image.shape

        sz1 = sp[0]#height(rows) of image
        sz2 = sp[1]#width(colums) of image
        if(sz2>=1024):
            image=cv2.resize(image,(int(sz2*2.0/4.0),int(sz1*2.0/4.0)),interpolation=cv2.INTER_AREA)
        #print(sz1,sz2)
        result =[[0 for col in range(2)] for row in range(5)]
        #image,res=pp.SimpleRecognizePlateByE2E(image)
        image,res,flag,result  = pp.SimpleRecognizePlate(image,file_name)
        print(result)
        if flag==1:
            success=success+1
        count = count + 1

        print(time.time() - t0,"s")
        cv2.imshow(str(count)+'.jpg',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print('result:%.1f'%(success/count*100))
    print('success:%d'%success)
    print ('total image is:%d'%count)
    '''
    cap = cv2.VideoCapture('video//bandicam0.mp4')
    while(1):
       t0 = time.time()
       # get a frame
       ret, frame = cap.read()

       # show a frame
       #cv2.imshow("capture", frame)
       if cv2.waitKey(100) & 0xFF == ord('q'):
          break
       sp = frame.shape
       if sp[0]==0|sp[1]==0|sp[2]==0:
           break
       sz1 = sp[0]#height(rows) of image
       sz2 = sp[1]#width(colums) of image
       if(sz2>1280):
          image=cv2.resize(image,(int(sz2*2.0/4.0),int(sz1*2.0/4.0)),interpolation=cv2.INTER_AREA)
        #print(sz1,sz2)
       result =[[0 for col in range(2)] for row in range(5)]
       #image,res=pp.SimpleRecognizePlateByE2E(image)
       image,res,flag,result  = pp.SimpleRecognizePlate(frame,'capture')
       print(result)
       print(time.time() - t0,"s")
       cv2.imshow('current Result',image)
       cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
   '''
