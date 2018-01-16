# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 19:30:56 2017

@author: NI
"""

import os
import cv2
from hyperlpr_py3 import  pipline as  pp
import time
FindPath = 'F:\\NI\\Materials\\license\\night'
FileNames = os.listdir(FindPath)
count=0
for file_name in FileNames:
    t0 = time.time()
    fullfilename=os.path.join(FindPath,file_name)  
    #print file_name
    image = cv2.imread(fullfilename)
    sp = image.shape
    sz1 = sp[0]#height(rows) of image
    sz2 = sp[1]#width(colums) of image
    #print(sz1,sz2)
    image,res  = pp.SimpleRecognizePlateByE2E(image)
    out=image
    if(sz2>1280):
        out=cv2.resize(image,(int(sz2*3.0/4.0),int(sz1*3.0/4.0)),interpolation=cv2.INTER_AREA)
    count = count + 1
    print('current id:%d'%count)
    print(time.time() - t0,"s")
    cv2.imshow(file_name,out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print ('total image is:%d'%count)