#! /usr/bin/env python
# coding:utf-8


import cv2


# cap = cv2.VideoCapture(u'http://192.168.43.1:8081')

cap = cv2.VideoCapture(1)

while True:
    ret,img = cap.read()
    print(ret,img.shape)
    if not ret:
		break
		    
    cell = 10
    h,w = img.shape[:2]
    cv2.rectangle(img,(w//cell*(cell-1),h//cell*(cell-1)),(w//cell,h//cell),(0,0,255),2)
    cv2.imshow('img',img)
    crop = img[h//cell:h//cell*(cell-1),w//cell:w//cell*(cell-1)]
    
    
    cv2.waitKey(10)
    
  
cap.release()

cv2.destroyAllWindows()

