#! /usr/bin/env python
# coding:utf-8

import cv2 as cv
import numpy as np



# cap = cv2.VideoCapture(u'http://192.168.43.1:8081')

cap = cv.VideoCapture(1)
fourcc = cv.VideoWriter_fourcc(*'mp4v')
outfile = cv.VideoWriter('out.mp4',fourcc,25,(640,480))

while True:
    ret,frame = cap.read()
    print(ret,frame.shape)
    if ret:
        cv.imshow("frame", frame)
        outfile.write(frame)
        cv.waitKey(10)
    else:
        break
cap.release()

cv2.destroyAllWindows()

