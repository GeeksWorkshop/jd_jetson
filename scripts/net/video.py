#! /usr/bin/env python
# coding:utf-8

import cv2 as cv
import numpy as np
from predict import predict
from crop_image import get_crop_img


def start(video_path):
    # 开启ip摄像头
    capture = cv.VideoCapture(video_path)

    print('准备完成，开启录像')
    while True:
        flag, img = capture.read()

        # cv.imshow("camera", img)
        if not flag:
            break
        try:
            # print(img.shape)
	    ps = get_crop_img(img)
	    print(ps)

            # if ps == None:
		# continue	
            p1, p2 = ps
            print(p1, p2)
	    w,h = p2[0]-p1[0],p2[1]-p1[1]
            crop = img[p1[0]:p2[0], p1[1]:p2[1]]
	    cv.rectangle(img,(p1[1],p1[0]),(p2[1],p2[0]),(0,0,255),3)
            cv.imshow('img', img)

	    
            cv.imshow('crop', crop)
	    crop = cv.resize(crop, (224,224))
	    crop = np.reshape(crop,(1,224,224,3))
	    ret = predict(crop)
	    print(ret)
        except Exception as e:
            print(e)
            pass

        cv.waitKey(1)


if __name__ == '__main__':
    try:
        # start(1)
	pass
    except Exception as e:
        # print(e)
        pass
