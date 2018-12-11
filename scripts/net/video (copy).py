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
	print(flag,img)
        cv.imshow("camera", img)
        if not flag:
            break
        try:
            print(img.shape)
	    ps = get_crop_img(img)
            if not ps:
		continue
            p1, p2 = ps
            print(p1, p2)
            crop = img[p1[0]:p2[0], p1[1]:p2[1]]
            cv.imshow('crop', crop)
	    crop = cv.resize(crop, (224,224))
	    predict(crop)
        except Exception as e:
            # print(e)
            pass

        # cv.waitKey(5)


if __name__ == '__main__':
    try:
        # start(1)
	pass
    except Exception as e:
        # print(e)
        pass
