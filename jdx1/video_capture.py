# coding=utf-8
import cv2 as cv
import numpy as np
from mobile_net import predict


def get_crop_img(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # blurred = cv.GaussianBlur(gray, (9, 9), 0)
    # cv.imshow('blur', blurred)
    # canny = cv.Canny(blurred, 50, 150)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel, iterations=4)

    canny = cv.Canny(gray, 30, 150)
    cv.imshow('canny', canny)

    # (_, thresh) = cv.threshold(canny, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # 改用自适应边缘提取
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv.THRESH_BINARY_INV, 25, 10)

    cv.imshow('thr', thresh)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel, iterations=1)

    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # 执行四次腐蚀操作
    # closed = cv.erode(closed, kernel, iterations=2)
    # # 膨胀操作
    # closed = cv.dilate(closed, kernel, iterations=2)
    #
    cv.imshow('closed', closed)

    (_, cnts, _) = cv.findContours(
        # 参数一： 二值化图像
        # closed.copy(),
        closed.copy(),
        # 参数二：轮廓类型
        cv.RETR_EXTERNAL,  # 表示只检测外轮廓
        # cv2.RETR_CCOMP,                #建立两个等级的轮廓,上一层是边界
        # cv2.RETR_LIST,                 #检测的轮廓不建立等级关系
        # cv.RETR_TREE,                 #建立一个等级树结构的轮廓
        # cv.CHAIN_APPROX_NONE,  # 存储所有的轮廓点，相邻的两个点的像素位置差不超过1
        # 参数三：处理近似方法
        cv.CHAIN_APPROX_SIMPLE,  # 例如一个矩形轮廓只需4个点来保存轮廓信息
        # cv2.CHAIN_APPROX_TC89_L1,
        # cv2.CHAIN_APPROX_TC89_KCOS
    )
    # print(cnts)
    c = sorted(cnts, key=cv.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv.minAreaRect(c)
    box = np.int0(cv.boxPoints(rect))
    # print('box:', box)
    # draw a bounding box arounded the detected barcode and display the image
    # 因为这个函数有极强的破坏性，所有需要在img.copy()上画
    draw_img = cv.drawContours(img.copy(), [box], -1, (0, 0, 255), 3)
    cv.imshow("draw_img", draw_img)

    x1 = min(box[:, 0])
    x2 = max(box[:, 0])
    y1 = min(box[:, 1])
    y2 = max(box[:, 1])

    height = y2 - y1
    width = x2 - x1
    if width <= 0 or height <= 0:
        return img
    crop_img = img[y1:y1 + height, x1:x1 + width]
    # cv.imshow('crop_img', crop_img)
    return {
        'crop_img': crop_img,
        'draw_img': draw_img,
    }
    # cv.waitKey(0)


def start(video_path):
    # 开启ip摄像头
    capture = cv.VideoCapture(video_path)
    flag, img = capture.read()
    print(flag, img)

    predict(cv.resize(img, (224, 224)).reshape((-1, 224, 224, 3)).astype(np.float32))
    print('准备完成，开启录像')
    while flag:
        cv.imshow("camera", img)
        flag, img = capture.read()
        if not flag:
            break
        try:
            imgs = get_crop_img(img)
            cv.imshow('crop_img', imgs['crop_img'])
            print('crop_img', imgs['crop_img'].shape)
            net_img = cv.resize(imgs['crop_img'], (224, 224))
            cv.imshow('net_img', net_img)
            net_img = np.expand_dims(net_img, axis=0).astype(np.float32)  # (1, 224, 224, 3)
            print('net_img ', net_img.shape)
            predict_val, logits_val = predict(net_img)
            print('predict_val ', predict_val)
        except Exception as e:
            # print(e)
            pass

        cv.waitKey(5)


if __name__ == '__main__':
    try:
        start(1)
    except Exception as e:
        # print(e)
        pass
