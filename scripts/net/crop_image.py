#! /usr/bin/env python
# -*- coding:utf-8 -*-

import cv2 as cv
import numpy as np


# 将图片进行剪裁
# 返回图片中主要物体所在的矩形框，左下角和右上角坐标
# 如果没有则返回None
def get_crop_img(img):
    #  获取宽高，用于将最后的限制在图像范围内
    h, w = img.shape[:2]
    # 转化为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel, iterations=4)

    # 边缘提取
    canny = cv.Canny(gray, 30, 150)
    # cv.imshow('canny ', canny)

    # (_, thresh) = cv.threshold(canny, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # 改用自适应边缘提取
    # thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                               cv.THRESH_BINARY_INV, 25, 10)
    # cv.imshow('thr', thresh)

    # 扩充,消除噪点和细线
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel, iterations=8)
    # cv.imshow('closed1', closed)

    # closed = cv.morphologyEx(canny, cv.MORPH_OPEN, kernel, iterations=1)
    # 膨胀操作,填充内部，拓展外界，将临近的轮廓连接起来
    closed = cv.dilate(closed, kernel, iterations=32)
    # cv.imshow('closed2', closed)
    # 闭操作，填充内部，并保存轮廓总体不变化太多
    closed = cv.morphologyEx(closed, cv.MORPH_CLOSE, kernel, iterations=16)
    # cv.imshow('closed3', closed)
    # 执行四次腐蚀操作,将轮廓外部进行修剪
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    closed = cv.erode(closed, kernel, iterations=16)
    # cv.imshow('closed', closed)

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
    # 没有轮廓直接返回 None
    if len(cnts) == 0:
        return None

    # 计算面积最大的轮廓
    c = sorted(cnts, key=cv.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv.minAreaRect(c)
    box = np.int0(cv.boxPoints(rect))
    # box: [[119 479]
    #  [119 232]
    #  [374 232]
    #  [374 479]]
    y1 = min(box[:, 0])
    y2 = max(box[:, 0])
    x1 = min(box[:, 1])
    x2 = max(box[:, 1])
    x1 = np.clip(x1, 0, h)
    x2 = np.clip(x2, 0, h)
    y1 = np.clip(y1, 0, w)
    y2 = np.clip(y2, 0, w)
    points = np.array([[x1, y1], [x2, y2]])
    return points
    # # print('box:', box)
    # # draw a bounding box arounded the detected barcode and display the image
    # # 因为这个函数有极强的破坏性，所有需要在img.copy()上画
    # draw_img = cv.drawContours(img.copy(), [box], -1, (0, 0, 255), 3)
    # cv.imshow("draw_img", draw_img)
    #
    # x1 = min(box[:, 0])
    # x2 = max(box[:, 0])
    # y1 = min(box[:, 1])
    # y2 = max(box[:, 1])
    #
    # height = y2 - y1
    # width = x2 - x1
    # if width <= 0 or height <= 0:
    #     return img
    # crop_img = img[y1:y1 + height, x1:x1 + width]
    # # cv.imshow('crop_img', crop_img)
    # return {
    #     'crop_img': crop_img,
    #     'draw_img': draw_img,
    # }
    # cv.waitKey(0)


def test(path):
    img = cv.imread(path)
    print(img.shape)
    p1, p2 = get_crop_img(img)
    print(p1, p2)
    crop = img[p1[0]:p2[0], p1[1]:p2[1]]
    cv.imshow('crop', crop)
    cv.waitKey(0)


if __name__ == '__main__':
    # test('./test_image/yagao.png')
    # test('./test_image/book.png')
    print('main')
    # test('/home/ace/Documents/flowers/imgs/0- (1).jpg')
