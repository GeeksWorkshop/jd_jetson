#! /usr/bin/env python
# coding:utf-8

import cv2 as cv
import numpy as np
import serial 
import time 
import binascii 
import keyboard
import threading

def get_msg(ch0=1024,ch1=1024,ch2=1021,ch3=1024,s1=1,s2=2):
	# stop
	# ch0 = 1024
	# ch1 = 1024
	# ch2 = 1021
	# ch3 = 1024

	# x
	# ch3 = 0x400
	# y
	# ch2 = 0x400

	# h
	# ch1 = 0x400

	# s1 = 1
	# s2 = 2

	pdata0 = ch0 & 0xff
	pdata1 = ((ch0 >> 8 ) & 0x07) + (ch1<<3) &  0xff
	pdata2 = ((ch1>>5) & 0x3f)+((ch2<<6) & 0xff)
	pdata3 = (ch2 >> 2) & 0xff
	pdata4 = ((ch2 >> 10) & 0x01) + ((ch3 <<1 ) & 0xfe)
	pdata5 = ((ch3 >> 7) & 0x0f) + ((s2<<4) & 0x30) + ((s1 << 6) & 0xc0)
	msg = [pdata0,pdata1,pdata2,pdata3,pdata4,pdata5]+[0]*12

	return msg



	
class Car(threading.Thread):

    def __init__(self, *args, **kwargs):
        super(Car, self).__init__(*args, **kwargs)
        self.__flag = threading.Event()     # 用于暂停线程的标识
        self.__flag.set()       # 设置为True
        self.__running = threading.Event()      # 用于停止线程的标识
        self.__running.set()      # 将running设置为True
        self.__ser = serial.Serial("/dev/ttyTHS2",100000,timeout=10)
        self.__msg = get_msg()
		
    def run(self):
        while self.__running.isSet():
            self.__flag.wait()      # 为True时立即返回, 为False时阻塞直到内部的标识位为True后返回
            # print('msg ',self.__msg)
            self.__ser.write(self.__msg)
            time.sleep(.01)

    def pause(self):
        self.__flag.clear()     # 设置为False, 让线程阻塞

    def resume(self):
        self.__flag.set()    # 设置为True, 让线程停止阻塞

    def stop(self):
        self.__flag.set()       # 将线程从暂停状态恢复, 如何已经暂停的话
        self.__running.clear()        # 设置为False
    def send(self,msg):
		self.__msg = msg
	
    def move(self, s):
		if s=='w':
			print('w key')
			msg = get_msg(ch3=0x500)
		elif s == 's':
			print('s key')
			msg = get_msg(ch3=0x300)
		elif s == 'a':
			print('a key')
			msg = get_msg(ch2=0x300)
		elif s == 'd':
			print('d key')
			msg = get_msg(ch2=0x500)
		elif s == 'y':
			print('y key')
			msg = get_msg(ch1=0x300)
		elif s == 'h':
			print('h key')
			msg = get_msg(ch1=0x500)
		else:
			print('stop')
			msg = get_msg(ch1=0x400)
		
		self.send(msg)
		
			
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
    cv.imshow('canny ', canny)

    # (_, thresh) = cv.threshold(canny, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # 改用自适应边缘提取
    # thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                               cv.THRESH_BINARY_INV, 25, 10)
    # cv.imshow('thr', thresh)

    # 扩充,消除噪点和细线
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel, iterations=8)
    cv.imshow('closed1', closed)

    # closed = cv.morphologyEx(canny, cv.MORPH_OPEN, kernel, iterations=1)
    # 膨胀操作,填充内部，拓展外界，将临近的轮廓连接起来
    closed = cv.dilate(closed, kernel, iterations=32)
    cv.imshow('closed2', closed)
    # 闭操作，填充内部，并保存轮廓总体不变化太多
    closed = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel, iterations=16)
    cv.imshow('closed3', closed)
    # 执行四次腐蚀操作,将轮廓外部进行修剪
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    closed = cv.erode(closed, kernel, iterations=16)
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
    # print(cnts)import threading

    # 没有轮廓直接返回 None
    if len(cnts) == 0:
        return None

    # 计算面积最大的轮廓
    c = sorted(cnts, key=cv.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
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
    points = [[x1, y1], [x2, y2]]
    return points


def t(path):
    img = cv.imread(path)
    # print(img.shape)
    p1, p2 = get_crop_img(img)
    # print(p1, p2)
    crop = img[p1[0]:p2[0], p1[1]:p2[1]]
    cv.imshow('crop', crop)
    xc = (p1[1] + p2[1]) // 2
    yc = (p1[0] + p2[0]) // 2
    cv.circle(img, (xc, yc), 3, (0, 0, 255), 3)
    cv.circle(img, (p1[1], p1[0]), 3, (0, 0, 255), 3)
    cv.circle(img, (p2[1], p2[0]), 3, (0, 0, 255), 3)
    cv.rectangle(img, (p1[1], p1[0]), (p2[1], p2[0]), (0, 255, 0), 3)

    img_xc = img.shape[1] // 2
    img_yc = img.shape[0] // 2

    cv.circle(img, (img_xc, img_yc), 3, (0, 255, 255), 3)
    dx = xc - img_xc
    dy = yc - img_yc
    # 左负右正
    # print('dx ', dx, 'dy ', dy)
    h,w = img.shape[:2]
    min_loss = 50

    cv.line(img, (img_xc+min_loss,0),(img_xc+min_loss,h),(0,0,0),1)
    cv.line(img, (img_xc-min_loss,0),(img_xc-min_loss,h),(0,0,0),1)

    cv.imshow('img', img)
    cv.waitKey(0)


def start():
	car = Car()
	car.start()
	cap = cv.VideoCapture(1)
	while True:
		ret,img = cap.read()
		# print(img.shape)
		try:
			p1, p2 = get_crop_img(img)
			# print(p1, p2)

		except Exception as e:
			print(e)
			continue
			
		crop = img[p1[0]:p2[0], p1[1]:p2[1]]
		cv.imshow('crop', crop)
		xc = (p1[1] + p2[1]) // 2
		yc = (p1[0] + p2[0]) // 2
		cv.circle(img, (xc, yc), 3, (0, 0, 255), 3)
		cv.circle(img, (p1[1], p1[0]), 3, (0, 0, 255), 3)
		cv.circle(img, (p2[1], p2[0]), 3, (0, 0, 255), 3)
		cv.rectangle(img, (p1[1], p1[0]), (p2[1], p2[0]), (0, 255, 0), 3)

		img_xc = img.shape[1] // 2
		img_yc = img.shape[0] // 2

		cv.circle(img, (img_xc, img_yc), 3, (0, 255, 255), 3)
		dx = xc - img_xc
		dy = yc - img_yc
		# 左负右正
		# print('dx ', dx, 'dy ', dy)
		h,w = img.shape[:2]
		min_loss = 50
		cv.line(img, (img_xc+min_loss,0),(img_xc+min_loss,h),(0,0,0),1)
		cv.line(img, (img_xc-min_loss,0),(img_xc-min_loss,h),(0,0,0),1)

		cv.imshow('img', img)
		
		if abs(dx)<= min_loss*2:
			car.move(' ')
		elif dx<0:
			car.move('a')
		else:
			car.move('d')
			
		cv.waitKey(5)
    
		
	cap.release()
	cv.destroyAllWindows()

if __name__ == '__main__':
	start()
    # t('./test_image/yagao.png')
    # t('./test_image/book.png')
    # print('main')
    # t('/home/nvidia/Documents/jdx/src/demo/scripts/net/imgs/1- (4).jpg')
    # t('./imgs/cat3.jpg')
    # test('./video/test.mp4')
