#! /usr/bin/env python
# coding:utf-8

import serial 
import time 
import binascii 
import keyboard

import threading

#!/usr/bin/env python
# coding: utf-8

import threading
import time

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


def move(s):
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
		msg = get_msg(ch1=0x500)
	
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
            print('msg ',self.__msg)
            self.__ser.write(self.__msg)
            time.sleep(.1)

    def pause(self):
        self.__flag.clear()     # 设置为False, 让线程阻塞

    def resume(self):
        self.__flag.set()    # 设置为True, 让线程停止阻塞

    def stop(self):
        self.__flag.set()       # 将线程从暂停状态恢复, 如何已经暂停的话
        self.__running.clear()        # 设置为False
    def send(self,msg):
		self.__msg = msg
		
		

car = Car()

car.start()
time.sleep(1)
print('aaa')

car.send('a')
time.sleep(1)
print('aaa')


car.send('b')
time.sleep(1)
print('bbb')
