import serial 
import time 
import binascii 
import keyboard
import struct
import numpy as np
import serial 
import time 
import binascii 
import keyboard
import struct
import threading

ser = serial.Serial("/dev/ttyTHS2",115200,timeout=10)

def get_msg(x=0,y=0,d=0,h=0,f=0,p=0):
	# x y d h f p
	
	# d h x y f p
	
	msg = [d,h,x,y,f,p,0]
	msg = struct.pack("4h2?h",*msg)
	print(binascii.b2a_hex(msg))
	# print(msg)

	return msg

class Car(threading.Thread):
	def __init__(self):
		super(Car,self).__init__()
		self.msg = get_msg()
		self.x = 0
		self.y = 0
		
	def run(self):
		while True:
			# print('msg ',self.msg)
			ser.write(self.msg)
			time.sleep(.01)
			
	
	
	def move(self,st,ed):
		ps = np.linspace(st,ed,20,dtype=np.int32)
		time.sleep(1)
		for i in ps:
			print(i)
			self.msg = get_msg(x=i)
			time.sleep(.8)
			# if i>300 :
				# break
 


		

car = Car()
car.start()
car.move(0,400)

