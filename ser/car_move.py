import serial 
import time 
import binascii 
import keyboard
import struct
import threading
import numpy as np

ser = serial.Serial("/dev/ttyTHS2",115200,timeout=.01)


def get_msg(x=0,y=0,d=0,h=0,f=0,p=0):
	# x y d h f p
	
	# d h x y f p
	
	msg = [d,h,x,y,f,p,0]
	msg = struct.pack("4h2?h",*msg)
	# print(binascii.b2a_hex(msg))
	# print(msg)

	return msg




class Car(threading.Thread):
	def __init__(self):
		super(Car,self).__init__()
		self.msg = get_msg()
		self.x = 0

		
	def run(self):
		while True:
			# self.msg = get_msg(x=self.x)
			ser.write(self.msg)
			# print('msg ',self.msg)
			data = ser.read_until(bytes(bytearray([0x11, 0xa0])))
			# print(data,binascii.b2a_hex(data))
			size = len(data)
			if size!=10:
				continue
			# print(data,len(data))
			# data = struct.unpack("16b",data)
			data = struct.unpack("%dh" % (size/2),data)
			# print(data)
			# print(binascii.b2a_hex(data))
			self.x = data[1]
			print('x ',self.x)
			time.sleep(.001)

	
	
	def move(self,st,ed):
		ps = np.linspace(st,ed,40,dtype=np.int32)
		for i in ps:
			print('i ',i)
			
			self.msg = get_msg(x=i)
			time.sleep(.01)
			if -25<=i-200<=25:
				break
			
					
car = Car()
car.start()
time.sleep(2)
print(car.x)
car.move(0,400)

# print(car.x)
# time.sleep(5)
# print(car.x)
# car.msg = get_msg(x=400)
# time.sleep(5)
# print(car.x)




