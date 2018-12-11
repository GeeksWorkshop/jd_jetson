import serial 
import time 
import binascii 
import keyboard
import struct

ser1 = serial.Serial("/dev/ttyTHS2",115200,timeout=10)


def get_msg(x=0,y=0,d=0,h=0,f=0,p=0):
	# x y d h f p
	
	# d h x y f p
	
	msg = [d,h,x,y,f,p,0]
	msg = struct.pack("4h2?h",*msg)
	print(binascii.b2a_hex(msg))
	# print(msg)

	return msg



		

while True:
	msg = get_msg(x=00)
	ser1.write(msg)
	time.sleep(.001)

