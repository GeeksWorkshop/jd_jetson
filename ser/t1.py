import serial 
import time 
import binascii 

ser1 = serial.Serial("/dev/ttyTHS2",100000,timeout=1.5)
ser2 = serial.Serial("/dev/ttyTHS3",9600,timeout=1.5)

while True:
	# a = 'a'
	# b = binascii.b2a_hex(a)
	# print(b)

	# ser1.write(b)
	
	for i in range(5,30):
		for j in range(5):
			data = ser1.read(i)
			# print(i,data,type(data))
			print(i,binascii.b2a_hex(data))
	
	break
