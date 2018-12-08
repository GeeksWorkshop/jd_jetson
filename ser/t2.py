import serial 
import time 
import binascii 

ser1 = serial.Serial("/dev/ttyTHS2",99999,timeout=1.5)
ser2 = serial.Serial("/dev/ttyTHS3",9600,timeout=1.5)

while True:
	# a = 'a'
	# b = binascii.b2a_hex(a)
	# print(b)

	# ser1.write(b)

	data = ser1.read(36)
	# print(type(data))
	# data = ser1.readline()
	# print(i,data,type(data))
	print(binascii.b2a_hex(data))

	# time.sleep(.1)

