import serial 
import time 
import binascii 

ser1 = serial.Serial("/dev/ttyTHS2",100000,timeout=10)

while True:
	# ser1.write(msg)
	# time.sleep(.1)
	data = ser1.read(18)
	print(binascii.b2a_hex(data))

# time.sleep(.1)

