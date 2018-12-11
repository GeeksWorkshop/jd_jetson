import serial 
import time 
import binascii 
import struct

ser1 = serial.Serial("/dev/ttyTHS2",115200,timeout=.05)

while True:
	# ser1.write(msg)
	# time.sleep(.1)
	st = time.time()
	data = ser1.read_until(bytes(bytearray([0x11, 0xa0])))
	# data = ser1.read(10)
	ed = time.time()
	print('t ',ed-st)
	print(data,binascii.b2a_hex(data))
	size = len(data)
	print(size)
	if size!=10:
		continue
	# print(data,len(data))
	# data = struct.unpack("16b",data)
	data = struct.unpack("%dh" % (size/2),data)
	print(data)
	# print(binascii.b2a_hex(data))

# time.sleep(.1)

