import serial 
import time 
import binascii 

ser1 = serial.Serial("/dev/ttyTHS2",100000,timeout=1.5)

msg = '04205b2ea8'
# ser1.write(b)
print(msg)
msg=[0x0,0x05,0x20,0x5b,0x2e,0xa8,0x0] + [0x0]*11
# msg = [0]+[0x1]*16
msg=[0x0,0x0,0x0,0x05,0x20,0x5b,0x2e,0xa8,0x0,0x0,0x0]
msg = '0'+'1'*17
print(msg)
while True:
	ser1.write(msg)
# print(type(data))
# data = ser1.readline()
# print(i,data,type(data))
# print(binascii.b2a_hex(data))

# time.sleep(.1)

