import serial 
import time 
import binascii 

ser1 = serial.Serial("/dev/ttyTHS2",100000,timeout=10)

# stop
ch0 = 1024
ch1 = 1024
ch2 = 1021
ch3 = 1024

# x
ch3 = 0x400
# y
# ch2 = 0x400

# h
# ch1 = 0x400

s1 = 1
s2 = 2

pdata0 = ch0 & 0xff
pdata1 = ((ch0 >> 8 ) & 0x07) + (ch1<<3) &  0xff
pdata2 = ((ch1>>5) & 0x3f)+((ch2<<6) & 0xff)
pdata3 = (ch2 >> 2) & 0xff
pdata4 = ((ch2 >> 10) & 0x01) + ((ch3 <<1 ) & 0xfe)
pdata5 = ((ch3 >> 7) & 0x0f) + ((s2<<4) & 0x30) + ((s1 << 6) & 0xc0)

msg = [pdata0,pdata1,pdata2,pdata3,pdata4,pdata5]+[0]*12
while True:

	print('write ', msg)
	for i in range(1000):
		ser1.write(msg)
		time.sleep(.01)

	


	break
