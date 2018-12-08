import serial 
import time 
import binascii 

ser1 = serial.Serial("/dev/ttyTHS2",100000,timeout=10)

msg = '04205b2ea8'
# ser1.write(b)
print(msg)
# msg = [0]+[0x1]*16
msg=[0x0,0x0,0x0,0x05,0x20,0x5b,0x2e,0xa8,0x0,0x0,0x0]
msg = '0'+'1'*17
msg = [0]*17+[1]
msg = '0'*17+'1'

msg=[0x0,0x05,0x20,0x5b,0x2e,0xa8,0x0] + [0x0]*11

msg = [0,0x04,0xa0,0xff,0x00,0xe8] + [0]*12


msg = [0,0x42,0x05,0xb3,0x2a,0x80] + [0]*12

print(msg)
while True:
	time.sleep(.01)
	data = ser1.read(18)
	data = binascii.b2a_hex(data)
	print('read ',data, type(data))
	data = [
		int(data[2*i:2*i+2],16)
		for i in range(18)
	]
	print('write ', data)
	ser1.write(data)

# time.sleep(.1)

