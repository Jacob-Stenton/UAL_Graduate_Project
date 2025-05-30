from smbus2 import SMBus

addr = 0x8
bus = SMBus(1)

while True:
    byte = bus.read_byte_data()
    print(byte)