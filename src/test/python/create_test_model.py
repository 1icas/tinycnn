import struct
from struct import pack

with open('model.p', 'wb+') as f:
  f.write(pack('i', 2)) #layer count
  f.write(pack('2i', 0, 60)) #every layer index
  f.write(pack('5i', 4, 1, 2, 2, 1)) #first layer 
  f.write(pack('2i', 4, 0)) #second layer