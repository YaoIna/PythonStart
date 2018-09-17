import numpy

lat_a_degree, lon_a_degree = map(float, input("input point A latitude and longitude:").split())
lat_b_degree, lon_b_degree = map(float, input("input point B latitude and longitude:").split())

radian_c = numpy.pi / 180

r = 111 * 360 / (2 * numpy.pi)

lat_a, lon_a = lat_a_degree * radian_c, lon_a_degree * radian_c
lat_b, lon_b = lat_b_degree * radian_c, lon_b_degree * radian_c

distance = r * numpy.arccos(
    numpy.sin(lat_a) * numpy.sin(lat_b) + numpy.cos(lat_a) * numpy.cos(lat_b) * numpy.cos(lon_a - lon_b))

print('Distance between a and b is ', distance, 'cos(lon) is ',
      numpy.sin(lat_a) * numpy.sin(lat_b) + numpy.cos(lat_a) * numpy.cos(lat_b) * numpy.cos(lon_a - lon_b), ' r = ', r,
      ' great distance is ', 2 * numpy.pi * r - distance)

'''git test'''
