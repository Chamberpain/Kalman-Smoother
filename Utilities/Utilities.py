from geopy import Point
from geopy.distance import GreatCircleDistance
import math
from GeneralUtilities.Compute.constants import r_earth
import numpy as np

def parse_filename(filename):
	filename = filename.split('.npy')[0]
	toa_noise,process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise = [float(x.replace('-','.')) for x in filename.split('_')]
	return (toa_noise,process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise)

def make_filename(toa_noise_multiplier,process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise):
	tuplelist = []
	for dummy in toa_noise_multiplier,process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise:
		tuplelist.append(str(dummy).replace('.','-'))
	return '%s_%s_%s_%s_%s_%s'%tuple(tuplelist)

class KalmanPoint(Point):

	def __init__(self,*args,**kwargs):
		assert self.latitude< -35
		assert self.longitude<40
		assert self.longitude>-140
		assert self.latitude>-75

	def add_displacement(self,dx,dy):
		lat_const = 180 / math.pi
		lon_const = lat_const / math.cos(self.latitude * math.pi / 180)
		new_longitude = self.longitude + (dx / r_earth) * lon_const
		new_latitude = self.latitude + (dy / r_earth) * lat_const
		return self.__class__(new_latitude,new_longitude)

def dx_dy_distance(point1,point2):
	assert isinstance(point1,KalmanPoint)
	assert isinstance(point2,KalmanPoint)
	lat1 = point1.latitude
	lon1 = point1.longitude

	lat2 = point2.latitude
	lon2 = point2.longitude

	dy = np.sign(lat2-lat1)*GreatCircleDistance(KalmanPoint(lat1,lon1),KalmanPoint(lat2,lon1)).km

	dx1 = np.sign(lon2-lon1)*GreatCircleDistance(KalmanPoint(lat1,lon1),KalmanPoint(lat1,lon2)).km
	dx2 = np.sign(lon2-lon1)*GreatCircleDistance(KalmanPoint(lat2,lon1),KalmanPoint(lat2,lon2)).km
	dx = (dx1+dx2)/2
	return (dy,dx)
