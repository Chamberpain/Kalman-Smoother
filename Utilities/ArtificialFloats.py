from __future__ import print_function
import os
import datetime
import numpy as np
import random
from KalmanSmoother.Utilities.Observations import SourceArray
from KalmanSmoother.Utilities.Utilities import KalmanPoint


class ArtificialFloats():
	def __init__(self,number,vel_percent,**kwds):
		self.sources = SourceArray()
		self.list = []
		with open (get_kalman_folder()+'code/weddell_vel.pkl','rb') as f:
			vel_spectrum = pickle.load(f)
		self.var_x = np.var(vel_spectrum['dx'])
		self.var_y = np.var(vel_spectrum['dy'])
		for dummy in range(number):
			print(dummy)
			self.list.append(FloatGen.generate_randomly(dummy,self.sources,vel_percent,self.var_x,self.var_y))

class FloatGen(FloatBase):
	type = 'Numerical'
	gps_noise = 0.1
	def __init__(self,floatname,sources,vel_percent,var_x,var_y,toa_noise,gps_chance,toa_number):
		self.floatname = floatname
		weddell_source_list = []
		for item in sources.array:
			if sources.array[item].mission=='Weddell':
				weddell_source_list.append(sources.array[item]) 		

		self.vel_percent = vel_percent

		initial_loc = KalmanPoint(-23.5,-64)
		pos_list = [initial_loc]
		date_list = [datetime.datetime(1986,3,12)]
		for _ in range(100):
			dx,dy = self.return_vel_vector(var_x,var_y)
			pos_list += [pos_list[-1].add_displacement(dx,dy)]
			date_list += [date_list[-1]+datetime.timedelta(days=1)]
		self.exact_pos = pos_list
		self.exact_date = date_list
		self.clock = Clock(self.floatname,0,self.exact_date[0],self.exact_date[-1],drift=0)

		toa_list = []
		toa_date = []
		gps_list = []
		gps_date = []

		for k,(pos,date) in enumerate(zip(self.exact_pos,self.exact_date)):
			
			if random.choice(range(101))<self.gps_chance:
				dx = np.random.normal(scale = self.gps_noise/np.sqrt(2))
				dy = np.random.normal(scale = self.gps_noise/np.sqrt(2))
				gps_list.append(pos.add_displacement(dx,dy))
				gps_date.append(date)
			elif (k == 0)|(k == (len(self.exact_pos)-1)):
				dx = np.random.normal(scale = self.gps_noise/np.sqrt(2))
				dy = np.random.normal(scale = self.gps_noise/np.sqrt(2))
				gps_list.append(pos.add_displacement(dx,dy))
				gps_date.append(date)	
			sources = np.random.choice(weddell_source_list,self.toa_number)
			obs_list = []
			for source in sources:
				dist = (source.position-pos).magnitude
				assert dist<2000
				obs_list.append(source.toa_from_dist(dist)+np.random.normal(scale = self.toa_noise))
				assert source.clock.offset==0
				assert source.clock.drift==0
			if list(sources):
				toa_list.append(zip(obs_list,sources))
				toa_date.append(date)
		self.gps = GPS(gps_list,gps_date,gps_interp=False)
		self.gps.clock = self.clock

		self.toa = TOA(toa_list,toa_date,clock = self.clock)
		self.depth = Depth([],[],self.clock)
		self.stream = Stream([],[],self.clock)
		
	def return_vel_vector(self,var_x,var_y):
		dx = var_x+np.random.normal(var_x)*self.vel_percent
		dy = var_y+np.random.normal(var_y)*self.vel_percent
		return (dx,dy)

	@staticmethod
	def generate_randomly(floatname,sources,vel_percent,var_x,var_y):
		toa_noise = random.choice(range(51))+1
		gps_chance = random.choice(range(101))
		toa_number = random.choice(range(6))
		return FloatGen(floatname,sources,vel_percent,var_x,var_y,toa_noise,gps_chance,toa_number)
