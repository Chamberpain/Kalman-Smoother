from __future__ import print_function
import os
import datetime
import numpy as np
from KalmanSmoother.Utilities.DataLibrary import SOSO_coord,SOSO_drift,float_info
from plot_utilities.Basemap.eulerian_plot import Basemap
import LatLon
from sets import Set
import scipy
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random
from netCDF4 import Dataset
import pickle
import gsw
import sys
from compute_utilities.list_utilities import flat_list
from data_save_utilities.lagrangian.argo.argo_read import ArgoReader
from compute_utilities.constants import degree_dist
from data_save_utilities.search_utilities import find_files
from scipy import stats


class ArtificialFloats():
	def __init__(self,number,sources,vel_percent,**kwds):
		self.sources = sources
		self.list = []
		with open (get_kalman_folder()+'code/weddell_vel.pkl','rb') as f:
			vel_spectrum = pickle.load(f)
		self.var_x = np.var(vel_spectrum['dx'])
		self.var_y = np.var(vel_spectrum['dy'])
		for _ in range(number):
			print(_)
			self.list.append(FloatGen(_,self.sources,vel_percent,self.var_x,self.var_y))

class FloatGen(FloatBase):
	def __init__(self,floatname,sources,vel_percent,var_x,var_y):
		self.type = 'Numerical'
		self.floatname = floatname
		weddell_source_list = []
		for item in sources.array:
			if sources.array[item].mission=='Weddell':
				weddell_source_list.append(sources.array[item]) 		
		self.toa_noise = random.choice(range(51))+1
		self.gps_chance = random.choice(range(101))
		self.toa_number = random.choice(range(6))
		self.vel_percent = vel_percent
		self.gps_noise = 0.1

		initial_loc = LatLon.LatLon(-64,-23.5)
		pos_list = [initial_loc]
		date_list = [datetime.datetime(1986,3,12)]
		for _ in range(100):
			pos_list += [pos_list[-1]+self.return_vel_vector(var_x,var_y)]
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
				gps_list.append(pos+LatLon.GeoVector(dx,dy))
				gps_date.append(date)
			elif (k == 0)|(k == (len(self.exact_pos)-1)):
				dx = np.random.normal(scale = self.gps_noise/np.sqrt(2))
				dy = np.random.normal(scale = self.gps_noise/np.sqrt(2))
				gps_list.append(pos+LatLon.GeoVector(dx,dy))
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
		return LatLon.GeoVector(dx,dy)



