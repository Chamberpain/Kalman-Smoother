import numpy as np
import datetime
import matplotlib.pyplot as plt
from KalmanSmoother.Utilities.Utilities import dx_dy_distance
from geopy.distance import GreatCircleDistance
from KalmanSmoother.Utilities.Observations import Depth,Stream


class ObsHolder(object):
	def __init__(self,floatclass):
		self.gps_class = floatclass.gps
		self.depth_class = floatclass.depth
		self.toa_class = floatclass.toa
		self.stream_class = floatclass.stream

	def toa_detrend(self,toa,sound_source):
		toa = self.gps_class.clock.detrend_offset(toa)
		toa = sound_source.clock.detrend_offset(toa)
		return toa

	def return_J(self,forecast):
		J = []
		if self.gps:
			J.append([1,0,0,0])
			J.append([0,0,1,0])
		if self.interp:
			J.append([1,0,0,0])
			J.append([0,0,1,0])
		for (toa_reading,sound_source) in self.toa:
			dy,dx = dx_dy_distance(sound_source.position,forecast)
			dist = GreatCircleDistance(forecast,sound_source.position).km
			dT_dx = dx/dist*sound_source.slow()
			dT_dy = dy/dist*sound_source.slow()
			J.append([dT_dx,0,dT_dy,0])
		if self.depth:
			k = -1 
			dz_dx,dz_dy = self.depth_class.return_gradient(forecast)
			while any([np.isnan(dz_dx),np.isnan(dz_dy)]):
				k -= 1
				dz_dx,dz_dy = self.depth_class.return_gradient(forecast)
			J.append([dz_dx,0,dz_dy,0]) #now add depth jacobian
		if self.stream:
			k = -1 
			dz_dx,dz_dy = self.stream_class.return_gradient(forecast)
			while any([np.isnan(dz_dx),np.isnan(dz_dy)]):
				k -= 1
				dz_dx,dz_dy = self.stream_class.return_gradient(forecast)
			J.append([dz_dx,0,dz_dy,0]) #now add stream jacobian
		J = np.array(J).reshape([self.get_obs_num(),4])
		return J

	def return_Z(self,analysis):
		Z = []
		for _ in self.gps:
			dy,dx = dx_dy_distance(self.gps_class.obs[0],self.gps[0])
			Z.append(dx)
			Z.append(dy)
		for _ in self.interp:
			dy,dx = dx_dy_distance(self.gps_class.obs[0],self.interp[0])
			Z.append(dx)
			Z.append(dy)
		for (toa_reading,sound_source) in self.toa:
			toa_actual = self.toa_detrend(toa_reading,sound_source)
			Z.append(toa_actual)
		if self.depth:
			Z.append(self.depth_class.return_z(analysis))
		if self.stream:
			Z.append(self.stream_class.return_z(analysis))
		Z = np.array(Z).reshape([self.get_obs_num(),1])
		return Z

	def return_R(self):
		R = []
		if self.gps:
			R.append(self.gps_class.uncertainty)
			R.append(self.gps_class.uncertainty)
		if self.interp:
			R.append(self.gps_class.interp_uncertainty)
			R.append(self.gps_class.interp_uncertainty)
		for dummy in self.toa:
			R.append(self.toa_class.uncertainty)
		if self.depth:
			R.append(self.depth_class.uncertainty)
		if self.stream:
			R.append(self.stream_class.uncertainty)
		obs_num = self.get_obs_num()
		return np.diag(R).reshape([obs_num,obs_num])


	def return_h(self,forecast):
		h = []
		if self.gps:
			dy,dx = dx_dy_distance(self.gps_class.obs[0],forecast)
			h.append(dx)
			h.append(dy)
		if self.interp:
			dy,dx = dx_dy_distance(self.gps_class.obs[0],forecast)
			h.append(dx)
			h.append(dy)
		for (toa_reading,sound_source) in self.toa:
			dist = GreatCircleDistance(sound_source.position,forecast).km
			h.append(sound_source.toa_from_dist(dist))
		if self.depth:
			h.append(self.depth_class.return_z(forecast))
		if self.stream:
			h.append(self.stream_class.return_z(forecast))
		h = np.array(h).reshape([self.get_obs_num(),1])
		return h

	def return_Y(self,Z,h): #this is added so that we can easily append innovation to the observation classes
		return Z-h

	def get_obs_num(self):
		return len(self.gps)*2+len(self.toa)+len(self.depth)+len(self.stream)+len(self.interp)*2 #2 for lat lon, one for depth

	def set_data(self):
		self.toa = self.toa_class.return_data()
		self.gps = self.gps_class.return_data()
		self.depth = self.depth_class.return_data()
		self.stream = self.stream_class.return_data()
		self.interp = self.gps_class.return_interp()


class FilterBase(object):
	I = np.identity(4)
	max_vel_uncert= 30
	max_vel = 35
	max_x_diff = 50
	def __init__(self,float_class,sources,obs_holder,process_position_noise=5,process_vel_noise =1):
		self.obs_holder = obs_holder
		self.sources = sources
		self.float=float_class
		self.process_position_noise = process_position_noise
		self.process_vel_noise = process_vel_noise
		self.Q = np.diag([self.process_position_noise,self.process_vel_noise,self.process_position_noise,self.process_vel_noise])
		self.Q_position_uncert_only = np.diag([self.process_position_noise,0,self.process_position_noise,0])
		self.date_list = []
		self.P_m = []
		self.P_p = [self.Q]
		self.X_p = [self.initialize_X()]
		self.X_m = []
		self.set_date(self.float.gps.date[0])

	def set_date(self,date):
		""" Method to set and record date in all associated source and float clocks
			Input: date you wish to set
			Output: None
		"""
		self.date = date
		self.sources.set_date(date)
		self.float.clock.set_date(date)
		self.obs_holder.set_data()
		self.date_list.append(date)
		print(date)

	def increment_date(self):
		self.set_date(self.date+datetime.timedelta(days=1))

	def decrement_date(self):
		self.set_date(self.date-datetime.timedelta(days=1))

	def initialize_velocity(self):
		""" Method to initialize the velocity of the float. We initialize velocity with our best first guess so that 
		there isnt a shock to the system when the apriori is not in line with the observations
			Input: None
			Output: Initial velocity """

		date_diff = (self.float.gps.date[1]-self.float.gps.date[0]).days	#check how many days between gps observations
		dy,dx = dx_dy_distance(self.float.gps.obs[1],self.float.gps.obs[0])	#calculate distance between 
		lat_vel = dy/date_diff 	# determine x and y components
		lon_vel = dx/date_diff
		return (lat_vel,lon_vel) # return

	def initialize_X(self):
		lat_vel,lon_vel = self.initialize_velocity()
		deploy_loc = self.float.gps.obs[0]
		X = np.array([0,lon_vel, \
		0,lat_vel]).reshape(4,1)
#state vector is defined as lon, lon speed, lat, lat speed
#this is recorded in KM and referenced back to the starting position
		return X

	def increment_filter(self):
		self.X_m.append(self.A.dot(self.X_p[-1])) #create the state forecast
		self.P_m.append(self.P_increment())	#create the covariance forecast

		h = self.obs_holder.return_h(self.pos_from_state(self.X_m[-1]))
		J = self.obs_holder.return_J(self.pos_from_state(self.X_p[-1]))
		Z = self.obs_holder.return_Z(self.pos_from_state(self.X_p[-1]))
		R = self.obs_holder.return_R()
		Y = self.obs_holder.return_Y(Z,h) #innovation
		S = J.dot(self.P_m[-1].dot(J.T))+R 	#innovation covariance
		K = self.P_m[-1].dot(J.T.dot(np.linalg.inv(S)))	#kalman gain
		self.X_p.append(self.X_checker(self.X_m[-1]+K.dot(Y))) #create the analysis state
		self.P_p.append((self.I-K.dot(J)).dot(self.P_m[-1]))	#create the analysis covariance

		try:
			assert self.X_p[-1].shape == (4,1) #state matrix must have x,xdot, y,ydot
			assert self.X_m[-1].shape == (4,1)
			assert self.P_p[-1].shape == (4,4) #error covariances must have all cross terms
			assert self.P_m[-1].shape == (4,4)
			assert not np.isnan(self.X_p[-1]).any() # no nan values in the state or covariance matrices
			assert not np.isnan(self.X_m[-1]).any()
			assert not np.isnan(self.P_p[-1]).any()
			assert not np.isnan(self.P_m[-1]).any()
		except AssertionError:
			raise

	def P_increment(self):
		if self.eig_checker(self.P_p[-1][[1,1,3,3],[1,3,1,3]],self.max_vel_uncert): # if the error covariance is growing too large
			Q = self.Q	#process noise with velocity noise
		else:
			Q = self.Q_position_uncert_only #process noise without velocity noise
		return self.A.dot(self.P_p[-1].dot(self.A.T))+Q #predicted estimate covariance

	def eig_checker(self,C,value):
		C = C.reshape([2,2])
		w,v = np.linalg.eig(C)
		return 2*max(w)*np.sqrt(5.991)<value

	def X_checker(self,X):
		for idx in [1,3]:
			dummy = X[idx]
			if abs(dummy)>self.max_vel:
				X[idx] = np.sign(dummy)*self.max_vel
		for idx in [0,2]:
			dummy_1 = X[idx]
			dummy_2 = self.X_m[-1][idx]
			diff = dummy_1-dummy_2
			if abs(dummy_1-dummy_2)>self.max_x_diff:
				X[idx] = dummy_2 + np.sign(diff)*self.max_x_diff
		return X

	def pos_from_state(self,state):
		pos = self.float.gps.obs[0].add_displacement(state.flatten()[0],state.flatten()[2])
		if np.isnan(pos.latitude):
			pos = self.float.gps.obs[0]
		return pos

	def toa_detrend(self,toa,sound_source):
		toa = self.float.clock.detrend_offset(toa)
		toa = sound_source.clock.detrend_offset(toa)
		return toa

	def state_vector_to_pos(self):		
		vel = []
		pos = []
		for _ in self.X_p:
			x = _[0]
			y = _[2]
			if (x==0) and (y==0):
				pos.append(self.float.gps.obs[0])
				vel.append((0,0))
			else:
				pos.append(self.float.gps.obs[0].add_displacement(x,y))
				vel.append((_[1],_[3]))
		return (pos,vel)


	def obs_date_diff_list(self,date_list):
		dates = np.sort(self.float.toa.date+self.float.gps.date)
		date_diff_list = []
		for date in date_list:
			dates_holder = dates[dates<date]
			if list(dates_holder):
				diff = (date-max(dates_holder)).days
				date_diff_list.append(diff)
			else:
				date_diff_list.append(0)
		return date_diff_list

	def linear_interp_between_obs(self):
		unique_date_list = np.sort(np.unique(self.date_list))
		date_diff_list = self.obs_date_diff_list(unique_date_list)
		idxs = np.where(np.array(date_diff_list)==14)[0]
		obs_dates = np.array(np.sort(self.float.toa.date+self.float.gps.date))
		unique_date_list = np.array(unique_date_list)
		for idx in idxs:
			date = unique_date_list[idx]
			min_date = max(obs_dates[obs_dates<date])-datetime.timedelta(days=4)
			max_date = min(obs_dates[obs_dates>date])+datetime.timedelta(days=4)
			min_idx = self.date_list.index(min_date)
			max_idx = self.date_list.index(max_date)

			min_pos = self.float.pos[min_idx]
			max_pos = self.float.pos[max_idx]

			time_delta = (max_date-min_date).days
			pos_insert = [min_pos + (_+1)*(max_pos-min_pos)/(time_delta+2) for _ in range(time_delta)]
			self.float.pos[min_idx:max_idx] = pos_insert


class LeastSquares(FilterBase):
	def __init__(self,float_class,sources,obs_holder,**kwds):
		super(LeastSquares,self).__init__(float_class,sources,obs_holder,**kwds)
		self.A=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) 
		self.increment()
# state matrix propogates to future with X(t)=X(t-1), V(t)=V(t-1)
	def increment(self):
		while self.date<=self.float.gps.date[-1]:
			#print(self.date)
			self.increment_date()
			assert self.date == self.float.clock.date
			self.increment_filter()
		self.float.ls_pos = [self.pos_from_state(_) for _ in self.X_p]
		assert len(self.float.ls_pos)==len(self.date_list)

class Kalman(FilterBase):
	def __init__(self,float_class,sources,obs_holder,**kwds):
		super(Kalman,self).__init__(float_class,sources,obs_holder,**kwds)
		self.A=np.array([[1,1,0,0],[0,0.90,0,0],[0,0,1,1],[0,0,0,0.90]]) 
		self.increment()
# state matrix propogates to future with X(t)=X(t-1)+V(t-1), V(t)=V(t-1)
	def increment(self):
		while self.date<=self.float.gps.date[-1]:
			self.increment_date()
			assert self.date == self.float.clock.date
			self.increment_filter()
		self.float.pos = [self.pos_from_state(_) for _ in self.X_p]
		assert len(self.float.pos)==len(self.date_list)
		self.float.pos_date = self.date_list
		self.float.kalman_pos = self.float.pos

class Smoother(Kalman):
	def __init__(self,float_class,sources,obs_holder,**kwds):
		super(Smoother,self).__init__(float_class,sources,obs_holder,**kwds)
		self.X = []
		self.P = []
		self.X.append(self.X_p.pop())
		self.P.append(self.P_p.pop())
		self.decrement_date()
		while self.date>=self.float.gps.date[0]:
			self.decrement_filter()
			self.decrement_date()
		self.float.pos = [self.pos_from_state(_) for _ in self.X[::-1]]
		self.float.P = self.P

	def decrement_filter(self):
		K = self.P_p[-1].dot(self.A.T.dot(np.linalg.inv(self.P_m[-1])))
		self.P.append(self.P_p.pop() - K.dot((self.P_m.pop()-self.P[-1]).dot(K.T)))
		self.X.append(self.X_p.pop() + K.dot(self.X[-1]-self.X_m.pop()))
