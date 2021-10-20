from KalmanSmoother.Utilities.DataLibrary import SOSO_coord,SOSO_drift,float_info
from KalmanSmoother.Utilities.Utilities import KalmanPoint
import datetime
import numpy as np 
from sklearn.linear_model import LinearRegression
import geopy
from GeneralUtilities.Compute.list import flat_list



class Clock():
	"""
	class that operates to keep time for the individual floats and sound sources called.
	inputs - 
	ID: what the clock is named
	initial_offset: the initial recorded clock offset in seconds
	initial_date: date of the initial offset
	final_date: date of the final offset (or when the clock was taken out of service)
	drift: The measured clock drift
	"""
	def __init__(self,ID,initial_offset,initial_date,final_date,final_offset=None,drift=None,**kwds):
		self.ID = ID
		self.initial_offset=initial_offset
		self.initial_date = initial_date
		self.final_date = final_date
		if not final_offset is None:
			diff_offset = final_offset-self.initial_offset
			total_day_diff = (self.final_date-self.initial_date).days
			self.drift = float(diff_offset)/total_day_diff
		else: 
			self.drift = drift

	def calculate_offset_from_drift(self,date):
		day_diff = (date-self.initial_date).days
		return self.initial_offset+self.drift*day_diff

	def set_date(self,date):
		self.date = date
		self.offset = self.calculate_offset_from_drift(date)

	def increment_date(self):
		self.set_date(self.date + datetime.timedelta(days=1))

	def decrement_date(self):
		self.set_date(self.date - datetime.timedelta(days=1))

	def detrend_offset(self,time):
		return time-self.offset

class ObsBase(object):
	"""this base class defines all functions of observations"""
	def __init__(self,obs,date,clock,**kwds):
		self.clock = clock
		self.date = date
		self.obs = obs
		
		self.error = {'kalman':[],'smoother':[],'innovation':[]}
		self.interp_error = {'kalman':[],'smoother':[],'innovation':[]}

	def return_data(self):
		try:
			obs_index = self.date.index(self.clock.date)
			obs_return = [self.obs[obs_index]]
		except ValueError:
			obs_return = []
		return obs_return

	def return_interp(self):
		try:
			interp_index = self.interp_date.index(self.clock.date)
			interp_return = [self.interp_obs[interp_index]]
		except ValueError:
			interp_return = []
		return interp_return

	def hist_plot(self):
		smooth_error = self.return_error('smoother')
		kalman_error = self.return_error('kalman')
		plt.hist(smooth_error,label = 'Kalman Smoother',alpha=0.2,bins=50)
		plt.hist(kalman_error,label = 'Kalman Filter',alpha=0.2,bins=50)
		plt.legend(prop={'size': 10})
		plt.title('Histogram of '+self.ID+' Error')

	def return_error(self,label):
		return flat_list(self.error[label])

class GPS(ObsBase):
	"""this class defines all functions of gps observations"""
	def __init__(self,obs,date,gps_interp=True,**kwds):
		super(GPS,self).__init__(obs,date,[],**kwds) #this is a hack for the clock because we need gps to initialize the clock
		if gps_interp:
			self.interp_calc()
		else:
			self.interp_obs = []
			self.interp_date = []
		self.dx_error = {'kalman':[],'smoother':[],'innovation':[]}
		self.dy_error = {'kalman':[],'smoother':[],'innovation':[]}
		self.dx_interp_error = {'kalman':[],'smoother':[],'innovation':[]}
		self.dy_interp_error = {'kalman':[],'smoother':[],'innovation':[]}
		self.ID = 'GPS'

	def interp_calc(self):
		gps_days = [(_ - self.date[0]).days for _ in self.date]
		interp_days = np.arange(0,max(gps_days),2)
		lat = [_.latitude for _ in self.obs]
		lon = [_.longitude for _ in self.obs]
		interp_lat = np.interp(interp_days,gps_days,lat)
		interp_lon = np.interp(interp_days,gps_days,lon)
		pos = [KalmanPoint(_[0],_[1]) for _ in list(zip(interp_lat,interp_lon))]
		date = [self.date[0] + datetime.timedelta(days = k) for k in interp_days.tolist()]
		self.interp_obs = pos
		self.interp_date = date

class Depth(ObsBase):
	"""this class defines all functions of depth observations"""
	def __init__(self,obs,date,clock,**kwds):
		super(Depth,self).__init__(obs,date,clock,**kwds)
		self.ID = 'Depth'

class Stream(ObsBase):
	"""this class defines all functions of depth observations"""
	def __init__(self,obs,date,clock,**kwds):
		super(Stream,self).__init__(obs,date,clock,**kwds)
		self.ID = 'Stream'

class TOA(ObsBase):
	"""this class defines all functions of toa observations"""
	def __init__(self,obs,date,clock,**kwds):
		super(TOA,self).__init__(obs,date,clock,**kwds)

	def calculate_error_list(self,pos_list,date_list):
		dist_error_list = []
		toa_error_list = []
		soso_list = []
		dist_list = []
		date_return_list = []
		pos_dist_list = []
		for k,(date,pos) in enumerate(list(zip(date_list,pos_list))):
			try:
				idx = self.date.index(date)
				for (obs,soso) in self.obs[idx]:
					soso.clock.set_date(date)
					obs = soso.clock.detrend_offset(obs)
					toa_dist = soso.dist_from_toa(obs)
					pos_dist = geopy.distance.GreatCircleDistance(soso.position,pos).km
					pos_toa = soso.toa_from_dist(pos_dist)
					pos_dist_list.append(pos_dist)
					toa_error_list.append(obs-pos_toa)
					dist_list.append(pos_dist)
					error = toa_dist-pos_dist
					dist_error_list.append(error)
					soso_list.append(soso.ID)
					date_return_list.append(date)
			except ValueError:
				continue
		return dist_error_list,toa_error_list,dist_list,soso_list,date_return_list,pos_dist_list

class SoundSource():
	"""class object that holds a clock an id and a position. Keeps track of the date and the clock offset"""
	def __init__(self,ID,**kwds):
		self.ID = ID
		print(ID)
		self.position = KalmanPoint(SOSO_coord[ID][0],SOSO_coord[ID][1])
		self.mission = SOSO_coord[ID][2]
		recorded_drift,date_start,offset,extraction_date = SOSO_drift[ID]
		self.clock = Clock(ID,offset,date_start,extraction_date,drift=recorded_drift)
		self.speed_of_sound = 1.467 # km/s
		self.error = {'kalman':[],'smoother':[],'innovation':[]}

	def toa_from_dist(self,dist):
		toa = abs(dist)*1/self.speed_of_sound
		return toa

	def dist_from_toa(self,toa):
		dist = self.speed_of_sound*abs(toa)
		return dist

	def slow(self):
		return 1/self.speed_of_sound

	def calculate_offset_and_sound_speed(self):
		mlr = LinearRegression()
		mlr.fit(self.error_dataframe[['Dist','Days']],self.error_dataframe['TOA'])
		self.clock.initial_offset = mlr.intercept_
		self.speed_of_sound = 1/mlr.coef_[0]
		self.clock.drift = mlr.coef_[1]

		error_list = []
		for row in self.error_dataframe.iterrows():
			self.clock.set_date(row[1].Date)
			error_list.append(self.toa_from_dist(row[1]['Dist'])-self.clock.detrend_offset(row[1]['TOA']))
		self.error_dataframe['Error']=error_list

	def plot_error_dataframe(self,variable):
		fig,ax = plt.subplots() 
		for dummy_float in self.error_dataframe.Float.unique():
			dummy_dataframe = self.error_dataframe[self.error_dataframe.Float==dummy_float]
			dummy_dataframe.set_index(variable).sort_index()['Error'].plot(marker = 'x',ax=ax)
		plt.title('ID = '+str(self.ID)+' by '+variable)
		plt.ylabel('Error (s)', fontsize=18)

	def plot_errors(self):
		plt.figure()
		smooth_error = [item for sublist in self.error['smoother'] for item in sublist]
		kalman_error = [item for sublist in self.error['kalman'] for item in sublist]
		bins=np.histogram(np.hstack((smooth_error,kalman_error)), bins=50)[1]
		plt.hist(smooth_error,bins,label = 'Kalman Smoother',alpha=0.2)
		plt.hist(kalman_error,bins,label = 'Kalman Filter',alpha=0.2)
		plt.gca().set_yscale("log")
		plt.legend(prop={'size': 10})
		plt.title('Histogram of '+self.ID+' Error')
		plt.xlabel('Seconds', fontsize=18)
		plt.savefig('hist_'+self.ID+'_error')
		plt.close()

	def return_error(self,label):
		return flat_list(self.error[label])

class SourceArray():
	""" Source array is a class that holds many sound sources. Can set the date of all sources, 
increment the date of all sources, or decrement the date of all sources"""
	def __init__(self,**kwds):
		self.array = dict([(_,SoundSource(_)) for _ in SOSO_coord.keys()])

	def set_date(self,date):
		for _ in self.array:
			self.array[_].clock.set_date(date)

	def increment_date(self):
		for _ in self.array:
			self.array[_].clock.increment_date()

	def decrement_date(self):
		for _ in self.array:
			self.array[_].clock.decrement_date()

	def plot_errors(self):
		for _ in self.array.items():
			if _[1].error['smoother']: 
				_[1].plot_errors()

	def return_error(self):
		token_list = []
		for _ in self.array.items():
			token_list += _[1].return_error('smoother') 
		return token_list

	def reset_error(self):
		for _ in self.array.items():
			_[1].error['smoother'] = []
			_[1].error['kalman'] = [] 

	def return_misfit(self):
		error = self.return_error()
		return np.linalg.norm(error)

	def set_speed(self,speed):
		for _ in self.array.items():
			_[1].speed_of_sound = speed

	def set_drift(self,drift):
		for _ in self.array.items():
			_[1].clock.drift = drift

	def set_offset(self,offset):
		for _ in self.array.items():
			_[1].clock.offset = offset
			_[1].clock.initial_offset = offset

	def set_location(self,initial_loc):
		for _ in self.array.items():
			dy = random.random()*random.choice([-1,1])*200
			dx = random.random()*random.choice([-1,1])*200
			_[1].position = initial_loc.add_displacement(dx,dy)