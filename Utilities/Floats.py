import os
import datetime
import numpy as np
from KalmanSmoother.Utilities.Observations import Clock,GPS,Depth,Stream,TOA,SourceArray
from KalmanSmoother.Utilities.Utilities import KalmanPoint
import pandas as pd
import geopy
from GeneralUtilities.Filepath.search import find_files
from KalmanSmoother.Utilities.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader
import scipy.io as io
from sklearn.linear_model import LinearRegression
from KalmanSmoother.Utilities.DataLibrary import float_info
from GeneralUtilities.Compute.list import flat_list
from scipy import stats
from GeneralUtilities.Filepath.instance import get_base_folder
from KalmanSmoother.__init__ import ROOT_DIR as project_base
from GeneralUtilities.Compute.list import TimeList
from GeneralUtilities.Data.depth.depth_utilities import ETopo1Depth
from GeneralUtilities.Data.agva.agva_utilities import AVGAStream
import pickle
import random


file_handler = FilePathHandler(ROOT_DIR,'Floats')
base_folder = get_base_folder() + 'Raw/ARGO/aoml/'


def end_adder(dummy_time_list,dummy_pos_list):
	adder_number = 6
	dummy_pos_list = [dummy_pos_list[0]]*adder_number+dummy_pos_list + [dummy_pos_list[-1]]*adder_number
	time_addition = [dummy_time_list[-1] + datetime.timedelta(days = (n+1)) for n in range(adder_number)]
	time_subtraction = [dummy_time_list[0] - datetime.timedelta(days = (n+1)) for n in range(adder_number)]
	dummy_time_list = time_subtraction+dummy_time_list+time_addition
	return (dummy_time_list,dummy_pos_list)

class SmoothDepth(ETopo1Depth):
	def return_data(self):
		return [True]
	def set_observational_uncertainty(self,obs_error):
		self.uncertainty = obs_error

class SmoothStream(AVGAStream):
	def return_data(self):
		return [True]
	def set_observational_uncertainty(self,obs_error):
		self.uncertainty = obs_error

class FloatBase(object):
	"""This is the base class of all floats"""

	def __init__(self,match,sources,**kwds):
		self.floatname = self.name_parser(match)
		self.gps = self.gps_parser(self.floatname)
		self.initialize_clock() #gets data from various sources (inital and final + drift and offset) to start the clock
		self.gps.clock = self.clock

	def obs_dates(self):
		return np.sort(np.unique(self.toa.date+self.gps.date))

	def percent_obs(self):
		dates = self.obs_dates()
		return float(len(dates))/(max(dates)-min(dates)).days

	def return_pos(self):
		pos_index = self.pos_date.index(self.clock.date)
		pos = self.pos[pos_index]
		return pos

class FloatToken(FloatBase):
	def __init__(self,match,sources,**kwds):
		super(FloatToken,self).__init__(match,sources,**kwds)
		mat = io.loadmat(match)
		self.toa = self.toa_parser(sources,mat)
		if abs(self.gps.date[0]-self.toa.date[0]).days>20:
			print(self.gps.date[0])
			print(self.toa.date[0])
			print(self.toa.date[-1])
			print(self.gps.date[-1])

	def toa_parser(self,sources,mat):
		toa = mat['TOA'].flatten() 
# this is the actual time of arrival data for all signals (many of which are erroneous)
		toa_sel = mat['TOA_SEL'].flatten() 
# these are the matchup of signals to sound sources
		toa_date = mat['TOA_DATE'].flatten() 
# these are the matchup of dates to all recorded toas
		toa_filter = toa_sel!=0 
# filter out the toas that do not correspond to received sound source signal
		toa = toa[toa_filter]
		toa_sel = toa_sel[toa_filter]
		toa_date = toa_date[toa_filter]
		soso_ref = mat['SOSOREF'].tolist() 
# this is the list that relates the number in TOA_SEL to a sound source ID
		soso_ref = [s.replace(" ", "") for s in soso_ref]
		soso_ref = ['dummy']+soso_ref 
# we do this because matlab indexes +1
		toa_sel = np.array([soso_ref[_] for _ in toa_sel])
# relate the sound source selection number to a sound source ID
		toa_date_list = []
		toa_list = []
		for date in np.sort(np.unique(toa_date)):
			toa_date_list.append(date)
			date_filter = toa_date==date
			toa_holder = toa[date_filter]
			toa_sel_holder = toa_sel[date_filter]
			toa_sel_holder = [sources.array[_] for _ in toa_sel_holder]
			toa_list.append(list(zip(toa_holder,toa_sel_holder)))
		toa_date_list = self.date_parser(toa_date_list)
		return TOA(toa_list,toa_date_list,self.clock)

	def date_parser(self,date_list):
		ref_date = datetime.datetime(2008,3,9)
		date = [ref_date + datetime.timedelta(days=int(dummy)) for dummy in np.array(date_list).astype(int)-14536]
		date = TimeList(date)
		date.set_ref_date(ref_date)
		assert min(date)>datetime.datetime(2006,1,1)
		assert max(date)<datetime.datetime(2015,1,1)
		return date

class DIMESFloatBase(FloatToken):
	type = 'DIMES'
	depth = SmoothDepth.load().regional_subsample(-20,-115,-40,-68)
	depth.guassian_smooth(sigma=4)
	depth.get_gradient()
	assert isinstance(depth,SmoothDepth)
	stream = SmoothStream.load()
	stream = stream.griddata_subsample(stream.z)
	stream = stream.regional_subsample(-20,-115,-40,-68)
	stream.get_gradient()
	assert isinstance(stream,SmoothStream)	


	def __init__(self,match,sources,**kwds):
		super(DIMESFloatBase,self).__init__(match,sources,**kwds)
		self.load_trj_file(match)
		assert self.trj_date[0]<=self.toa.date[0]
		assert self.trj_date[0]<=self.gps.date[6]
		assert self.trj_date[-1]>=self.toa.date[-1]
		assert self.trj_date[-1]>=self.gps.date[7]



	def name_parser(self,match):
		name = match.split('/')[-1]
		name.split('.')[0]
		name = ''.join([i for i in name if i.isdigit()])
		return int(name)

	def load_trj_file(self,match):
		dir_path = os.path.dirname(match)
		trj_path = find_files(dir_path,'*.trj4')[0]
		trj_df = pd.read_table(trj_path,skiprows=1,sep='\s+',names=['Date','Lat','Lon','dummy1','dummy2','dummy3','dummy4'],usecols=['Date','Lat','Lon'])
		trj_df = trj_df.replace(-999,np.nan)
		trj_df = trj_df.interpolate(limit_direction='both')
		trj_df = trj_df.drop_duplicates(subset=['Date'])
		start_date = self.gps.date[6]
		trj_df = trj_df[trj_df['Lat']>-90]
		trj_df = trj_df[trj_df['Lon']>-180]
		self.trj_date = [start_date + datetime.timedelta(days = x) for x in (trj_df.Date-trj_df.Date.tolist()[0]).tolist()]		
		trj_df['Date'] = self.trj_date
		self.trj_pos = [KalmanPoint(_[0],_[1]) for _ in list(zip(trj_df['Lat'].tolist(),trj_df['Lon'].tolist()))]
		if self.gps.date[-1]>self.trj_date[-1]:
			pos = []
			pos.append(self.gps.obs[6])
			pos.append(self.trj_pos[-1])
			time = []
			time.append(self.gps.date[6])
			time.append(self.trj_date[-1])
			time,pos = end_adder(time,pos)
			self.gps = GPS(pos,time,gps_interp_uncertainty=False)			
			self.gps.clock = self.clock
			self.toa.obs = [obs for date,obs in zip(self.toa.date,self.toa.obs) if date <=self.trj_date[-1]] 
			self.toa.date = [date for date,obs in zip(self.toa.date,self.toa.obs) if date <=self.trj_date[-1]] 

	def initialize_clock(self):
		dummy_1,dummy_2,initial_date,dummy_4,dummy_5,final_date, \
		initial_offset, final_offset = float_info[str(self.floatname)]
		self.clock = Clock(self.floatname,initial_offset,self.gps.date[6],self.gps.date[-6],final_offset=final_offset)

	def gps_parser(self,ID):
		deployed_lat, deployed_lon, deployed_date, recovered_lat,recovered_lon, recovered_date,\
		dummy_1, dummy_2 = float_info[str(ID)]
#float info also contains information about clock drift
		pos = []
		pos.append(KalmanPoint(deployed_lat,deployed_lon))
		pos.append(KalmanPoint(recovered_lat,recovered_lon))
		time = self.date_parser([deployed_date,recovered_date])
		time,pos = end_adder(time,pos)
		return GPS(pos,time,gps_interp_uncertainty=False)

class WeddellFloat(FloatToken):
	type = 'Weddell'
	depth = SmoothDepth.load().regional_subsample(15,-61,-55,-77)
	depth.guassian_smooth(sigma=4)
	depth.get_gradient()
	assert isinstance(depth,SmoothDepth)
	stream = SmoothStream.load()
	stream = stream.griddata_subsample(stream.z)
	stream.regional_subsample(15,-61,-55,-77)
	stream.get_gradient()
	assert isinstance(stream,SmoothStream)	

	def __init__(self,match,sources,**kwds):
		super(WeddellFloat,self).__init__(match,sources,**kwds)
	def initialize_clock(self):
		self.clock = Clock(self.floatname,0,self.gps.date[0],self.gps.date[-1],drift=0)


	def name_parser(self,match):
		from KalmanSmoother.Utilities.DataLibrary import float_library
		name = match.split('/')[-1]
		name.split('.')[0]
		name = name.split('_')[0]
		name = ''.join([i for i in name if i.isdigit()])
		name = float_library[name]
		print('the float name is ',name)
		return int(name)

	def gps_parser(self,mat):
		root = base_folder+str(self.floatname)+'/'
		data = ArgoReader(root)
		gps_list = [KalmanPoint(_.latitude,_.longitude) for _ in data.prof.pos._list]
		date_list = [datetime.datetime(_.year,_.month,_.day) for _ in data.prof.date._list]

		date_list,gps_list = end_adder(date_list,gps_list)
		return GPS(gps_list,date_list)

class AllFloats(object):
	sources = SourceArray()
	list = []
	def __init__(self,*args,**kwargs):
		dir_,float_type,ext_ = self.sources_dict
		base_matches = find_files(dir_,ext_)
		for match in base_matches:
			print(match)
			float_ = float_type(match,self.sources)
			self.list.append(float_)
		self.combine_classes()
		self.assign_soso_drift_errors()

	@classmethod
	def reset_floats(cls):
		cls.list = []

	def combine_classes(self):
		new_list = []
		name_list = np.array([_.floatname for _ in self.list])
		for name in np.unique(name_list): 
			mask = name_list==name
			sublist = np.array(self.list)[mask]
			base = sublist[0]
			for dummy in sublist[1:]:
				base.toa.obs += dummy.toa.obs
				base.toa.date += dummy.toa.date
				mask = np.isin(dummy.gps.date,base.gps.date)
				base.gps.obs += np.array(dummy.gps.obs)[~mask].tolist()
				base.gps.date += np.array(dummy.gps.date)[~mask].tolist()
			base.date = min(base.gps.date)
			base.toa.obs = [x for _,x in sorted(list(zip(base.toa.date,base.toa.obs)), key=lambda x: x[0])]
			base.gps.obs = [x for _,x in sorted(list(zip(base.gps.date,base.gps.obs)), key=lambda x: x[0])]
			base.toa.date = sorted(base.toa.date)
			base.gps.date = sorted(base.gps.date)

			new_list.append(base)
		self.list = new_list


class WeddellAllFloats(AllFloats):
	sources_dict = (project_base+'/Data/Data/',WeddellFloat,'*.itm')
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	def assign_soso_drift_errors(self):
		error_list = []
		for float_ in self.list:
			gps_date = \
			float_.gps.date+\
			[dummy - datetime.timedelta(days=1) for dummy in float_.gps.date]+\
			[dummy + datetime.timedelta(days=1) for dummy in float_.gps.date]+\
			[dummy + datetime.timedelta(days=2) for dummy in float_.gps.date]+\
			[dummy - datetime.timedelta(days=2) for dummy in float_.gps.date]

			gps = float_.gps.obs*5
			toa = float_.toa.obs*5
			toa_date = float_.toa.date*5
			intersection = set(gps_date).intersection(set(toa_date))
			for intersection_dummy in intersection:
				float_.clock.set_date(intersection_dummy)
				loc = gps[gps_date.index(intersection_dummy)]
				toa_holder = toa[toa_date.index(intersection_dummy)]
				for toa_token in toa_holder:
					toa_measured,soso = toa_token
					soso.clock.set_date(intersection_dummy)
					toa_measured_detrend = float_.clock.detrend_offset(toa_measured)
					soso_measured_detrend = soso.clock.detrend_offset(toa_measured_detrend)
					dist = geopy.distance.GreatCircleDistance(soso.position,loc).km # this is in km
					days = (intersection_dummy-soso.clock.initial_date).days
					error_list.append((soso.ID,float_.floatname,dist,toa_measured_detrend,days,intersection_dummy,soso_measured_detrend))
		df = pd.DataFrame(error_list,columns=['SOSO','Float','Dist','TOA','Days','Date','SOSO Initial Observed'])
		df = df[df['SOSO Initial Observed']>-50]
		lr = LinearRegression()
		speed_of_sound, _, _, _ = np.linalg.lstsq(df['SOSO Initial Observed'].to_numpy().reshape(-1,1),df['Dist'].to_numpy().reshape(-1,1))

		df['SOSO TOA'] = [self.sources.array[soso].toa_from_dist(toa) for soso,toa in zip(df['SOSO'].tolist(),df['Dist'].tolist())]
		df['Misfit'] = df['SOSO Initial Observed']-df['SOSO TOA']

		self.sources.set_speed(speed_of_sound[0][0])

		for _,soso in self.sources.array.items():
			print(soso.ID)
			print(soso.clock.initial_offset)
			print(soso.clock.drift)
			df_holder = df[df.SOSO==soso.ID]
			if df_holder.empty:
				continue
			lr.fit(df_holder['Days'].to_numpy().reshape(-1,1),df_holder['Misfit'].to_numpy().reshape(-1,1))
			soso.clock.initial_offset += lr.intercept_[0]
			soso.clock.drift += lr.coef_[0][0]
			print(soso.clock.initial_offset)
			print(soso.clock.drift)

			corrected_toa_list = []
			for date,toa in zip(df_holder['Date'].tolist(),df_holder['TOA'].tolist()):
				soso.clock.set_date(date)
				corrected_toa = soso.clock.detrend_offset(toa)
				print('toa is ',toa)
				print('corrected toa is',toa)
				print('offset is ',soso.clock.offset)
				corrected_toa_list.append(soso.clock.detrend_offset(toa))
			df_holder['Corrected TOA'] = corrected_toa_list
			df_holder['Error'] = df_holder['SOSO TOA']-df_holder['Corrected TOA']

			if not df_holder.empty:
				print('the dataframe is not empty')
				soso.error_dataframe = df_holder
			else:
				print('the dataframe is empty')
				soso.error_dataframe = pd.DataFrame([])

class DIMESAllFloats(AllFloats):
	sources_dict =	(project_base+'/Data/DIMES/',DIMESFloatBase,'rf*toa.mat')

	def assign_soso_drift_errors(self):
		df_list = []
		for float_ in self.list:
			dist_error_list,toa_error_list,dist_list,soso_list,date_return_list,obs_list = float_.toa.calculate_error_list(float_.trj_pos,float_.trj_date)
			df_list.append(pd.DataFrame({'floatname':float_.floatname,'dist_error':dist_error_list,'toa_error':toa_error_list,'dist_list':dist_list,'soso_list':soso_list,'date_return_list':date_return_list,'obs_list':obs_list}))
		df = pd.concat(df_list)
#it turns out the speed of sound turns out ot almost exactly be 1.5 km/sec, so no need to change anything
		speed_of_sound, _, _, _ = np.linalg.lstsq(np.array(df.obs_list.tolist()).reshape(-1,1),np.array(df.dist_list.tolist()).reshape(-1,1))
		df = df[df.toa_error.abs()<50]
		for _,soso in self.sources.array.items():
			print(soso.ID)
			print(soso.clock.initial_offset)
			print(soso.clock.drift)
			df_holder = df[df.soso_list==soso.ID]
			if df_holder.empty:
				continue
			df_holder['Days'] = df_holder.date_return_list-df_holder.date_return_list.min()
			lr = LinearRegression()
			lr.fit(np.array([x.days for x in df_holder.Days.tolist()]).reshape(-1,1),df_holder['toa_error'].to_numpy().reshape(-1,1))
			soso.clock.initial_offset += lr.intercept_[0]
			soso.clock.drift += lr.coef_[0][0]
			print(soso.clock.initial_offset)
			print(soso.clock.drift)
		for float_ in self.list:
			print(float_.floatname)
			obs_list = []
			df_holder = df[df.floatname==float_.floatname]
			date_list = [datetime.datetime.combine(x.date(),datetime.datetime.min.time()) for x in df_holder.date_return_list]
			df_holder['Date']=date_list
			date_list = sorted(np.unique(date_list))
			for date in date_list:
				df_token = df_holder[df_holder['Date']==date]
				date_idx = float_.toa.date.index(date)
				obs_token = float_.toa.obs[date_idx]
				obs_list.append([(toa_token,soso_token) for toa_token,soso_token in obs_token if soso_token.ID in df_token.soso_list.tolist()])
			float_.toa.obs=obs_list
			float_.toa.date=date_list


class ArtificialFloats():
	sources = SourceArray()
	sources.set_offset(0)
	sources.set_drift(0)
	sources.set_speed(1.5)
	def __init__(self,**kwds):

		self.list = []
		with open (project_base+'/Data/weddell_vel.pkl','rb') as f:
			vel_spectrum = pickle.load(f)
		self.var_x = np.var(vel_spectrum['dx'])*0.3
		self.var_y = np.var(vel_spectrum['dy'])*0.3

	def random(self,vel_percent):
		return FloatGen.generate_randomly('particle',self.sources,vel_percent,self.var_x,self.var_y)

class FloatGen(FloatBase):
	type = 'Numerical'
	gps_noise = 0.1
	def __init__(self,floatname,sources,vel_percent,var_x,var_y,toa_noise,gps_chance,toa_number):
		self.floatname = floatname
		self.toa_noise = toa_noise
		self.toa_number = toa_number
		self.sources = sources
		self.sources.set_location(KalmanPoint(-64,-23.5))
		weddell_source_list = []
		for item in sources.array:
			if sources.array[item].mission=='Weddell':
				weddell_source_list.append(sources.array[item]) 		
		self.vel_percent = vel_percent
		initial_loc = KalmanPoint(-64,-23.5)
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
			
			if random.choice(range(101))<gps_chance:
				dx = np.random.normal(scale = self.gps_noise/np.sqrt(2))
				dy = np.random.normal(scale = self.gps_noise/np.sqrt(2))
				gps_list.append(pos.add_displacement(dx,dy))
				gps_date.append(date)
			elif (k == 0)|(k == (len(self.exact_pos)-1)):
				dx = np.random.normal(scale = self.gps_noise/np.sqrt(2))
				dy = np.random.normal(scale = self.gps_noise/np.sqrt(2))
				gps_list.append(pos.add_displacement(dx,dy))
				gps_date.append(date)	
			sources = np.random.choice(weddell_source_list,toa_number)
			obs_list = []
			for source in sources:
				dist = geopy.distance.GreatCircleDistance(source.position,pos).km
				assert dist<2000
				obs_list.append(source.toa_from_dist(dist)+np.random.normal(scale = toa_noise))
				assert source.clock.offset==0
				assert source.clock.drift==0
			if list(sources):
				toa_list.append(list(zip(obs_list,sources)))
				toa_date.append(date)
		self.gps = GPS(gps_list,gps_date,gps_interp_uncertainty=False)
		self.gps.clock = self.clock

		self.toa = TOA(toa_list,toa_date,clock = self.clock)
		self.toa.set_observational_uncertainty(toa_noise)
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

