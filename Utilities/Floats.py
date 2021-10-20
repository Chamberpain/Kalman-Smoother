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


file_handler = FilePathHandler(ROOT_DIR,'Floats')
base_folder = get_base_folder() + 'Raw/ARGO/aoml/'

def end_adder(dummy_time_list,dummy_pos_list):
	adder_number = 6
	dummy_pos_list = [dummy_pos_list[0]]*adder_number+dummy_pos_list + [dummy_pos_list[-1]]*adder_number
	time_addition = [dummy_time_list[-1] + datetime.timedelta(days = (n+1)) for n in range(adder_number)]
	time_subtraction = [dummy_time_list[0] - datetime.timedelta(days = (n+1)) for n in range(adder_number)]
	dummy_time_list = time_subtraction+dummy_time_list+time_addition
	return (dummy_time_list,dummy_pos_list)


class FloatBase(object):
	"""This is the base class of all floats"""

	def __init__(self,match,sources,**kwds):
		self.floatname = self.name_parser(match)
		self.gps = self.gps_parser(self.floatname)
		self.initialize_clock()
		self.gps.clock = self.clock
		self.depth = self.depth_parser(self.floatname)
		self.stream = self.stream_parser()

	def stream_parser(self):	# the weddell floats recorded no depth information
		return Stream([],[],self.clock)

	def obs_dates(self):
		return np.sort(np.unique(self.toa.date+self.gps.date))

	def percent_obs(self):
		dates = self.obs_dates()
		return float(len(dates))/(max(dates)-min(dates)).days

	def return_data(self):
		toa = self.toa.return_data()
		if toa:
			toa = toa[0]
		return (self.gps.return_data(),toa,self.depth.return_data(),self.stream.return_data(),self.gps.return_interp())

	def return_pos(self):
		pos_index = self.pos_date.index(self.clock.date)
		pos = self.pos[pos_index]
		return pos

	def plot(self,urlat,lllat,urlon,lllon,lon_0=0):
		m = Basemap.auto_map(urlat,lllat,urlon,lllon,lon_0)
		lat,lon = list(zip(*[(_.lat.decimal_degree,_.lon.decimal_degree) for _ in self.pos]))
		if lon_0!=0:
			lon = np.array(lon)
			lon[lon<0]= lon[lon<0]+360
			lon = lon.tolist()
		m.plot(lon,lat,'k',latlon=True)
		# lat,lon = zip(*[(_.lat.decimal_degree,_.lon.decimal_degree) for _ in [self.pos[0],self.pos[-1]]])
		if lon_0!=0:
			lon = np.array(lon)
			lon[lon<0]= lon[lon<0]+360
			lon = lon.tolist()
		# m.plot((lon[0],lon[-1]),(lat[0],lat[-1]),'y*',latlon=True,markersize=10)
		return m

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
		date = [ref_date + datetime.timedelta(days=int(_)) \
		for _ in np.array(date_list)-14536]
		return date

class DIMESFloat(FloatToken):
	def __init__(self,match,sources,**kwds):
		super(DIMESFloat,self).__init__(match,sources,**kwds)
		self.type = 'DIMES'
		self.load_trj_file(match)

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
		self.trj_df = trj_df

		dist_error_list,toa_error_list,dist_list,soso_list,date_return_list,pos_dist_list = self.toa.calculate_error_list(self.trj_pos,self.trj_date)

		toa,soso = list(zip(*flat_list(np.array(self.toa.obs)[[x in date_return_list for x in  self.toa.date]])))
		toa_error_array = np.array(toa_error_list) 
		toa_array = np.array(toa)
		z_threshhold = 2
		for soso_token in list(set(soso)):
			mask = [x==soso_token for x in soso]
			mask = np.array(mask)
			mask[mask==True] = abs(stats.zscore(toa_error_array[mask]))<z_threshhold
			mask = mask.tolist()
			toa_array[mask] = toa_array[mask] - np.nanmean(toa_error_array[mask])
			toa_error_array[mask] = toa_error_array[mask] - np.nanmean(toa_error_array[mask])
		mask = abs(toa_error_array)<20
		toa_error_array = toa_error_array[mask]
		soso = np.array(soso)[mask].tolist()
		toa = toa_array[mask].tolist()
		date_return_list = np.array(date_return_list)[mask].tolist()
		# soso_list = 

		self.toa.abs_error = toa_error_array.tolist()
		

		toa_list = []
		date_list = []
		for date in np.unique(date_return_list):
			date_list.append(date)
			idx = np.where(np.array(date_return_list)==date)[0]
			dummy_toa = tuple(np.array(toa)[idx].tolist())
			dummy_soso = tuple(np.array(soso)[idx].tolist())
			toa_list.append(list(zip(dummy_toa,dummy_soso)))
		assert len(toa_list)<=len(self.toa.obs)
		assert len(date_list)<=len(self.toa.date)
		self.toa.obs = toa_list
		self.toa.date = date_list


	def depth_parser(self,dummy):	# the dimes floats recorded no depth information
		print('I am parsing depth')
		return Depth([],[],self.clock)

	def name_parser(self,match):
		name = match.split('/')[-1]
		name.split('.')[0]
		name = ''.join([i for i in name if i.isdigit()])
		return int(name)

	def traj_plot(self):
		plt.subplot(2,1,1)
		lllon = -125
		urlon = -20
		lllat = -70
		urlat = -40
		m = self.plot(urlat,lllat,urlon,lllon)
		m.plot(self.trj_df['Lon'].tolist(),self.trj_df['Lat'].tolist(),latlon=True)
		plt.subplot(2,1,2)

		dist_error_list,toa_error_list,dist_list,soso_list,date_return_list,pos_dist_list = self.toa.calculate_error_list(self.pos,self.pos_date)
		soso_misfit_dict = {}
		for soso_token in np.unique(soso_list):
			soso_misfit_dict[soso_token]=[]

		for k,date in enumerate(self.pos_date):
			if date in date_return_list:
				idxs = np.where([x==date for x in date_return_list])[0]
				for idx in idxs:
					soso_misfit_dict[soso_list[idx]].append((k,toa_error_list[idx]))
			else:
				continue
		for soso_token in soso_misfit_dict.keys():
			x,y = list(zip(*soso_misfit_dict[soso_token]))
			plt.plot(x,y,label=soso_token)
		plt.legend()



class DIMESFloatBase(DIMESFloat):
	def __init__(self,match,sources,**kwds):
		super(DIMESFloatBase,self).__init__(match,sources,**kwds)
		
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
		return GPS(pos,time,gps_interp=False)

class WeddellFloat(FloatToken):
	def __init__(self,match,sources,**kwds):
		super(WeddellFloat,self).__init__(match,sources,**kwds)
		self.type = 'Weddell'
	def initialize_clock(self):
		self.clock = Clock(self.floatname,0,self.gps.date[0],self.gps.date[-1],drift=0)

	def depth_parser(self,dummy):	# the weddell floats recorded no depth information
		return Depth([],[],self.clock)

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

	def traj_plot(self):
		lllon = -60
		urlon = 35
		lllat = -75
		urlat = -50
		self.plot(urlat,lllat,urlon,lllon)

	def uncertainty_ellipse(self,date):
		date_idx = self.pos_date.index(date)
		P_ = self.P[date_idx]
		pos = self.pos[date_idx]
		C = P_[[0,0,2,2],[0,2,0,2]].reshape([2,2])
		w,v = np.linalg.eig(C)
		angle = np.degrees(np.arctan(v[1,np.argmax(w)]/v[0,np.argmax(w)]))
		A = 2*max(w)*np.sqrt(9.210)/abs(degree_dist*np.cos(np.deg2rad(pos.lat.decimal_degree)))
		B = 2*min(w)*np.sqrt(9.210)/degree_dist
		return A,B,angle

	def ellipse_plot(self,sources,P):
		lat,lon = list(zip(*[(_.lat.decimal_degree,_.lon.decimal_degree) for _ in self.pos]))
		lllon = min(lon)-2
		urlon = max(lon)+2
		lllat = min(lat)-2
		urlat = max(lat)+2


		for k,idx in enumerate(np.where([len(_)>2 for _ in self.toa.obs])[0][::5]):
			print(str(self.floatname)+'_toa_'+str(k))
			fig, ax = plt.subplots()
			m = self.plot(urlat,lllat,urlon,lllon)
			date = self.toa.date[idx]
			pos_idx = self.pos_date.index(date)
			A,B,angle = self.uncertainty_ellipse(date)
			m.ellipse(lon[pos_idx],lat[pos_idx],A,B,60,m,phi=angle,line=False,ax=ax)
			m.plot(lon[pos_idx],lat[pos_idx],latlon=True,marker='X',color='lime',markersize=12,zorder=10)
			m.plot(lon[0],lat[0],latlon=True,marker='^',color='m',markersize=12)
			m.plot(lon[-1],lat[-1],latlon=True,marker='s',color='m',markersize=12)
			m.plot(lon[0],lat[0],latlon=True,color='k',markersize=12)
			self.clock.set_date(date) 
			sources.set_date(date)
			# colorlist = ['m','b','g','y']
			for data in self.toa.return_data()[0]:
				# color = colorlist.pop()
				(toa,source) = data
				print(source.ID)
				detrend_toa = source.clock.detrend_offset(toa)
				dist = source.dist_from_toa(detrend_toa)
				print(source.position)
				print(dist)
				source_lat = source.position.lat.decimal_degree
				source_lon = source.position.lon.decimal_degree
				m.scatter(source_lon,source_lat,latlon=True,s=20,color='r')
				coords = m.ellipse(source_lon,source_lat,dist/(np.cos(np.deg2rad(source_lat))*degree_dist),dist/degree_dist,50,m)
				lons,lats = list(zip(*coords))
				plt.annotate(source.ID,m(source_lon+0.5,source_lat-0.3))
				m.plot(lons,lats,color='r',latlon=True)
			plt.savefig(str(self.floatname)+'_toa_'+str(k))
			plt.close()

class AllFloats(object):
	def __init__(self,type=None,**kwds):
		self.sources = SourceArray()
		if type=='Weddell':
			self.sources.set_speed(1.5)
		self.list = []
		base_folder = file_handler.store_file('')
		sources_dict = {
		'Weddell':[(base_folder+'Data/',WeddellFloat,'*.itm')],
		'DIMES':[(base_folder+'DIMES/',DIMESFloatBase,'rf*toa.mat')],
		}
		float_sources = sources_dict[type]
		for dir_,float_type,ext_ in float_sources:
			base_matches = find_files(dir_,ext_)
			for match in base_matches:
				print(match)
				float_ = float_type(match,self.sources,interp_depth=True)
				self.list.append(float_)
		self.combine_classes()
		self.assign_soso_drift_errors()

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
				base.depth.obs += dummy.depth.obs
				base.depth.date += dummy.depth.date
			base.date = min(base.gps.date)
			base.toa.obs = [x for _,x in sorted(list(zip(base.toa.date,base.toa.obs)), key=lambda x: x[0])]
			base.gps.obs = [x for _,x in sorted(list(zip(base.gps.date,base.gps.obs)), key=lambda x: x[0])]
			base.depth.obs = [x for _,x in sorted(list(zip(base.depth.date,base.depth.obs)), key=lambda x: x[0])]
			base.toa.date = sorted(base.toa.date)
			base.gps.date = sorted(base.gps.date)
			base.depth.date = sorted(base.depth.date)

			new_list.append(base)
		self.list = new_list

	def assign_soso_drift_errors(self):
		error_list = []
		for float_ in self.list:
			gps_date = \
			float_.gps.date+\
			[_ - datetime.timedelta(days=1) for _ in float_.gps.date]+\
			[_ + datetime.timedelta(days=1) for _ in float_.gps.date]+\
			[_ + datetime.timedelta(days=2) for _ in float_.gps.date]+\
			[_ - datetime.timedelta(days=2) for _ in float_.gps.date]

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
					toa_measured_detrend = float_.clock.detrend_offset(toa_measured)
					dist = geopy.distance.GreatCircleDistance(soso.position,loc).km # this is in km
					days = (intersection_dummy-soso.clock.initial_date).days
					error_list.append((soso.ID,float_.floatname,dist,toa_measured_detrend,days,intersection_dummy))
		df = pd.DataFrame(error_list,columns=['SOSO','Float','Dist','TOA','Days','Date'])
		for _,soso in self.sources.array.items():
			if soso.mission=='DIMES':
				continue
				print('I am passing because this is a dimes sound source')
			print(soso)
			print(soso.clock.initial_offset)
			print(soso.clock.drift)
			df_holder = df[df.SOSO==soso.ID]
			if not df_holder.empty:
				print('the dataframe is not empty')
				soso.error_dataframe = df_holder
				soso.calculate_offset_and_sound_speed()
			else:
				print('the dataframe is empty')
				soso.error_dataframe = pd.DataFrame([])

	def plot_soso_errors_by_date(self):
		for _,soso in self.sources.array.items():
			if not soso.error_dataframe.empty:
				soso.plot_error_dataframe('Date')
				plt.show()

	def plot_soso_errors_by_dist(self):
		for _,soso in self.sources.array.items():
			if not soso.error_dataframe.empty:
				soso.plot_error_dataframe('Dist')
				print(soso.speed_of_sound)
				plt.show()

	def plot_gps_error_hist(self):
		kalman_error,smooth_error=self.all_gps_error_list()
		kalman_error = np.array(kalman_error)
		smooth_error = np.array(smooth_error)
		kalman_error = kalman_error[kalman_error<0.5]
		smooth_error = smooth_error[smooth_error<0.5]
		bins=np.histogram(np.hstack((smooth_error,kalman_error)), bins=50)[1]
		plt.figure()
		plt.hist(smooth_error,bins,label = 'Kalman Smoother',alpha=0.2)
		plt.hist(kalman_error,bins,label = 'Kalman Filter',alpha=0.2)
		plt.gca().set_yscale("log")
		plt.legend(prop={'size': 10})
		plt.title('Histogram of GPS Error')
		plt.savefig('gps_error')
		plt.close()

	def all_gps_error_list(self):
		kalman_list = []
		smoother_list = []
		for label,dummy_list in [('kalman',kalman_list),('smoother',smoother_list)]:
			for _ in self.list:
				dummy_1 = [item for sublist in _.gps.dx_error[label] for item in sublist]
				dummy_2 = [item for sublist in _.gps.dy_error[label] for item in sublist]
				dummy_list += [np.sqrt(_[0]**2+_[1]**2) for _ in list(zip(dummy_1,dummy_2))]
		return (kalman_list,smoother_list)

	def all_gps_date_list(self):
		dummy_list = []
		for _ in self.list:
			dummy_list+=_.gps.date
		return dummy_list

	def all_toa_error_list(self,error_type):
		dummy_list = []
		for _ in self.sources.array.items():
			dummy_list+=_[1].error[error_type]
		dummy_list = [item for sublist in dummy_list for item in sublist]
		return dummy_list

	def all_DIMES_error_list(self):
		dummy_list = []
		for _ in self.list:
			dummy_list+=_.toa.abs_error
		return dummy_list

	def all_DIMES_artoa_speed_list(self):
		dummy_list = []
		for _ in self.list:
			df = _.trj_df.drop_duplicates(subset=['Lat','Lon'])
			pos_list = [KalmanPoint(x_[0],x_[1]) for x_ in list(zip(df['Lat'],df['Lon']))] 
			dummy_list+=(np.array([x.magnitude for x in np.diff(pos_list)])/np.array([x.days for x in df.diff()['Date'][1:]])).tolist()
		return dummy_list

	def all_DIMES_kalman_speed_list(self):
		dummy_list = []
		for _ in self.list:
			mask = np.isin(_.pos_date,_.trj_date)
			dummy_pos = np.array(_.pos)[mask]
			dummy_pos_date = np.array(_.pos_date)[mask]
			dummy_list += (np.array([x.magnitude for x in np.diff(dummy_pos)])/np.array([x.days for x in np.diff(dummy_pos_date)])).tolist()

	def all_speed_list(self):
		dummy_list = []
		for _ in self.list:		
			dummy_list += (np.array([x.magnitude for x in np.diff(_.pos)])/np.array([x.days for x in np.diff(_.pos_date)])).tolist()
		return dummy_list

	def DIMES_speed_hist(self):
		kalman_speed = self.all_speed_list()
		dimes_speed = self.all_DIMES_artoa_speed_list()

		bins = np.linspace(0,41,100)
		plt.hist(kalman_speed,bins=bins,label='Kalman',alpha=0.2)
		plt.hist(dimes_speed,bins=bins,label='DIMES',alpha=0.2)
		plt.legend()
		plt.savefig('dimes_vel_stats')
		plt.close()

	def all_toa_date_list(self):
		dummy_list = []
		for _ in self.list:
			dummy_list+=_.toa.date
		return dummy_list

	def all_toa_number(self):
		dummy_list = []
		for _ in self.list:
			total_len = (_.gps.date[-7]-_.gps.date[6]).days
			for dummy_toa in _.toa.obs:
				dummy_list+=[len(dummy_toa)]
			nothing_heard = total_len-len(_.toa.obs)
			dummy_list+=[0]*nothing_heard
		return dummy_list

	def all_toa_list(self):
		dummy_list = []
		for _ in self.list:
			for dummy_toa in _.toa.obs:
				dummy_list+=dummy_toa
		return dummy_list

	def all_traj_plot(self,title):
		for all_dummy in self.list:
			try:
				all_dummy.traj_plot()
			except AttributeError:
				continue
		plt.savefig(title)
		plt.close

	def weddell_plot(self):
		plt.figure()
		lllon = -60
		urlon = 35
		lllat = -75
		urlat = -50
		m = Basemap.auto_map(urlat,lllat,urlon,lllon,0,spacing=15,aspect=False)
		first_lat_list= []
		first_lon_list = []
		last_lat_list = []
		last_lon_list = []

		for k,all_dummy in enumerate(self.list):	
			lat,lon = list(zip(*[(_.lat.decimal_degree,_.lon.decimal_degree) for _ in all_dummy.pos]))
			first_lat_list.append(lat[0])
			first_lon_list.append(lon[0])
			last_lat_list.append(lat[-1])
			last_lon_list.append(lon[-1])

			A,B,angle = list(zip(*[all_dummy.uncertainty_ellipse(_) for _ in all_dummy.pos_date]))
			m.scatter(lon,lat,s=1.5,c=np.array(A)*11.3,latlon=True,alpha=0.2)
		lat_list = []
		lon_list = []
		for _,source in self.sources.array.items():
			if source.mission=='Weddell':
				lat_list.append(source.position.latitude)
				lon_list.append(source.position.longitude)
		plt.colorbar(label='Uncertainty (km)')
		m.scatter(first_lon_list,first_lat_list,s=13,c='m',latlon=True,marker='^')
		m.scatter(last_lon_list,last_lat_list,s=13,c='m',latlon=True,marker='s')
		m.scatter(lon_list,lat_list,s=13,c='r',latlon=True)
		plt.savefig('weddell_plot')
		plt.close()
		plt.figure()
		dummy_list = self.all_toa_error_list('smoother')
		plt.hist(dummy_list,bins=75)
		plt.xlim([-20,20])
		plt.xlabel('Misfit (seconds)', fontsize=18)
		plt.savefig('Weddell_stats')

	def all_dimes_error_hist(self):
		from scipy import stats		
		kalman_speed = self.all_speed_list()
		dimes_speed = self.all_DIMES_artoa_speed_list()
		trj_list = []
		trj_error_list = []
		kalman_list = []
		kalman_error_list = []
		kalman_date_list = []
		float_list = []
		diff_list = []
		long_diff_list = []
		std_list = []
		mean_list = []
		percent_list = []
		date_diff_list = []
		toa_number = []
		two_d_hist_diff = []
		for k,all_dummy in enumerate(self.list):
			print(all_dummy.floatname)
			try:
				trj_date = np.unique(all_dummy.trj_date)
				dummy_error,toa_error_list,dist_list,soso_list,date_return_list,pos_dist_list = all_dummy.toa.calculate_error_list(all_dummy.pos,all_dummy.pos_date)
				kalman_date_list+=date_return_list
				float_list+=[k]*len(date_return_list)
				kalman_error_list+=dummy_error
				mean_list.append(np.mean(np.abs(dummy_error)))
				std_list.append(np.std(dummy_error))
				percent_list.append(all_dummy.percent_obs())
				trj_error,toa_error_list,dist_list,soso_list,date_return_list,trj_dist_list = all_dummy.toa.calculate_error_list(all_dummy.trj_pos,trj_date)
				trj_error_list+=trj_error
				mask_trj = np.isin(trj_date,all_dummy.pos_date)
				pos = np.array(all_dummy.trj_pos)[mask_trj]
				mask_smooth = np.isin(all_dummy.pos_date,trj_date)
				smooth_pos = np.array(all_dummy.pos)[mask_smooth]
				diff = [(x[0]-x[1]).magnitude for x in list(zip(pos,smooth_pos))]
				diff_list+=diff
				two_d_hist_diff += np.array(diff)[np.isin(trj_date,all_dummy.toa.date)].tolist()
				mask = np.isin(all_dummy.toa.date,trj_date)
				date_diff_list += ([1]+[x.days for x in np.diff(np.array(all_dummy.toa.date)[mask])])
				toa_number += [len(x) for x in all_dummy.toa.obs]

			except:
				continue

		N = 1
		label_pad = 0
		x_date = np.convolve(date_diff_list,np.ones(N)/N, mode = 'valid')
		x_toa = np.convolve(toa_number,np.ones(N)/N, mode = 'valid')
		y = np.convolve(two_d_hist_diff,np.ones(N)/N, mode = 'valid')
		plt_date_diff = []

		bins = np.linspace(min(x_date),max(x_date),14)
		bin_means, bin_edges, binnumber = stats.binned_statistic(x_date,
			y, statistic='median', bins=bins)
		bin_std, bin_edges, binnumber = stats.binned_statistic(x_date,
			y, statistic='median', bins=bins)
		bin_width = (bin_edges[1] - bin_edges[0])
		bin_centers = bin_edges[1:] - bin_width/2
		mask = ~np.isnan(bin_means)

		# for i,date_diff_token in enumerate(np.unique(plt_date_diff_list)):
		# 	mask = (np.array(plt_date_diff_list)==date_diff_token)
		# 	plt_date_diff.append(plt_two_d_hist_diff[mask].mean())
		slope, intercept, r_value, p_value, std_err = stats.linregress(x_date, y)
		print('slope of date difference to uncertainty')
		print(slope)
		plt.subplot(3,1,1)
		plt.errorbar(bin_centers[mask], bin_means[mask], bin_std[mask],ecolor='r')
		plt.scatter(bin_centers[mask], bin_means[mask],color='b')
		plt.ylim([0,900])
		plt.xlabel('Time Without Positioning (days)',labelpad=label_pad)
		plt.annotate('a', xy = (0.8,0.75),xycoords='axes fraction',zorder=10,size=14,bbox=dict(boxstyle="round", fc="0.8"),)

		plt_toa = []
		for j,toa_number_token in enumerate(np.unique(toa_number)):
			mask = (np.array(toa_number)==toa_number_token)
			plt_toa.append(np.array(two_d_hist_diff)[mask].mean())
		slope, intercept, r_value, p_value, std_err = stats.linregress(x_toa, y)
		print('slope of toa heard to uncertainty')
		print(slope)
		plt.subplot(3,1,2)
		plt.scatter(np.unique(toa_number),plt_toa,alpha=0.2)
		plt.plot([min(toa_number),max(toa_number)],[intercept+slope*min(toa_number),intercept+slope*max(toa_number)])
		plt.ylim([0,100])
		plt.xlim([0.9,6.1])
		plt.xlabel('Sources Heard',labelpad=label_pad)
		plt.ylabel('Trajectory Difference (km)')
		plt.annotate('b', xy = (0.8,0.75),xycoords='axes fraction',zorder=10,size=14,bbox=dict(boxstyle="round", fc="0.8"),)
		
		plt.subplot(3,1,3)

		bins = np.linspace(min(percent_list)*100,max(percent_list)*100,7)
		bin_means, bin_edges, binnumber = stats.binned_statistic(np.array(percent_list)*100,
			mean_list, statistic='median', bins=bins)
		bin_std, bin_edges, binnumber = stats.binned_statistic(np.array(percent_list)*100,
			mean_list, statistic='median', bins=bins)
		bin_width = (bin_edges[1] - bin_edges[0])
		bin_centers = bin_edges[1:] - bin_width/2
		plt.errorbar(bin_centers, bin_means, bin_std,ecolor='r')
		plt.scatter(bin_centers, bin_means,color='b')
		plt.xlabel('Days Sources Heard (%)',labelpad=label_pad)
		plt.annotate('c', xy = (0.8,0.75),xycoords='axes fraction',zorder=10,size=14,bbox=dict(boxstyle="round", fc="0.8"),)



		plt.subplots_adjust(hspace=0.4)
		plt.savefig('uncertainty_increase')
		plt.close()

		from scipy.stats import norm
		plt.figure()
		plt.subplot(3,1,1)
		bins = np.linspace(-40,40,200)
		trj_n,dummy,dummy = plt.hist(trj_error_list,bins=bins,color='b',label='ARTOA',alpha=0.3)
		kalman_n,dummy,dummy = plt.hist(kalman_error_list,bins=bins,color='g',label='Smoother',alpha=0.3)
		x = np.arange(-30,30,0.01)
		trj_plot_list = np.array(trj_error_list)[(np.array(trj_error_list)<30)&(np.array(trj_error_list)>-30)]
		trj_plot_list = norm.pdf(x,trj_plot_list.mean(),trj_plot_list.std())
		trj_plot_list = trj_n.max()/trj_plot_list.max()*trj_plot_list
		plt.plot(x,trj_plot_list,'b',alpha=0.5,linewidth=3,linestyle='dashed')
		kalman_plot_list = np.array(kalman_error_list)[(np.array(kalman_error_list)<30)&(np.array(kalman_error_list)>-30)]
		kalman_plot_list = norm.pdf(x,kalman_plot_list.mean(),kalman_plot_list.std())
		kalman_plot_list = kalman_n.max()/kalman_plot_list.max()*kalman_plot_list

		plt.plot(x,kalman_plot_list,'g',alpha=0.5,linewidth=3,linestyle='dashed')
		print('mean of the trj_error_list')
		print(np.mean(trj_error_list))
		print('std of the trj_error_list')
		print(np.std(trj_error_list))

		print('mean of the kalman')
		print(np.mean(kalman_error_list))
		print('std of the kalman')
		print(np.std(kalman_error_list))


		plt.xlabel('Misfit (seconds)', fontsize=14)
		plt.xlim([-30,30])
		plt.legend(prop={'size': 10})
		plt.annotate('a', xy = (0.2,0.7),xycoords='axes fraction',zorder=10,size=12,bbox=dict(boxstyle="round", fc="0.8"),)
		plt.subplot(3,1,2)
		bins = np.linspace(0,200,40)
		plt.xlim([0,200])
		plt.hist(diff_list,bins=bins)
		plt.xlabel('Trajectory Difference (km)', fontsize=14)
		plt.annotate('b', xy = (0.2,0.7),xycoords='axes fraction',zorder=10,size=12,bbox=dict(boxstyle="round", fc="0.8"),)		
		print('the mean misfit is')
		print(np.mean(diff_list))
		print('the standard deviation of the misfit is ')
		print(np.std(diff_list))
		counts,bins = np.histogram(diff_list,bins)
		print ('misfit peaked at ')
		print(bins[counts.tolist().index(counts.max())])


		plt.subplot(3,1,3)
		bins = np.linspace(0,41,30)
		plt.hist(kalman_speed,bins=bins,color='g',label='Kalman',alpha=0.3)
		plt.hist(dimes_speed,bins=bins,color='b',label='DIMES',alpha=0.3)
		plt.annotate('c', xy = (0.2,0.7),xycoords='axes fraction',zorder=10,size=12,bbox=dict(boxstyle="round", fc="0.8"),)		
		plt.xlim([0,20])
		plt.xlabel('Speed (km $day^{-1}$)', fontsize=14)

		print('mean artoa speed')
		print(np.mean(dimes_speed))

		counts,bins = np.histogram(dimes_speed,bins)
		print ('artoa speed peaked at ')
		print(bins[counts.tolist().index(counts.max())])

		print('median artoa speed')
		print(np.median(dimes_speed))


		print('mean kalman speed')
		print(np.mean(kalman_speed))

		counts,bins = np.histogram(kalman_speed,bins)
		print ('kalman speed peaked at ')
		print(bins[counts.tolist().index(counts.max())])

		print('median kalman speed')
		print(np.median(kalman_speed))


		plt.tight_layout(pad=1.5)
		plt.savefig('all_dimes_error')
		plt.close()

	def dimes_all_plot(self):
		lon_0 =0 
		for k,all_dummy in enumerate(self.list):
			print(k)
			if all_dummy.floatname ==837:
				continue
			if int(all_dummy.floatname)>868:
				print(all_dummy.floatname)
				continue
			if k == 0:
				lllon = -115
				urlon = -40
				lllat = -70
				urlat = -45
				m = Basemap(projection='cea',llcrnrlat=lllat,urcrnrlat=urlat,\
					llcrnrlon=lllon,urcrnrlon=urlon,resolution='l',lon_0=lon_0,\
					fix_aspect=False)
				# m.drawmapboundary(fill_color='darkgray')
				m.fillcontinents(color='dimgray')
				m.drawmeridians(np.arange(0,360,10),labels=[1,1,1,1])
				m.drawparallels(np.arange(-90,90,10),labels=[1,1,1,1])

			try:
				pos = np.array(all_dummy.pos)[np.isin(all_dummy.pos_date,all_dummy.trj_df['Date'].tolist())]
				lat,lon = list(zip(*[(_.latitude,_.longitude) for _ in pos]))
				m.plot(lon,lat,color='g',alpha=0.2,latlon=True)
				m.plot(all_dummy.trj_df['Lon'].tolist(),all_dummy.trj_df['Lat'].tolist(),color='b',alpha=0.2,latlon=True)
			except AttributeError:
				continue
		lats = []
		lons = []
		for _ in self.sources.array.items():
			if _[1].mission == 'DIMES':
				lat = _[1].position.latitude
				lon = _[1].position.longitude
				lats.append(lat)
				lons.append(lon)
		x,y = m(lons,lats)
		m.scatter(x,y,color='r',zorder=10)
		plt.savefig('all_dimes_plot')