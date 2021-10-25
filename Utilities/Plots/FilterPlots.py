from KalmanSmoother.Utilities.Filters import LeastSquares,Kalman,Smoother,ObsHolder
from KalmanSmoother.Utilities.Observations import SourceArray
from KalmanSmoother.Utilities.Floats import DIMESAllFloats,WeddellAllFloats
from GeneralUtilities.Filepath.instance import FilePathHandler
from GeneralUtilities.Plot.Cartopy.eulerian_plot import BaseCartopy
from GeneralUtilities.Data.depth.depth_utilities import ETopo1Depth
import cartopy.crs as ccrs
import matplotlib.pyplot as plt



class TrajDictCartopy(BaseCartopy):
	def __init__(self,lons,lats,*args,pad=1,**kwargs):
		super().__init__(*args,**kwargs)
		llcrnrlon=(min(lons)-pad)
		llcrnrlat=(min(lats)-pad)
		urcrnrlon=(max(lons)+pad)
		urcrnrlat=(max(lats)+pad)
		self.ax.set_extent([llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat], crs=ccrs.PlateCarree())
		self.finish_map()


def optimal_weddell():
	process_position_noise = 3
	process_vel_noise = 1.6
	interp_noise = 336
	depth_noise = 160
	stream_noise = .024
	gps_noise = .1
	max_vel_uncert=20
	max_vel = 20
	max_x_diff = 20	
	toa_noise = 28
	
	# depth = Depth()
	# depth.guassian_smooth(sigma=4)
	# stream = Stream()
	# depth.guassian_smooth(sigma=2)
	# depth_flag = True
	# stream_flag = True
	# lin_between_obs=True
	all_floats = WeddellAllFloats()
	for idx,dummy in enumerate(all_floats.list[:5]):
		print(idx)
		obs_holder = ObsHolder(dummy)
		dummy.toa.set_observational_uncertainty(25)
		dummy.depth.set_observational_uncertainty(1500)
		dummy.stream.set_observational_uncertainty(100)
		dummy.gps.interp_uncertainty = 336
		obs_holder = ObsHolder(dummy)
		dummy.toa.set_observational_uncertainty(10)
		smooth =Smoother(dummy,all_floats.sources,obs_holder,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)
	for holder in all_floats.list:
		print(holder.floatname)
		lons,lats = zip(*[(x.longitude,x.latitude) for x in holder.pos])

		lon_grid,lat_grid,ax = TrajDictCartopy(lons,lats,pad=1).get_map()
		ax.scatter(lons,lats,label='Kalman',alpha=0.3)
		plt.savefig(str(holder.floatname))
		plt.close()


	from scipy import stats		
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
		dummy_error,toa_error_list,dist_list,soso_list,date_return_list,pos_dist_list = all_dummy.toa.calculate_error_list(all_dummy.pos,all_dummy.pos_date)
		toa_error_list = [abs(x) for x in toa_error_list]
		date_dict = ([1]+[x.days for x in np.diff(np.array(np.unique(date_return_list)))])
		for date,date_diff in zip(np.unique(date_return_list),date_dict):
			num = (np.array(date_return_list)==date).sum()
			toa_number += [num]*num
			date_diff_list += [date_diff]*num



		kalman_date_list+=date_return_list
		float_list+=[k]*len(date_return_list)
		kalman_error_list+=toa_error_list
		mean_list.append(np.mean(np.abs(toa_error_list)))
		std_list.append(np.std(toa_error_list))
		percent_list.append(all_dummy.percent_obs())

	N = 1
	label_pad = 0
	x_date = np.convolve(date_diff_list,np.ones(N)/N, mode = 'valid')
	x_toa = np.convolve(toa_number,np.ones(N)/N, mode = 'valid')
	y = np.convolve(kalman_error_list,np.ones(N)/N, mode = 'valid')
	plt_date_diff = []

	bins = np.linspace(min(x_date),max(x_date),7)
	bin_means, bin_edges, binnumber = stats.binned_statistic(x_date,
		y, statistic='median', bins=bins)
	bin_std, bin_edges, binnumber = stats.binned_statistic(x_date,
		y, statistic='median', bins=bins)
	bin_width = (bin_edges[1] - bin_edges[0])
	bin_centers = bin_edges[1:] - bin_width/2
	mask = ~np.isnan(bin_means)
	slope, intercept, r_value, p_value, std_err = stats.linregress(x_date, y)
	print('slope of date difference to uncertainty')
	print(slope)
	plt.subplot(3,1,1)
	plt.errorbar(bin_centers[mask], bin_means[mask], bin_std[mask],ecolor='r')
	plt.scatter(bin_centers[mask], bin_means[mask],color='b')
	plt.ylim([0,20])
	plt.xlabel('Time Without Positioning (days)',labelpad=label_pad)
	plt.annotate('a', xy = (0.8,0.75),xycoords='axes fraction',zorder=10,size=14,bbox=dict(boxstyle="round", fc="0.8"),)

	plt_toa = []
	for j,toa_number_token in enumerate(np.unique(toa_number)):
		mask = (np.array(toa_number)==toa_number_token)
		plt_toa.append(np.array(kalman_error_list)[mask].mean())
	slope, intercept, r_value, p_value, std_err = stats.linregress(x_toa, y)
	print('slope of toa heard to uncertainty')
	print(slope)
	plt.subplot(3,1,2)
	plt.scatter(np.unique(toa_number),plt_toa,alpha=0.2)
	plt.plot([min(toa_number),max(toa_number)],[intercept+slope*min(toa_number),intercept+slope*max(toa_number)])
	plt.ylim([0,13])
	plt.xlim([0.9,6.1])
	plt.xlabel('Sources Heard',labelpad=label_pad)
	plt.ylabel('TOA Misfit (s)')
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
	plt.savefig('weddell_uncertainty_increase')
	plt.close()

	dummy_list = self.all_toa_error_list('smoother')
	plt.subplot(2,1,1)
	plt.hist(dummy_list,bins=150)
	plt.xlim(-20,20)
	plt.xlabel('Misfit (s)')
	plt.annotate('a', xy = (0.8,0.75),xycoords='axes fraction',zorder=10,size=18,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.subplot(2,1,2)

	plt.annotate('b', xy = (0.8,0.75),xycoords='axes fraction',zorder=10,size=18,bbox=dict(boxstyle="round", fc="0.8"),)
	kalman_speed = self.all_speed_list()
	plt.hist(kalman_speed,bins=150)
	plt.xlim([-0.1,20])
	plt.xlabel('Speed $km\ day^{-1}$)')
	plt.subplots_adjust(hspace=0.4)
	plt.savefig('weddell_misfit_hist')
	plt.close()



def optimal_dimes():
	process_position_noise = 2
	process_vel_noise = 2

	all_floats = DIMESAllFloats()
	for idx,dummy in enumerate(all_floats.list):
		print(idx)
		process_position_noise = 3
		process_vel_noise = 1.6
		dummy.toa.set_observational_uncertainty(25)
		dummy.depth.set_observational_uncertainty(1500)
		dummy.stream.set_observational_uncertainty(100)
		obs_holder = ObsHolder(dummy)
		smooth =Smoother(dummy,all_floats.sources,obs_holder,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)

	for holder in all_floats.list:
		print(holder.floatname)
		lons,lats = zip(*[(x.longitude,x.latitude) for x in holder.pos])
		trj_lons,trj_lats = zip(*[(x.longitude,x.latitude) for x in holder.trj_pos])

		lon_grid,lat_grid,ax = TrajDictCartopy(lons+trj_lons,lats+trj_lats,pad=1).get_map()
		ax.scatter(lons,lats,label='Kalman',alpha=0.3,zorder=10)
		ax.scatter(trj_lons,trj_lats,label='ARTOA',alpha=0.3,zorder=10)
		plt.savefig(str(holder.floatname))
		plt.close()

	all_floats.DIMES_speed_hist()

def particle_release():
	def make_filename():
		error_folder = root_folder+'output/particle/'
		unique_filename = str(uuid.uuid4())
		return (error_folder+unique_filename)
	import uuid
	from kalman_smoother.code.acoustic_read import ArtificialFloats
	depth = Depth()
	depth.guassian_smooth(sigma=2)
	stream = Stream()
	depth_flag = False
	stream_flag = False
	lin_between_obs=False

	pos_list = []
	label_list = []
	artoa_error_list = []
	smoother_error_list = []
	sources = SourceArray()
	sources.set_speed(1.5)
	sources.set_drift(0)
	sources.set_offset(0)
	initial_loc = LatLon.LatLon(-64,-23.5)
	sources.set_location(initial_loc)
	gps_noise = .1
	max_vel_uncert=60
	max_vel = 30
	max_x_diff = 35

	for percent in [0.1,0.3,0.7]:
		all_floats = ArtificialFloats(9970,sources,percent)
		for idx,dummy in enumerate(all_floats.list):
			try:
				all_floats.sources.set_drift(0)
				all_floats.sources.set_offset(0)

				process_noise = (all_floats.var_x)*percent
				process_position_noise = process_noise
				process_vel_noise = process_noise
				interp_noise = 300
				all_floats.sources.set_speed(1.5)
				I = np.identity(4)
				max_vel_uncert= 30
				max_vel = 35
				max_x_diff = 50
				toa_noise = dummy.toa_noise

				ls = LeastSquares(dummy,all_floats.sources,depth,stream,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)
				smooth =Smoother(dummy,all_floats.sources,depth,stream,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)

				gps_number= len(dummy.gps.obs)
				smoother_error = (np.mean([_.magnitude for _ in np.array(dummy.pos[:-1])-np.array(dummy.exact_pos)]))
				kalman_error = (np.mean([_.magnitude for _ in np.array(dummy.kalman_pos[:-1])-np.array(dummy.exact_pos)]))
				ls_error = (np.mean([_.magnitude for _ in np.array(dummy.ls_pos[:-1])-np.array(dummy.exact_pos)]))
				toa_error = (dummy.toa_noise)
				toa_number = (dummy.toa_number)
				percent_list = (percent)
				filename= make_filename()
				np.save(filename,[gps_number,smoother_error,kalman_error,ls_error,toa_error,toa_number,percent_list])
			except:
				continue
			all_floats.sources.reset_error()


