from KalmanSmoother.Utilities.Filters import LeastSquares,Kalman,Smoother
from GeneralUtilities.Data.agva.agva_utilities import AVGAStream as Stream
from GeneralUtilities.Data.depth.depth_utilities import ETopo1Depth as Depth
from KalmanSmoother.Utilities.Observations import SourceArray
from KalmanSmoother.Utilities.Floats import AllFloats
from GeneralUtilities.Filepath.instance import FilePathHandler
from KalmanSmoother.Utilities.__init__ import ROOT_DIR

file_handler = FilePathHandler(ROOT_DIR,'FilterPlots')

def optimal_weddell():
	process_position_noise = 6
	process_vel_noise = 1.6
	interp_noise = 336
	depth_noise = 160
	stream_noise = .024
	gps_noise = .1
	max_vel_uncert=20
	max_vel = 20
	max_x_diff = 20	
	toa_noise = 28
	
	depth = Depth()
	depth.guassian_smooth(sigma=4)
	stream = Stream()
	depth.guassian_smooth(sigma=2)
	depth_flag = True
	stream_flag = True
	lin_between_obs=True
	all_floats = AllFloats(type='Weddell')
	for idx,dummy in enumerate(all_floats.list):
		print(idx)
		smooth =Smoother(dummy,all_floats.sources,depth,stream,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)
		dummy.traj_plot()
		plt.savefig('traj_'+str(dummy.floatname))
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
	process_position_noise = 5
	process_vel_noise = 2.4
	interp_noise = 500
	depth_noise = 240
	stream_noise = .024
	gps_noise = .1
	max_vel_uncert=20
	max_vel = 20
	max_x_diff = 20	
	
	depth = Depth()
	depth.guassian_smooth(sigma=4)
	stream = Stream()
	depth.guassian_smooth(sigma=2)
	depth_flag = True
	stream_flag = True
	lin_between_obs=True
	I = np.identity(4)
	all_floats = AllFloats(type='DIMES')

	all_dimes_error = all_floats.all_DIMES_error_list()
	bins = np.arange(-40,40.5,0.5)
	plt.hist(all_dimes_error,bins=bins)
	plt.savefig('all_dimes_toa_error')
	plt.close()
	for idx,dummy in enumerate(all_floats.list):
		print(idx)
		toa_noise = 1.2*np.std(dummy.toa.abs_error)
		smooth =Smoother(dummy,all_floats.sources,depth,stream,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)
		dummy.traj_plot()
		plt.savefig('traj_'+str(dummy.floatname))
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

def dimes_error_save():
	error_folder = root_folder+'output/error/'
	process_position_noise_base = 5
	process_vel_noise_base = 2
	interp_noise_base = 500
	depth_noise_base = 200
	stream_noise_base = .02
	gps_noise = .1
	max_vel_uncert=20
	max_vel = 20
	max_x_diff = 20	
	
	depth = Depth()
	depth.guassian_smooth(sigma=4)
	stream = Stream()
	depth.guassian_smooth(sigma=2)
	depth_flag = True
	stream_flag = True
	lin_between_obs=True
	I = np.identity(4)
	interp_noise = 0
	all_floats = AllFloats(type='DIMES')
	if root_folder == '/Users/pchamberlain/Projects/kalman_smoother/':
		multiplier_list =[0.8,1.0,1.2]
	else:
		multiplier_list = [1.2,0.8,1.0]
	def calculate(toa_noise_multiplier,process_position_noise,process_vel_noise,depth_noise,stream_noise,all_floats,depth,stream,Smoother):
		for idx,dummy in enumerate(all_floats.list):
			print(idx)
			toa_noise = np.std(dummy.toa.abs_error)*toa_noise_multiplier
			global toa_noise
			smooth =Smoother(dummy,all_floats.sources,depth,stream,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)
		misfit = all_floats.sources.return_misfit()/toa_noise
		all_floats.sources.reset_error()
		filename_token = error_folder+make_filename(toa_noise_multiplier,process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise)
		np.save(filename_token,misfit)

	for process_position_noise_multiplier in multiplier_list:
		process_position_noise = process_position_noise_multiplier*process_position_noise_base
		global process_position_noise
		for process_vel_noise_multiplier in multiplier_list:
			process_vel_noise = process_vel_noise_multiplier*process_vel_noise_base
			global process_vel_noise
			for depth_noise_multiplier in multiplier_list:
				depth_noise = depth_noise_multiplier*depth_noise_base
				global depth_noise
				for stream_noise_multiplier in multiplier_list:
					stream_noise = stream_noise_multiplier*stream_noise_base
					global stream_noise
					for toa_noise_multiplier in multiplier_list:
						filename_token = error_folder+make_filename(toa_noise_multiplier,process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise)+'.npy'
						try:
							np.load(filename_token)
						except IOError:
							np.save(filename_token,0)
							name_string = 'toa noise %d, process position noise %d, process vel noise %d, interp noise %d, depth noise %d, stream noise %d'%(toa_noise_multiplier,
								process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise)
							print(name_string+' was not found, calculating')
							calculate(toa_noise_multiplier,process_position_noise,process_vel_noise,depth_noise,stream_noise,all_floats,depth,stream,Smoother)


def weddell_error_save():
	error_folder = root_folder+'output/weddell_error/'
	process_position_noise_base = 5
	process_vel_noise_base = 2
	interp_noise_base = 280
	depth_noise_base = 200
	stream_noise_base = .02
	toa_noise_base = 35

	gps_noise = .1
	max_vel_uncert=20
	max_vel = 20
	max_x_diff = 20	
	
	depth = Depth()
	depth.guassian_smooth(sigma=4)
	stream = Stream()
	depth.guassian_smooth(sigma=2)
	depth_flag = True
	stream_flag = True
	lin_between_obs=True
	I = np.identity(4)
	all_floats = AllFloats(type='Weddell')
	if root_folder == '/Users/pchamberlain/Projects/kalman_smoother/':
		multiplier_list =[0.8,1.0,1.2]
	else:
		multiplier_list = [1.2,0.8,1.0]
	def calculate(toa_noise_multiplier,process_position_noise,process_vel_noise,depth_noise,stream_noise,interp_noise,all_floats,depth,stream,Smoother):
		for idx,dummy in enumerate(all_floats.list):
			print(idx)
			smooth =Smoother(dummy,all_floats.sources,depth,stream,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)
		misfit = all_floats.sources.return_misfit()/toa_noise
		all_floats.sources.reset_error()
		filename_token = error_folder+make_filename(toa_noise_multiplier,process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise)
		np.save(filename_token,misfit)

	for process_position_noise_multiplier in multiplier_list[::-1]:
		process_position_noise = process_position_noise_multiplier*process_position_noise_base
		global process_position_noise
		for process_vel_noise_multiplier in multiplier_list[::-1]:
			process_vel_noise = process_vel_noise_multiplier*process_vel_noise_base
			global process_vel_noise
			for depth_noise_multiplier in multiplier_list[::-1]:
				depth_noise = depth_noise_multiplier*depth_noise_base
				global depth_noise
				for stream_noise_multiplier in multiplier_list[::-1]:
					stream_noise = stream_noise_multiplier*stream_noise_base
					global stream_noise
					for toa_noise_multiplier in multiplier_list[::-1]:
						toa_noise = toa_noise_multiplier*toa_noise_base
						global toa_noise
						global toa_noise_multiplier
						for interp_noise_multiplier in multiplier_list[::-1]:
							interp_noise = interp_noise_multiplier*interp_noise_base
							global interp_noise



							filename_token = error_folder+make_filename(toa_noise_multiplier,process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise)+'.npy'
							try:
								np.load(filename_token)
							except IOError:
								np.save(filename_token,0)
								name_string = 'toa noise %d, process position noise %d, process vel noise %d, interp noise %d, depth noise %d, stream noise %d'%(toa_noise_multiplier,
									process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise)
								print(name_string+' was not found, calculating')
								calculate(toa_noise,process_position_noise,process_vel_noise,depth_noise,stream_noise,interp_noise,all_floats,depth,stream,Smoother)

