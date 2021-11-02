from KalmanSmoother.Utilities.Filters import LeastSquares,Kalman,Smoother,ObsHolder
from KalmanSmoother.Utilities.Observations import SourceArray,Depth,Stream
from KalmanSmoother.Utilities.Floats import DIMESAllFloats,WeddellAllFloats
from GeneralUtilities.Filepath.instance import FilePathHandler
from KalmanSmoother.Utilities.__init__ import ROOT_DIR
from KalmanSmoother.Utilities.Utilities import parse_filename,make_filename
import numpy as np

def dimes_error_toa_save():
	file_handler = FilePathHandler(ROOT_DIR,'Tuning/TOADimes')
	all_floats = DIMESAllFloats()
	multiplier_list = [0.5,1,1.5]
	interp_noise = 0
	process_position_base_noise = 6
	for process_position_noise in [x*process_position_base_noise for x in multiplier_list+[2.0,2.5]]:
		process_vel_noise_base = 3
		for process_vel_noise in [x*process_vel_noise_base for x in multiplier_list]:
			toa_noise_base = 25
			for toa_noise in [x*toa_noise_base for x in multiplier_list+[2.0,2.5]]:
				filename_token = make_filename(toa_noise,process_position_noise,process_vel_noise,0,0,0)+'.npy'
				try:
					np.load(file_handler.out_file(filename_token))
				except IOError:
					np.save(file_handler.out_file(filename_token),0)
					name_string = 'toa noise %d, process position noise %d, process vel noise %d, interp noise %d, depth noise %d, stream noise %d'%(toa_noise,
						process_position_noise,process_vel_noise,0,0,0)
					print(name_string+' was not found, calculating')
					for idx,dummy in enumerate(all_floats.list):
						print(idx)
						dummy.toa.set_observational_uncertainty(toa_noise)
						dummy.depth = Depth([],[],dummy.clock)
						dummy.stream = Stream([],[],dummy.clock)
						dummy.gps.gps_interp_uncertainty = interp_noise
						obs_holder = ObsHolder(dummy)
						smooth =Smoother(dummy,all_floats.sources,obs_holder,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)
					misfit = all_floats.sources.return_misfit()[2]/toa_noise
					all_floats.sources.reset_error()
					np.save(file_handler.out_file(filename_token),misfit)


def dimes_error_save():
	file_handler = FilePathHandler(ROOT_DIR,'Tuning/Dimes')
	all_floats = DIMESAllFloats()
	multiplier_list = [0.5,1,1.5]
	interp_noise = 0
	process_position_base_noise = 6
	for process_position_noise in [x*process_position_base_noise for x in multiplier_list+[2.0,2.5]]:
		process_vel_noise_base = 3
		for process_vel_noise in [x*process_vel_noise_base for x in multiplier_list]:
			depth_noise_base = 1500
			for depth_noise in [x*depth_noise_base for x in multiplier_list+[2.0,2.5]]:
				stream_noise_base = 65
				for stream_noise in [x*stream_noise_base for x in multiplier_list]:
					toa_noise_base = 25
					for toa_noise in [x*toa_noise_base for x in multiplier_list+[2.0,2.5]]:
						filename_token = make_filename(toa_noise,process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise)+'.npy'
						try:
							np.load(file_handler.out_file(filename_token))
						except IOError:
							np.save(file_handler.out_file(filename_token),0)
							name_string = 'toa noise %d, process position noise %d, process vel noise %d, interp noise %d, depth noise %d, stream noise %d'%(toa_noise,
								process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise)
							print(name_string+' was not found, calculating')
							for idx,dummy in enumerate(all_floats.list):
								print(idx)
								dummy.toa.set_observational_uncertainty(toa_noise)
								dummy.depth.set_observational_uncertainty(depth_noise)
								dummy.stream.set_observational_uncertainty(stream_noise)
								dummy.gps.gps_interp_uncertainty = interp_noise
								obs_holder = ObsHolder(dummy)
								smooth =Smoother(dummy,all_floats.sources,obs_holder,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)
							misfit = all_floats.sources.return_misfit()[2]/toa_noise
							all_floats.sources.reset_error()
							np.save(file_handler.out_file(filename_token),misfit)

def weddell_error_save():
	file_handler = FilePathHandler(ROOT_DIR,'Tuning/Weddell')
	all_floats = WeddellAllFloats()
	multiplier_list = [0.5,1,1.5]
	process_position_base_noise = 6
	for process_position_noise in [x*process_position_base_noise for x in multiplier_list+[2.0,2.5]]:
		process_vel_noise_base = 3
		for process_vel_noise in [x*process_vel_noise_base for x in multiplier_list]:
			depth_noise_base = 1500
			for depth_noise in [x*depth_noise_base for x in multiplier_list+[2.0,2.5]]:
				stream_noise_base = 65
				for stream_noise in [x*stream_noise_base for x in multiplier_list]:
					toa_noise_base = 25
					for toa_noise in [x*toa_noise_base for x in multiplier_list+[2.0,2.5]]:
						interp_noise_base = 240
						for interp_noise in [x*interp_noise_base for x in multiplier_list]:
							filename_token = make_filename(toa_noise,process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise)+'.npy'
							try:
								np.load(file_handler.out_file(filename_token))
							except IOError:
								np.save(file_handler.out_file(filename_token),0)
								name_string = 'toa noise %d, process position noise %d, process vel noise %d, interp noise %d, depth noise %d, stream noise %d'%(toa_noise,
									process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise)
								print(name_string+' was not found, calculating')
								for idx,dummy in enumerate(all_floats.list):
									print(idx)
									dummy.toa.set_observational_uncertainty(toa_noise)
									dummy.depth.set_observational_uncertainty(depth_noise)
									dummy.stream.set_observational_uncertainty(stream_noise)
									dummy.gps.gps_interp_uncertainty = interp_noise
									obs_holder = ObsHolder(dummy)
									smooth =Smoother(dummy,all_floats.sources,obs_holder,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)
								misfit = all_floats.sources.return_misfit()[2]/toa_noise
								all_floats.sources.reset_error()
								np.save(file_handler.out_file(filename_token),misfit)