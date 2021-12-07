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
	process_position_base_noise = (3)**2
	for process_position_noise in [x*process_position_base_noise for x in multiplier_list+[2.0,2.5]]:
		process_vel_noise_base = (2)**2
		for process_vel_noise in [x*process_vel_noise_base for x in multiplier_list]:
			toa_noise_base = (8)**2
			for toa_noise in [x*toa_noise_base for x in multiplier_list+[2.0,2.5,0.25,0.125,3.0,3.5,4,4.5]]:
				filename_token = make_filename(toa_noise,process_position_noise,process_vel_noise,0,0,0)+'.npy'
				try:
					np.load(file_handler.out_file(filename_token))
				except IOError:
					np.save(file_handler.out_file(filename_token),0)
					name_string = 'toa noise %d, process position noise %d, process vel noise %d, interp noise %d, depth noise %d, stream noise %d'%(toa_noise,
						process_position_noise,process_vel_noise,0,0,0)
					print(name_string+' was not found, calculating')
					try:
						for idx,dummy in enumerate(all_floats.list):
							print(idx)
							dummy.toa.set_observational_uncertainty(toa_noise)
							dummy.depth = Depth([],[],dummy.clock)
							dummy.stream = Stream([],[],dummy.clock)
							dummy.gps.gps_interp_uncertainty = interp_noise
							obs_holder = ObsHolder(dummy)
							smooth =Smoother(dummy,all_floats.sources,obs_holder,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)
					except AssertionError:
						all_floats.sources.reset_error()
						continue
					misfit = all_floats.sources.return_misfit()[2]/toa_noise_base
					P = np.diag([process_position_base_noise,process_vel_noise_base,process_position_base_noise,process_vel_noise_base])
					model_size = all_floats.return_model_size(P)
					np.save(file_handler.out_file(filename_token),(misfit,model_size))


def dimes_error_save():
	file_handler = FilePathHandler(ROOT_DIR,'Tuning/Dimes')
	all_floats = DIMESAllFloats()
	all_floats.sources.reset_error()
	all_floats.reset_state()
#	multiplier_list = [0.125,0.25,0.5,1,1.5]
	multiplier_list = [0.25,0.5,1.0,1.5]
	interp_noise = 0
	process_position_base_noise = (3)
#	for process_position_noise in [(x*process_position_base_noise)**2 for x in multiplier_list+[2.0,2.5]]:
	for process_position_noise in [(x*process_position_base_noise)**2 for x in multiplier_list]:
		process_vel_noise_base = (3)
		P = np.diag([process_position_base_noise,process_vel_noise_base,process_position_base_noise,process_vel_noise_base])

#		for process_vel_noise in [(x*process_vel_noise_base)**2  for x in multiplier_list+[2.0,2.5]]:
		for process_vel_noise in [(x*process_vel_noise_base)**2  for x in multiplier_list]:
			depth_noise_base = (300)
			for depth_noise in [(x*depth_noise_base)**2 for x in multiplier_list]:
				toa_noise_base = (8)
				for toa_noise in [(x*toa_noise_base)**2 for x in [0.25,0.5,1.0,1.5,2.0,2.5,3.0]]:
					stream_noise_base = (20)
					# stream_noise = (stream_noise_base)**2
					for stream_noise in [(x*stream_noise_base)**2 for x in [0.25,0.5,1.0,1.5]]:
						filename_token = make_filename(toa_noise,process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise)+'.npy'
						assert all_floats.sources.return_misfit()==[0,0,0]
						assert all_floats.return_model_size(P)==0.0
						try:
							np.load(file_handler.out_file(filename_token))
						except IOError:
							np.save(file_handler.out_file(filename_token),0)
							name_string = 'toa noise %d, process position noise %d, process vel noise %d, interp noise %d, depth noise %d, stream noise %d'%(toa_noise,
								process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise)
							print(name_string+' was not found, calculating')
							try:
								for idx,dummy in enumerate(all_floats.list):
									print(idx)
									dummy.toa.set_observational_uncertainty(toa_noise)
									dummy.depth.set_observational_uncertainty(depth_noise)
									dummy.stream.set_observational_uncertainty(stream_noise)
									dummy.gps.gps_interp_uncertainty = interp_noise
									obs_holder = ObsHolder(dummy)
									smooth =Smoother(dummy,all_floats.sources,obs_holder,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)
							except AssertionError:
								all_floats.sources.reset_error()
								all_floats.reset_state()
								continue
							misfit = all_floats.sources.return_misfit()[2]/toa_noise_base
							model_size = all_floats.return_model_size(P)
							all_floats.sources.reset_error()
							all_floats.reset_state()
							np.save(file_handler.out_file(filename_token),(misfit,model_size))


def weddell_error_save():
	file_handler = FilePathHandler(ROOT_DIR,'Tuning/Weddell')
	all_floats = WeddellAllFloats()
	all_floats.sources.reset_error()
	all_floats.reset_state()
	multiplier_list = [0.25,0.5,1.0,1.5]
	process_position_base_noise = (3)
	for process_position_noise in [(x*process_position_base_noise)**2 for x in multiplier_list]:
		process_vel_noise_base = (3)
		P = np.diag([process_position_base_noise,process_vel_noise_base,process_position_base_noise,process_vel_noise_base])

		for process_vel_noise in [(x*process_vel_noise_base)**2  for x in multiplier_list]:
			depth_noise_base = (300)
			for depth_noise in [(x*depth_noise_base)**2 for x in multiplier_list]:
				toa_noise_base = (8)
				for toa_noise in [(x*toa_noise_base)**2 for x in [0.25,0.5,1.0,1.5,2.0,2.5,3.0]]:
					stream_noise_base = (20)
					for stream_noise in [(x*stream_noise_base)**2 for x in [0.25,0.5,1.0,1.5]]:

						interp_noise_base = 120
						filename_token = make_filename(toa_noise,process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise)+'.npy'
						assert all_floats.sources.return_misfit()==[0,0,0]
						assert all_floats.return_model_size(P)==0.0
						try:
							np.load(file_handler.out_file(filename_token))
						except IOError:
							np.save(file_handler.out_file(filename_token),0)
							name_string = 'toa noise %d, process position noise %d, process vel noise %d, interp noise %d, depth noise %d, stream noise %d'%(toa_noise,
								process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise)
							print(name_string+' was not found, calculating')
							try:
								for idx,dummy in enumerate(all_floats.list):
									print(idx)
									dummy.toa.set_observational_uncertainty(toa_noise)
									dummy.depth.set_observational_uncertainty(depth_noise)
									dummy.stream.set_observational_uncertainty(stream_noise)
									dummy.gps.gps_interp_uncertainty = interp_noise
									obs_holder = ObsHolder(dummy)
									smooth =Smoother(dummy,all_floats.sources,obs_holder,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)
							except AssertionError:
								all_floats.sources.reset_error()
								continue
							misfit = all_floats.sources.return_misfit()[2]/toa_noise_base
							model_size = all_floats.return_model_size(P)
							all_floats.sources.reset_error()
							all_floats.reset_state()
							np.save(file_handler.out_file(filename_token),(misfit,model_size))