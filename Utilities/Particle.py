from GeneralUtilities.Filepath.instance import FilePathHandler
from KalmanSmoother.Utilities.Floats import ArtificialFloats,AllFloats
from KalmanSmoother.Utilities.__init__ import ROOT_DIR
from KalmanSmoother.Utilities.Filters import LeastSquares,Smoother,ObsHolder
from KalmanSmoother.Utilities.Utilities import KalmanPoint
from geopy.distance import GreatCircleDistance
import uuid
import numpy as np

file_handler = FilePathHandler(ROOT_DIR,'Particle')

def make_filename():
	return file_handler.out_file(str(uuid.uuid4()))

def pos_misfit_calc(pos_list,exact_list):
	return np.mean([GreatCircleDistance(estimate,exact).km for estimate,exact in zip(pos_list,exact_list)])

def particle_release():
	all_floats = ArtificialFloats()
	for percent in [0.1,0.3,0.7]:
		for idx in range(10000):
			print(idx)
			dummy = all_floats.random(percent)
			try:
				toa_error = (dummy.toa_noise)
				toa_number = (dummy.toa_number)
				percent_list = (percent)

				process_noise = (all_floats.var_x)*percent
				process_position_noise = process_noise
				process_vel_noise = process_noise
				gps_number= len(dummy.gps.obs)
				ls = LeastSquares(dummy,all_floats.sources,ObsHolder(dummy),process_position_noise=process_position_noise*1000,process_vel_noise =process_vel_noise*1000)
				ls_pos_misfit = pos_misfit_calc(dummy.pos[:-1],dummy.exact_pos)
				assert ls_pos_misfit>0
				smooth =Smoother(dummy,dummy.sources,ObsHolder(dummy),process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)
				smoother_pos_misfit = pos_misfit_calc(dummy.pos[:-1],dummy.exact_pos)
				kalman_pos_misfit = pos_misfit_calc(dummy.kalman_pos[:-1],dummy.exact_pos)
				assert kalman_pos_misfit>0
				ls_toa_misfit,kalman_toa_misfit,smoother_toa_misfit = dummy.sources.return_misfit()
				if smoother_pos_misfit>ls_pos_misfit:
					print('LS Pos Misfit is Lowest')

				filename= make_filename()
				np.save(filename,[toa_error,toa_number,percent_list,gps_number, \
					smoother_pos_misfit,kalman_pos_misfit,ls_pos_misfit,smoother_toa_misfit,kalman_toa_misfit,ls_toa_misfit])
			except:
				print('encountered an error, advancing')
				continue
			all_floats.sources.reset_error()