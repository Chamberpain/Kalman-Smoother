from KalmanSmoother.Utilities.Filters import LeastSquares,Kalman,Smoother,ObsHolder
from KalmanSmoother.Utilities.Observations import SourceArray, Depth, Stream
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
	process_position_noise = 15.0
	process_vel_noise = 4.5
	interp_noise = 360.0
	depth_noise = 3750
	stream_noise = 97.5
	gps_noise = .1
	toa_noise = 62.5
	
	# depth = Depth()
	# depth.guassian_smooth(sigma=4)
	# stream = Stream()
	# depth.guassian_smooth(sigma=2)
	# depth_flag = True
	# stream_flag = True
	# lin_between_obs=True
	all_floats = WeddellAllFloats()
	for idx,dummy in enumerate(all_floats.list):
		print(idx)
		dummy.toa.set_observational_uncertainty(toa_noise)
		dummy.depth.set_observational_uncertainty(depth_noise)
		dummy.stream.set_observational_uncertainty(stream_noise)
		dummy.gps.interp_uncertainty = interp_noise
		obs_holder = ObsHolder(dummy)
		smooth =Smoother(dummy,all_floats.sources,obs_holder,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)
	for holder in all_floats.list:
		print(holder.floatname)
		lons,lats = zip(*[(x.longitude,x.latitude) for x in holder.pos])

		lon_grid,lat_grid,ax = TrajDictCartopy(lons,lats,pad=1).get_map()
		ax.scatter(lons,lats,label='Kalman',alpha=0.3)
		plt.savefig(str(holder.floatname))
		plt.close()

def optimal_dimes():
	process_position_noise = 15.0
	process_vel_noise = 4.5
	depth_noise = 3750
	stream_noise = 65.0
	gps_noise = .1
	toa_noise = 62.5
	
	all_floats = DIMESAllFloats()

	for idx,dummy in enumerate(all_floats.list):
		print(idx)
		dummy.toa.set_observational_uncertainty(toa_noise)
		dummy.stream.set_observational_uncertainty(stream_noise)
		dummy.depth.set_observational_uncertainty(depth_noise)

		obs_holder = ObsHolder(dummy)
		smooth =Smoother(dummy,all_floats.sources,obs_holder,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)

	for holder in all_floats.list:
		print(holder.floatname)
		lons,lats = zip(*[(x.longitude,x.latitude) for x in holder.pos])
		trj_lons,trj_lats = zip(*[(x.longitude,x.latitude) for x in holder.trj_pos])

		lon_grid,lat_grid,ax = TrajDictCartopy(lons+trj_lons,lats+trj_lats,pad=1).get_map()
		ax.scatter(lons,lats,label='Kalman',alpha=0.3,zorder=10)
		ax.scatter(trj_lons,trj_lats,label='ARTOA',alpha=0.3,zorder=10)
		plt.legend()
		plt.savefig(str(holder.floatname))
		plt.close()

	all_floats.DIMES_speed_hist()
