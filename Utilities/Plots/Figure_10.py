import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.pyplot as plt
from KalmanSmoother.__init__ import ROOT_DIR
base_dir = ROOT_DIR+'/Data/output/'
from GeneralUtilities.Filepath.instance import FilePathHandler
from KalmanSmoother.Utilities.__init__ import ROOT_DIR as ROOT_DIR_PLOT
import pickle
from GeneralUtilities.Plot.Cartopy.eulerian_plot import BaseCartopy
import cartopy.crs as ccrs
import matplotlib as mpl
from KalmanSmoother.Utilities.Floats import DIMESAllFloats
from KalmanSmoother.Utilities.Filters import LeastSquares,Kalman,Smoother,ObsHolder
import matplotlib
from KalmanSmoother.Utilities.Observations import SourceArray,Depth,Stream
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime
from KalmanSmoother.Utilities.DataLibrary import dimes_position_process,dimes_velocity_process,dimes_depth_noise,dimes_stream_noise,dimes_toa_noise,dimes_interp_noise

font = {'family' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)


file_handler = FilePathHandler(ROOT_DIR_PLOT,'FinalFloatsPlot')

class TrajDictCartopy(BaseCartopy):
	def __init__(self,*args,pad=1,dictionary=None,**kwargs):
		super().__init__(*args,**kwargs)
		llcrnrlon=(min(dictionary['trj_lons']+dictionary['lons'])-pad)
		llcrnrlat=(min(dictionary['trj_lats']+dictionary['lats'])-pad)
		urcrnrlon=(max(dictionary['trj_lons']+dictionary['lons'])+pad)
		urcrnrlat=(max(dictionary['trj_lats']+dictionary['lats'])+pad)
		self.ax.set_extent([llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat], crs=ccrs.PlateCarree())
		self.finish_map()

artoa_cm = plt.cm.get_cmap('Blues')
cmaplist = [artoa_cm(i) for i in range(artoa_cm.N)]
artoa_cm = mpl.colors.LinearSegmentedColormap.from_list(
 'Custom cmap', cmaplist, artoa_cm.N)
bounds = np.linspace(0, 3, 4)
artoa_norm = mpl.colors.BoundaryNorm(bounds, artoa_cm.N)
kalman_cm = plt.cm.get_cmap('Greens')
cmaplist = [kalman_cm(i) for i in range(kalman_cm.N)]
kalman_cm = mpl.colors.LinearSegmentedColormap.from_list(
 'Custom cmap', cmaplist, kalman_cm.N)
bounds = np.linspace(0, 3, 4)
kalman_norm = mpl.colors.BoundaryNorm(bounds, kalman_cm.N)


all_floats = DIMESAllFloats()

for idx,dummy in enumerate(all_floats.list):
	print(idx)
	dummy.toa.set_observational_uncertainty(dimes_toa_noise)
	dummy.stream.set_observational_uncertainty(dimes_stream_noise)
	dummy.depth.set_observational_uncertainty(dimes_depth_noise)
	dummy.gps.gps_interp_uncertainty = dimes_interp_noise

	# dummy.depth = Depth([],[],dummy.clock)
	# dummy.stream = Stream([],[],dummy.clock)
	obs_holder = ObsHolder(dummy)
	smooth =Smoother(dummy,all_floats.sources,obs_holder,process_position_noise=dimes_position_process,process_vel_noise =dimes_velocity_process)



for dummy in all_floats.list:
	print(dummy.floatname)
	lats = [z.latitude for z in dummy.pos]
	lons = [z.longitude for z in dummy.pos]
	print(lons[0])
	print(lats[0])
	trj_lats = [z.latitude for z in dummy.trj_pos]
	trj_lons = [z.longitude for z in dummy.trj_pos]
	datelist = [dummy.clock.initial_date+datetime.timedelta(days=token) for token in range(len(lats))]
	toa_num = [len(dummy.toa.obs[dummy.toa.date.index(token)]) if token in dummy.toa.date else 0 for token in datelist]
	data = {
		'trj_lons':trj_lons,
		'trj_lats':trj_lats,
		'lons':lons, 
		'lats':lats,
		}
	plt.scatter(lons,lats,c = toa_num,label='Kalman',alpha=0.6,cmap=kalman_cm,norm=kalman_norm)
	plt.scatter(trj_lons,trj_lats,c=toa_num[:len(trj_lons)],label='ARTOA',alpha=0.6,cmap=artoa_cm,norm=artoa_norm)
	plt.savefig(str(dummy.floatname))
	plt.close()


float_list = [x.floatname for x in all_floats.list]
fig = plt.figure(figsize=(12,12))
axs = [fig.add_subplot(2,2,x,projection=ccrs.PlateCarree()) for x in [1,2,3,4]]
for holder in zip(axs,[808,853,854,802],['a','b','c','d']):
	print(holder)
	ax,name,label = holder
	float_idx = float_list.index(name)
	dummy = all_floats.list[float_idx]
	lats = [z.latitude for z in dummy.pos]
	lons = [z.longitude for z in dummy.pos]
	trj_lats = [z.latitude for z in dummy.trj_pos]
	trj_lons = [z.longitude for z in dummy.trj_pos]
	datelist = [dummy.clock.initial_date+datetime.timedelta(days=token) for token in range(len(lats))]
	toa_num = [len(dummy.toa.obs[dummy.toa.date.index(token)]) if token in dummy.toa.date else 0 for token in datelist]
	data = {
		'trj_lons':trj_lons,
		'trj_lats':trj_lats,
		'lons':lons, 
		'lats':lats,
		}
	lon_grid,lat_grid,ax = TrajDictCartopy(dictionary=data,pad=2,ax=ax).get_map()
	ks = ax.scatter(lons,lats,c = toa_num,label='Kalman',alpha=0.6,cmap=kalman_cm,norm=kalman_norm)
	art = ax.scatter(trj_lons,trj_lats,c=toa_num[:len(trj_lons)],label='ARTOA',alpha=0.6,cmap=artoa_cm,norm=artoa_norm)
	ax.annotate(label,xy = (0.1,0.8),xycoords='axes fraction',zorder=10,size=20,bbox=dict(boxstyle="round", fc="0.8"),)
	ax.annotate(name,xy = (0.3,0.8),xycoords='axes fraction',zorder=10,size=20,)

plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(top=0.85)

axins1 = inset_axes(axs[0],
                    width="100%",  # width = 50% of parent_bbox width
                    height="5%",  # height : 5%
                    loc='upper center',
					borderpad=-4
                    )

cbar1 = fig.colorbar(art,cax=axins1, orientation="horizontal",label='ARTOA Sources Heard')
cbar1.ax.set_xticklabels(['0','1','2','3+'])
axins2 = inset_axes(axs[1],
                    width="100%",  # width = 50% of parent_bbox width
                    height="5%",  # height : 5%
                    loc='upper center',
					borderpad=-4
                    )

cbar2 = fig.colorbar(ks,cax=axins2, orientation="horizontal",label='Kalman Sources Heard')
cbar2.ax.set_xticklabels(['0','1','2','3+'])


plt.savefig(file_handler.out_file('Figure_10'))
plt.close()