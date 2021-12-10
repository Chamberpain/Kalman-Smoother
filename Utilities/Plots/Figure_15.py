from GeneralUtilities.Plot.Cartopy.regional_plot import WeddellSeaCartopy
from KalmanSmoother.Utilities.Floats import WeddellAllFloats
from KalmanSmoother.Utilities.Filters import Smoother,ObsHolder
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from KalmanSmoother.Utilities.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
import cartopy.crs as ccrs
file_handler = FilePathHandler(ROOT_DIR,'FinalFloatsPlot')
from KalmanSmoother.Utilities.DataLibrary import weddell_position_process,weddell_velocity_process,weddell_depth_noise,weddell_stream_noise,weddell_toa_noise,weddell_interp_noise
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
WeddellAllFloats.list = []
all_floats = WeddellAllFloats()
for idx,dummy in enumerate(all_floats.list):
    print(idx)
    dummy.toa.set_observational_uncertainty(weddell_toa_noise)
    dummy.depth.set_observational_uncertainty(weddell_depth_noise)
    dummy.stream.set_observational_uncertainty(weddell_stream_noise)
    dummy.gps.interp_uncertainty = weddell_interp_noise
    obs_holder = ObsHolder(dummy)
    smooth =Smoother(dummy,all_floats.sources,obs_holder,process_position_noise=weddell_position_process,process_vel_noise =weddell_velocity_process)

first_lat_list= []
first_lon_list = []
last_lat_list = []
last_lon_list = []
lat = []
lon = []
uncert_list = []

for k,dummy in enumerate(all_floats.list):	
	for idx in range(len(dummy.P)):
		C = dummy.P[idx][[0,0,2,2],[0,2,0,2]]
		C = C.reshape([2,2])
		w,v = np.linalg.eig(C)
		uncert_list.append(2*max(w)*np.sqrt(5.991))
	lat_holder,lon_holder = list(zip(*[(x.latitude,x.longitude) for x in dummy.pos]))
	lat += lat_holder
	lon += lon_holder
	first_lat_list.append(lat_holder[0])
	first_lon_list.append(lon_holder[0])
	last_lat_list.append(lat_holder[-1])
	last_lon_list.append(lon_holder[-1])
fig = plt.figure(figsize=(15,15))
ax1 = fig.add_subplot(2,1,1,projection=ccrs.PlateCarree())
lon_grid,lat_grid,ax1 = WeddellSeaCartopy(ax=ax1).get_map()
for dummy in all_floats.list:
	gps_list = dummy.gps.obs
	lats,lons = zip(*[(gps.latitude,gps.longitude) for gps in gps_list])
	ax1.plot(lons,lats,'g-*')
lat_list = []
lon_list = []
for _,source in smooth.sources.array.items():
	if source.mission=='Weddell':
		lat_list.append(source.position.latitude)
		lon_list.append(source.position.longitude)
ax1.scatter(first_lon_list,first_lat_list,s=50,c='b',marker='^')
ax1.scatter(last_lon_list,last_lat_list,s=50,c='m',marker='s')
ax1.scatter(lon_list,lat_list,s=50,c='r')
ax1.annotate('a',xy = (0.1,0.8),xycoords='axes fraction',zorder=10,size=20,bbox=dict(boxstyle="round", fc="0.8"),)


ax2 = fig.add_subplot(2,1,2,projection=ccrs.PlateCarree())
lon_grid,lat_grid,ax2 = WeddellSeaCartopy(ax=ax2).get_map()
norm=colors.LogNorm(vmin=10, vmax=max(uncert_list))
cm = plt.cm.get_cmap('copper')
sc = ax2.scatter(lon,lat,s=5,c=uncert_list,alpha=0.8,norm=norm,cmap=cm)
PCM = ax2.get_children()[0]
fig.colorbar(PCM,ax=[ax1,ax2],pad=.05,label='Uncertainty (km)',location='bottom')
lat_list = []
lon_list = []
for _,source in smooth.sources.array.items():
	if source.mission=='Weddell':
		lat_list.append(source.position.latitude)
		lon_list.append(source.position.longitude)
ax2.scatter(first_lon_list,first_lat_list,s=50,c='b',marker='^')
ax2.scatter(last_lon_list,last_lat_list,s=50,c='m',marker='s')
ax2.scatter(lon_list,lat_list,s=50,c='r')
ax2.annotate('b',xy = (0.1,0.8),xycoords='axes fraction',zorder=10,size=20,bbox=dict(boxstyle="round", fc="0.8"),)

plt.savefig(file_handler.out_file('Figure_15'))
plt.close()
