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


WeddellAllFloats.list = []
process_position_noise = 20.25
process_vel_noise = 0.5625
interp_noise = 14400.0
depth_noise = 5625
stream_noise = 900.0
gps_noise = .1
toa_noise = 64.0
all_floats = WeddellAllFloats()
for idx,dummy in enumerate(all_floats.list):
    print(idx)
    dummy.toa.set_observational_uncertainty(toa_noise)
    dummy.depth.set_observational_uncertainty(depth_noise)
    dummy.stream.set_observational_uncertainty(stream_noise)
    dummy.gps.interp_uncertainty = interp_noise
    obs_holder = ObsHolder(dummy)
    smooth =Smoother(dummy,all_floats.sources,obs_holder,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)

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
	first_lat_list.append(lat[0])
	first_lon_list.append(lon[0])
	last_lat_list.append(lat[-1])
	last_lon_list.append(lon[-1])
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
lon_grid,lat_grid,ax = WeddellSeaCartopy(ax=ax).get_map()
norm=colors.LogNorm(vmin=10, vmax=max(uncert_list))
cm = plt.cm.get_cmap('copper')
sc = ax.scatter(lon,lat,s=5,c=uncert_list,alpha=0.8,norm=norm,cmap=cm)
plt.colorbar(sc,label='Uncertainty (km)',location='bottom')

lat_list = []
lon_list = []
for _,source in smooth.sources.array.items():
	if source.mission=='Weddell':
		lat_list.append(source.position.latitude)
		lon_list.append(source.position.longitude)
ax.scatter(first_lon_list,first_lat_list,s=50,c='m',marker='^')
ax.scatter(last_lon_list,last_lat_list,s=50,c='m',marker='s')
ax.scatter(lon_list,lat_list,s=50,c='r')
plt.savefig(file_handler.out_file('Figure_15'))
plt.close()
