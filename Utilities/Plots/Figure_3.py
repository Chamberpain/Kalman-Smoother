from KalmanSmoother.Utilities.Floats import AllFloats
from KalmanSmoother.Utilities.Plots.FilterPlots import TrajDictCartopy
from KalmanSmoother.Utilities.Floats import WeddellAllFloats
from KalmanSmoother.Utilities.Filters import Smoother,ObsHolder
from GeneralUtilities.Compute.constants import degree_dist
import cartopy.crs as ccrs
from pyproj import Geod
import numpy as np
from matplotlib import patches
import shapely.geometry as sgeom
import matplotlib.pyplot as plt
from KalmanSmoother.Utilities.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
file_handler = FilePathHandler(ROOT_DIR,'FinalFloatsPlot')


WeddellAllFloats.list = []
process_position_noise = 9
process_vel_noise = 3.0
interp_noise = 360.0
depth_noise = 2250
stream_noise = 32.5
gps_noise = .1
toa_noise = 37.5
all_floats = WeddellAllFloats()
for idx,dummy in enumerate(all_floats.list):
    print(idx)
    dummy.toa.set_observational_uncertainty(toa_noise)
    dummy.depth.set_observational_uncertainty(depth_noise)
    dummy.stream.set_observational_uncertainty(stream_noise)
    dummy.gps.interp_uncertainty = interp_noise
    obs_holder = ObsHolder(dummy)
    smooth =Smoother(dummy,all_floats.sources,obs_holder,process_position_noise=process_position_noise,process_vel_noise =process_vel_noise)

geod = Geod(ellps='WGS84')
floatname_list = [x.floatname for x in all_floats.list]
idx = floatname_list.index(5901718)
dummy = all_floats.list[idx]	
lat,lon = zip(*[(x.latitude,x.longitude) for x in dummy.pos])
idx =630
lon_grid,lat_grid,ax = TrajDictCartopy(lon,lat,pad=1).get_map()
pos = dummy.pos[idx]
C = dummy.P[idx][[0,0,2,2],[0,2,0,2]]
C = C.reshape([2,2])
w,v = np.linalg.eig(C)
angle = np.degrees(np.arctan(v[1,np.argmax(w)]/v[0,np.argmax(w)]))
A = 2*max(w)*np.sqrt(9.210)/abs(degree_dist*np.cos(np.deg2rad(pos.latitude)))
B = 2*min(w)*np.sqrt(9.210)/degree_dist
ax.plot(lon[idx],lat[idx],marker='X',color='lime',markersize=12,zorder=10)
ax.plot(lon[0],lat[0],marker='^',color='m',markersize=12)
ax.plot(lon[-1],lat[-1],marker='s',color='m',markersize=12)
ax.plot(lon,lat,color='k')
lons, lats = ax.ellipse(geod,pos.longitude, pos.latitude,A*100000,B*100000,phi=angle)
holder = sgeom.Polygon(zip(lons, lats))
ax.add_geometries([sgeom.Polygon(zip(lons, lats))], ccrs.Geodetic(), facecolor='blue', alpha=0.7)
date = dummy.pos_date[idx]
dummy.clock.set_date(date) 
smooth.sources.set_date(date)
# colorlist = ['m','b','g','y']
for data in dummy.toa.return_data():
	# color = colorlist.pop()
	(toa,source) = data
	print(source.ID)
	detrend_toa = dummy.clock.detrend_offset(toa)
	detrend_toa = source.clock.detrend_offset(detrend_toa)
	dist = source.dist_from_toa(detrend_toa)
	print(source.position)
	print(dist)
	source_lat = source.position.latitude
	source_lon = source.position.longitude
	ax.scatter(source_lon,source_lat,s=20,color='r')
	lons, lats = ax.ellipse(geod,source_lon,source_lat,dist*1000,dist*1000)
	ax.annotate(source.ID,(source_lon+0.5,source_lat-0.3))
	ax.plot(lons,lats,color='r')
plt.savefig(file_handler.out_file('Figure_3'))
plt.close()