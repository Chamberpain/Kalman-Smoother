import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.pyplot as plt
from Projects.KalmanSmoother.__init__ import ROOT_DIR
base_dir = ROOT_DIR+'/Data/output/'
from GeneralUtilities.Filepath.instance import FilePathHandler
from KalmanSmoother.Utilities.__init__ import ROOT_DIR as ROOT_DIR_PLOT
import pickle
from GeneralUtilities.Plot.Cartopy.eulerian_plot import BaseCartopy
import cartopy.crs as ccrs
import matplotlib as mpl

file_handler = FilePathHandler(ROOT_DIR_PLOT,'dimes_traj_plot')

class TrajDictCartopy(BaseCartopy):
	def __init__(self,*args,pad=1,dictionary=None,**kwargs):
		super().__init__(*args,**kwargs)
		llcrnrlon=(min(dictionary['trj_lons']+dictionary['lons'])-pad)
		llcrnrlat=(min(dictionary['trj_lats']+dictionary['lats'])-pad)
		urcrnrlon=(max(dictionary['trj_lons']+dictionary['lons'])+pad)
		urcrnrlat=(max(dictionary['trj_lats']+dictionary['lats'])+pad)
		self.ax.set_extent([llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat], crs=ccrs.PlateCarree())
		self.finish_map()


with open(base_dir+'dimes_trj.pickle', 'rb') as pickle_file:
    dimes_trj_dict = pickle._Unpickler(pickle_file)
    dimes_trj_dict.encoding = 'latin1'
    dimes_trj_dict = dimes_trj_dict.load()

artoa_cm = plt.cm.get_cmap('Blues')
cmaplist = [artoa_cm(i) for i in range(artoa_cm.N)]
artoa_cm = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, artoa_cm.N)
bounds = np.linspace(0, 4, 5)
artoa_norm = mpl.colors.BoundaryNorm(bounds, artoa_cm.N)
kalman_cm = plt.cm.get_cmap('Greens')
cmaplist = [kalman_cm(i) for i in range(kalman_cm.N)]
kalman_cm = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, kalman_cm.N)
bounds = np.linspace(0, 4, 5)
kalman_norm = mpl.colors.BoundaryNorm(bounds, kalman_cm.N)


fig = plt.figure(figsize=(16,12))
axs = [fig.add_subplot(2,2,x,projection=ccrs.PlateCarree()) for x in [1,2,3,4]]
for holder in zip(axs,['811','810','815','832'],['a','b','c','d']):
	ax,name,label = holder
	data = dimes_trj_dict[name]
	trj_lons = data['trj_lons']
	trj_lats = data['trj_lats']
	lons = data['lons']
	lats = data['lats']
	toa_num = data['toa_number']
	lon_grid,lat_grid,ax = TrajDictCartopy(dictionary=data,pad=2,ax=ax).get_map()
	ks = ax.scatter(lons,lats,c = toa_num,label='Kalman',alpha=0.6,cmap=kalman_cm,norm=kalman_norm)
	art = ax.scatter(trj_lons,trj_lats,c=toa_num[:len(trj_lons)],label='ARTOA',alpha=0.6,cmap=artoa_cm,norm=artoa_norm)
	ax.annotate(label,xy = (0.1,0.8),xycoords='axes fraction',zorder=10,size=20,bbox=dict(boxstyle="round", fc="0.8"),)
plt.subplots_adjust(wspace=0.35)

fig.colorbar(art,ax=[axs[0],axs[2]],label='ARTOA Sources Heard',location='top',shrink=0.9)
fig.colorbar(ks,ax=[axs[1],axs[3]],label='Kalman Sources Heard',location='top',shrink=0.9)
plt.subplots_adjust(top=0.75)


plt.savefig(file_handler.out_file('dimes_traj_plot'))
plt.close()