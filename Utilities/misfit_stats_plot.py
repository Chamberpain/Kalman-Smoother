import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from Projects.KalmanSmoother.__init__ import ROOT_DIR
base_dir = ROOT_DIR+'/Data/output/'
from GeneralUtilities.Filepath.instance import FilePathHandler
from KalmanSmoother.Utilities.__init__ import ROOT_DIR as ROOT_DIR_PLOT
import pickle
from GeneralUtilities.Plot.Cartopy.eulerian_plot import BaseCartopy
import cartopy.crs as ccrs
import matplotlib

file_handler = FilePathHandler(ROOT_DIR_PLOT,'misfit_stats_plot')


weddell_speed = np.load(base_dir+'weddell_all_toa_speed_list.npy')
weddell_toa_error = np.load(base_dir+'weddell_all_toa_error_list.npy')

matplotlib.rcParams.update({'font.size': 22})
fig = plt.figure(figsize=(12,12))

ax = fig.add_subplot(2,1,1)
ax.hist(weddell_toa_error,bins=100)
ax.set_yscale('log')
ax.set_xlim([-50,50])
ax.set_xlabel('Misfit (s)')
ax.annotate('a',xy = (0.8,0.75),xycoords='axes fraction',zorder=10,size=28,bbox=dict(boxstyle="round", fc="0.8"),)


ax = fig.add_subplot(2,1,2)
ax.hist(weddell_speed,bins=100)
ax.set_yscale('log')
ax.set_xlim([0,20])
ax.set_xlabel('Speed ($km\ day^{-1}$)')
ax.annotate('b',xy = (0.8,0.75),xycoords='axes fraction',zorder=10,size=28,bbox=dict(boxstyle="round", fc="0.8"),)
plt.savefig(file_handler.out_file('weddell_stats'))

with open(base_dir+'dimes_error.pickle', 'rb') as pickle_file:
    dimes_error_dict = pickle._Unpickler(pickle_file)
    dimes_error_dict.encoding = 'latin1'
    dimes_error_dict = dimes_error_dict.load()


plt.figure(figsize=(12,12))
plt.subplot(3,1,1)
bins = np.linspace(-30,30,200)
trj_n,dummy,dummy = plt.hist(dimes_error_dict['trj_error_list'],bins=bins,color='b',label='ARTOA',alpha=0.3)
kalman_n,dummy,dummy = plt.hist(dimes_error_dict['kalman_error_list'],bins=bins,color='g',label='Smoother',alpha=0.3)
plt.yscale('log')
plt.legend()
plt.xlabel('Misfit (seconds)', fontsize=22)
plt.annotate('a', xy = (0.2,0.75),xycoords='axes fraction',zorder=10,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
plt.subplot(3,1,2)
bins = np.linspace(0,300,40)
plt.xlim([0,300])
plt.hist(dimes_error_dict['diff_list'],bins=bins)
plt.yscale('log')
plt.xlabel('Trajectory Difference (km)', fontsize=22)
plt.annotate('b', xy = (0.2,0.75),xycoords='axes fraction',zorder=10,size=32,bbox=dict(boxstyle="round", fc="0.8"),)     
plt.subplot(3,1,3)
bins = np.linspace(0,41,30)
plt.hist(dimes_error_dict['kalman_speed'],bins=bins,color='g',label='Kalman',alpha=0.3)
plt.hist(dimes_error_dict['dimes_speed'],bins=bins,color='b',label='DIMES',alpha=0.3)
plt.annotate('c', xy = (0.8,0.7),xycoords='axes fraction',zorder=10,size=32,bbox=dict(boxstyle="round", fc="0.8"),)     
plt.xlim([0,20])
plt.xlabel('Speed (km $day^{-1}$)', fontsize=22)
plt.tight_layout(pad=1.5)
plt.savefig(file_handler.out_file('all_dimes_error'))
plt.close()




