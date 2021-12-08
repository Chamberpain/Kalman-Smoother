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
from KalmanSmoother.Utilities.Floats import DIMESAllFloats,WeddellAllFloats
from geopy.distance import GreatCircleDistance
from KalmanSmoother.Utilities.Utilities import speed_calc
from KalmanSmoother.Utilities.Filters import Smoother,ObsHolder
from KalmanSmoother.Utilities.DataLibrary import dimes_position_process,dimes_velocity_process,dimes_depth_noise,dimes_stream_noise,dimes_toa_noise,dimes_interp_noise
from KalmanSmoother.Utilities.DataLibrary import weddell_position_process,weddell_velocity_process,weddell_depth_noise,weddell_stream_noise,weddell_toa_noise,weddell_interp_noise

file_handler = FilePathHandler(ROOT_DIR_PLOT,'FinalFloatsPlot')
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

smooth_toa_error = []
smooth_speed = []
for idx,dummy in enumerate(all_floats.list):
    print(idx)
    dist_error_list,toa_error_list,dist_list,soso_list,date_return_list,obs_list = dummy.toa.calculate_error_list(dummy.pos,dummy.pos_date)
    smooth_toa_error += toa_error_list
    smooth_speed += speed_calc(dummy.pos,dummy.pos_date)
smooth_speed = [x for x in smooth_speed]


fig = plt.figure(figsize=(12,12))

plt.subplot(2,1,1)
plt.hist(smooth_toa_error,bins=100)
plt.yscale('log')
plt.xlim([-70,70])
plt.xlabel('Misfit (s)')
plt.annotate('a',xy = (0.8,0.75),xycoords='axes fraction',zorder=10,size=28,bbox=dict(boxstyle="round", fc="0.8"),)


plt.subplot(2,1,2)
plt.hist(smooth_speed,bins=100)
plt.yscale('log')
plt.xlim([0,20])
plt.xlabel('Speed ($km\ day^{-1}$)')
plt.annotate('b',xy = (0.8,0.75),xycoords='axes fraction',zorder=10,size=28,bbox=dict(boxstyle="round", fc="0.8"),)
plt.savefig(file_handler.out_file('Figure_17'))


del all_floats
all_floats = DIMESAllFloats()
for idx,dummy in enumerate(all_floats.list):
    print(idx)
    dummy.toa.set_observational_uncertainty(dimes_toa_noise)
    dummy.stream.set_observational_uncertainty(dimes_stream_noise)
    dummy.depth.set_observational_uncertainty(dimes_depth_noise)
    obs_holder = ObsHolder(dummy)
    smooth =Smoother(dummy,all_floats.sources,obs_holder,process_position_noise=dimes_position_process,process_vel_noise =dimes_velocity_process)
trj_dist_error = []
trj_toa_error = []
trj_speed = []
smooth_dist_error = []
smooth_toa_error = []
smooth_speed = []
for idx,dummy in enumerate(all_floats.list):
    print(idx)
    dist_error_list,toa_error_list,dist_list,soso_list,date_return_list,obs_list = dummy.toa.calculate_error_list(dummy.trj_pos,dummy.trj_date)
    trj_dist_error += dist_error_list
    trj_toa_error += toa_error_list
    trj_speed += speed_calc(dummy.trj_pos,dummy.trj_date)
    dist_error_list,toa_error_list,dist_list,soso_list,date_return_list,obs_list = dummy.toa.calculate_error_list(dummy.pos,dummy.pos_date)
    smooth_dist_error += dist_error_list
    smooth_toa_error += toa_error_list
    smooth_speed += speed_calc(dummy.pos,dummy.pos_date)
trj_diff_list = []
for idx,dummy in enumerate(all_floats.list):
    for pos,date in zip(dummy.trj_pos,dummy.trj_date):
        try:
            idx = dummy.pos_date.index(date)
            trj_diff_list.append(GreatCircleDistance(pos,dummy.pos[idx]).km)
        except:
            continue
plt.figure(figsize=(13,13))
plt.subplot(3,1,1)
bins = np.linspace(-30,30,200)
trj_n,dummy,dummy = plt.hist(trj_toa_error,bins=bins,color='b',label='ARTOA',alpha=0.3)
kalman_n,dummy,dummy = plt.hist(smooth_toa_error,bins=bins,color='g',label='Smoother',alpha=0.3)
plt.yscale('log')
plt.legend()
plt.xlabel('Misfit (seconds)', fontsize=22)
plt.annotate('a', xy = (0.2,0.75),xycoords='axes fraction',zorder=10,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
plt.subplot(3,1,2)
bins = np.linspace(0,300,40)
plt.xlim([0,300])
plt.hist(trj_diff_list,bins=bins)
plt.yscale('log')
plt.xlabel('Trajectory Difference (km)', fontsize=22)
plt.annotate('b', xy = (0.2,0.75),xycoords='axes fraction',zorder=10,size=32,bbox=dict(boxstyle="round", fc="0.8"),)     
plt.subplot(3,1,3)
bins = np.linspace(0,41,30)
plt.hist(smooth_speed,bins=bins,color='g',label='Kalman',alpha=0.3)
plt.hist(trj_speed,bins=bins,color='b',label='DIMES',alpha=0.3)
plt.yscale('log')
plt.annotate('c', xy = (0.8,0.7),xycoords='axes fraction',zorder=10,size=32,bbox=dict(boxstyle="round", fc="0.8"),)     
plt.xlim([0,35])
# plt.yscale('log')
plt.xlabel('Speed (km $day^{-1}$)', fontsize=22)
plt.subplots_adjust(hspace=0.3)
plt.savefig(file_handler.out_file('Figure_11'))
plt.close()