from Projects.KalmanSmoother.Utilities.Utilities import parse_filename
from Projects.KalmanSmoother.__init__ import ROOT_DIR
import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from GeneralUtilities.Filepath.instance import FilePathHandler
from KalmanSmoother.Utilities.__init__ import ROOT_DIR as PLOT_ROOT_DIR

file_handler = FilePathHandler(PLOT_ROOT_DIR,'MisfitPlots')


error_folder = ROOT_DIR+'/Data/output/error/'

misfit_list = []
toa_list = []
process_position_list = []
process_vel_list = []
interp_list = []
depth_list = []
stream_list = []


for file_ in os.listdir(error_folder):
	if not file_.endswith('.npy'):
		continue
	misfit_value = np.load(error_folder+file_).tolist()
	if misfit_value==0:
		print(file_+' is zero, you should delete')
		continue
	misfit_list.append(np.load(error_folder+file_).tolist())
	toa_noise,process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise = parse_filename(file_)
	toa_list.append(toa_noise)
	process_position_list.append(process_position_noise)
	process_vel_list.append(process_vel_noise)
	depth_list.append(depth_noise)
	stream_list.append(stream_noise)
toa_list = np.array(toa_list, dtype='a5')
toa_list[toa_list==b'1.2']=['High']
toa_list[toa_list==b'1.0']=['Med']
toa_list[toa_list==b'0.8']=['Low']

process_position_list = np.array(process_position_list, dtype='a5')
process_position_list[process_position_list==b'6.0']=['High']
process_position_list[process_position_list==b'5.0']=['Med']
process_position_list[process_position_list==b'4.0']=['Low']

process_vel_list = np.array(process_vel_list, dtype='a5')
process_vel_list[process_vel_list==b'2.4']=['High']
process_vel_list[process_vel_list==b'2.0']=['Med']
process_vel_list[process_vel_list==b'1.6']=['Low']

depth_list = np.array(depth_list, dtype='a5')
depth_list[depth_list==b'240.0']=['High']
depth_list[depth_list==b'200.0']=['Med']
depth_list[depth_list==b'160.0']=['Low']

stream_list = np.array(stream_list, dtype='a5')
stream_list[stream_list==b'0.024']=['High']
stream_list[stream_list==b'0.02']=['Med']
stream_list[stream_list==b'0.016']=['Low']

dataframe_list = []
for token_name,token in [('TOA Noise',toa_list),
('Process Position Noise',process_position_list),('Process Velocity Noise',process_vel_list)
,('Depth Noise',depth_list),('Stream Noise',stream_list)]:
	dataframe = pd.DataFrame({'Float Type':'DIMES','Error Type':token_name,'Condition':token,'Misfit':misfit_list})
	dataframe_list.append(dataframe)
dataframe = pd.concat(dataframe_list)
dataframe = dataframe.reset_index()

dataframe['Condition'] = [x.decode('utf-8') for x in dataframe['Condition'].tolist()]

fig = plt.figure(figsize=(20,14))
plt.rcParams['font.size'] = 30
params = dict(data=dataframe,
              y="Misfit",
              x="Error Type",
              hue="Condition",
              dodge=True)
p = sns.stripplot(size=8,
                  jitter=0.15,
                  edgecolor='black',
                  linewidth=1,
                  **params)
p_box = sns.boxplot(linewidth=6,**params)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[:3], labels[:3], title="Condition",
          handletextpad=0.5, columnspacing=1,
          loc="upper left", frameon=False)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(file_handler.out_file('Dimes_Sensitivity'))
plt.close()



error_folder = ROOT_DIR+'/Data/output/weddell_error/'
misfit_list = []
toa_list = []
process_position_list = []
process_vel_list = []
interp_list = []
depth_list = []
stream_list = []


for file_ in os.listdir(error_folder):
	if not file_.endswith('.npy'):
		continue
	misfit_value = np.load(error_folder+file_).tolist()
	if misfit_value==0:
		print(file_+' is zero, you should delete')
		continue
	misfit_list.append(np.load(error_folder+file_).tolist())
	toa_noise,process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise = parse_filename(file_)
	toa_list.append(toa_noise)
	process_position_list.append(process_position_noise)
	process_vel_list.append(process_vel_noise)
	depth_list.append(depth_noise)
	stream_list.append(stream_noise)
	interp_list.append(interp_noise)

toa_list = np.array(toa_list, dtype='a5')
toa_list[toa_list==b'42.0']=['High']
toa_list[toa_list==b'35.0']=['Med']
toa_list[toa_list==b'28.0']=['Low']

process_position_list = np.array(process_position_list, dtype='a5')
process_position_list[process_position_list==b'6.0']=['High']
process_position_list[process_position_list==b'5.0']=['Med']
process_position_list[process_position_list==b'4.0']=['Low']

process_vel_list = np.array(process_vel_list, dtype='a5')
process_vel_list[process_vel_list==b'2.4']=['High']
process_vel_list[process_vel_list==b'2.0']=['Med']
process_vel_list[process_vel_list==b'1.6']=['Low']

depth_list = np.array(depth_list, dtype='a5')
depth_list[depth_list==b'240.0']=['High']
depth_list[depth_list==b'200.0']=['Med']
depth_list[depth_list==b'160.0']=['Low']

stream_list = np.array(stream_list, dtype='a5')
stream_list[stream_list==b'0.024']=['High']
stream_list[stream_list==b'0.02']=['Med']
stream_list[stream_list==b'0.016']=['Low']

interp_list = np.array(interp_list, dtype='a5')
interp_list[interp_list==b'336.0']=['High']
interp_list[interp_list==b'280.0']=['Med']
interp_list[interp_list==b'224.0']=['Low']

dataframe_list = []
for token_name,token in [('TOA Noise',toa_list),
('Process Position Noise',process_position_list),('Process Velocity Noise',process_vel_list)
,('Depth Noise',depth_list),('Stream Noise',stream_list),('Interp Noise',interp_list)]:
	dataframe = pd.DataFrame({'Float Type':'Weddell','Error Type':token_name,'Condition':token,'Misfit':misfit_list})
	dataframe_list.append(dataframe)
dataframe = pd.concat(dataframe_list)
dataframe = dataframe.reset_index()

dataframe['Condition'] = [x.decode('utf-8') for x in dataframe['Condition'].tolist()]

fig = plt.figure(figsize=(20,14))
plt.rcParams['font.size'] = 30
params = dict(data=dataframe,
              y="Misfit",
              x="Error Type",
              hue="Condition",
              dodge=True)
p = sns.stripplot(size=8,
                  jitter=0.15,
                  edgecolor='black',
                  linewidth=1,
                  **params)
p_box = sns.boxplot(linewidth=6,**params)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[:3], labels[:3], title="Condition",
          handletextpad=0.5, columnspacing=1,
          loc="upper left", frameon=False)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(file_handler.out_file('Weddell_Sensitivity'))
plt.close()
