from KalmanSmoother.Utilities.Utilities import parse_filename
import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from GeneralUtilities.Filepath.instance import FilePathHandler
from KalmanSmoother.Utilities.__init__ import ROOT_DIR as ROOT_DIR

plot_file_handler = FilePathHandler(ROOT_DIR,'FinalFloatsPlot')

def compile_tuning_dataframe(error_folder):
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
	misfit_list = [misfit for misfit,toa in zip(misfit_list,toa_list)]
	toa_list = np.array(toa_list, dtype='a5')
	# toa_list[toa_list==b'62.5']=['High']
	# toa_list[toa_list==b'50.0']=['MedHigh']
	toa_list[toa_list==b'37.5']=['High']
	toa_list[toa_list==b'25.0']=['Med']
	toa_list[toa_list==b'12.5']=['Low']

	process_position_list = np.array(process_position_list, dtype='a5')
	# process_position_list[process_position_list==b'15.0']=['High']
	# process_position_list[process_position_list==b'12.0']=['MedHigh']
	process_position_list[process_position_list==b'9.0']=['High']
	process_position_list[process_position_list==b'6.0']=['Med']
	process_position_list[process_position_list==b'3.0']=['Low']

	process_vel_list = np.array(process_vel_list, dtype='a5')
	process_vel_list[process_vel_list==b'4.5']=['High']
	process_vel_list[process_vel_list==b'3.0']=['Med']
	process_vel_list[process_vel_list==b'1.5']=['Low']

	depth_list = np.array(depth_list, dtype='a5')
	# depth_list[depth_list==b'3750.']=['High']
	# depth_list[depth_list==b'3000.']=['MedHigh']
	depth_list[depth_list==b'2250.']=['High']
	depth_list[depth_list==b'1500.']=['Med']
	depth_list[depth_list==b'750.0']=['Low']

	stream_list = np.array(stream_list, dtype='a5')
	stream_list[stream_list==b'97.5']=['High']
	stream_list[stream_list==b'65.0']=['Med']
	stream_list[stream_list==b'32.5']=['Low']

	interp_list = np.array(interp_list, dtype='a5')
	interp_list[interp_list==b'360.0']=['High']
	interp_list[interp_list==b'240.0']=['Med']
	interp_list[interp_list==b'120.0']=['Low']


	dataframe_list = []
	for token_name,token in [('TOA Noise',toa_list),
	('Process Position Noise',process_position_list),('Process Velocity Noise',process_vel_list)
	,('Depth Noise',depth_list),('Stream Noise',stream_list),('Interp Noise',interp_list)]:
		dataframe = pd.DataFrame({'Error Type':token_name,'Condition':token,'Misfit':misfit_list})
		dataframe_list.append(dataframe)
	dataframe = pd.concat(dataframe_list)
	dataframe = dataframe.reset_index()
	dataframe['Condition'] = [x.decode('utf-8') for x in dataframe['Condition'].tolist()]
	return dataframe

# file_handler = FilePathHandler(ROOT_DIR,'Tuning/TOADimes')
# dataframe = compile_tuning_dataframe(file_handler.out_file(''))
# dataframe = dataframe[dataframe['Error Type']!='Interp Noise']
# dataframe = dataframe[dataframe.Condition.isin(['Med','Low','High'])]
# print('Best TOA DIMES Is')
# print(dataframe[dataframe.Misfit==dataframe.Misfit.min()])

hue_order = ['Low','Med','High']	
file_handler = FilePathHandler(ROOT_DIR,'Tuning/Dimes')
dataframe = compile_tuning_dataframe(file_handler.out_file(''))
dataframe = dataframe[dataframe['Error Type']!='Interp Noise']
dataframe = dataframe[~dataframe['index'].isin(dataframe[~dataframe.Condition.isin(['Med','Low','High'])]['index'].unique())]

fig = plt.figure(figsize=(20,14))
plt.rcParams['font.size'] = 24
params = dict(data=dataframe,
              y="Misfit",
              x="Error Type",
              hue="Condition",
              dodge=True)
p = sns.stripplot(size=8,
                  jitter=0.15,
                  edgecolor='black',
                  linewidth=1,
                  hue_order=hue_order,
                  **params)
# p.set_yscale("log")
p_box = sns.boxplot(linewidth=6,hue_order=hue_order,**params)
handles, labels = p_box.get_legend_handles_labels()
plt.legend(handles[:5], labels[:5], title="Condition",
          handletextpad=0.5, columnspacing=1,ncol=5,bbox_to_anchor=(0.5, 1.13),
          loc="upper center", frameon=False)
plt.xticks(rotation=20)
print('Best DIMES Is')
print(dataframe[dataframe.Misfit==dataframe.Misfit.min()])
plt.savefig(plot_file_handler.out_file('Figure_16'))
plt.close()

file_handler = FilePathHandler(ROOT_DIR,'Tuning/Weddell')
dataframe = compile_tuning_dataframe(file_handler.out_file(''))
dataframe = dataframe[~dataframe['index'].isin(dataframe[~dataframe.Condition.isin(['Med','Low','High'])]['index'].unique())]
fig = plt.figure(figsize=(20,14))
plt.rcParams['font.size'] = 24
params = dict(data=dataframe,
              y="Misfit",
              x="Error Type",
              hue="Condition",
              dodge=True)
p = sns.stripplot(size=8,
                  jitter=0.15,
                  edgecolor='black',
                  linewidth=1,
                  hue_order=hue_order,
                  **params)
p.set_yscale("log")
p_box = sns.boxplot(linewidth=6,hue_order=hue_order,**params)
handles, labels = p_box.get_legend_handles_labels()
plt.legend(handles[:5], labels[:5], title="Condition",
          handletextpad=0.5, columnspacing=3,ncol=5,bbox_to_anchor=(0.5, 1.13),
          loc="upper center", frameon=False)
plt.xticks(rotation=20)
print('Best Weddell Sea Is')
print(dataframe[dataframe.Misfit==dataframe.Misfit.min()])
plt.savefig(plot_file_handler.out_file('Figure_9'))
plt.close()
