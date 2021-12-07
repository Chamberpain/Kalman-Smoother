from KalmanSmoother.Utilities.Utilities import parse_filename
import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from GeneralUtilities.Filepath.instance import FilePathHandler
from KalmanSmoother.Utilities.__init__ import ROOT_DIR as ROOT_DIR

plot_file_handler = FilePathHandler(ROOT_DIR,'FinalFloatsPlot')
hue_order = ['XS','S','M','L']
plt.rcParams['font.size'] = 24

def compile_tuning_dataframe_other(error_folder):
	misfit_list = []
	model_size_list = []
	toa_list = []
	process_position_list = []
	process_vel_list = []
	interp_list = []
	depth_list = []
	stream_list = []


	for file_ in os.listdir(error_folder):
		if not file_.endswith('.npy'):
			continue
		try: 
			misfit_value,model_size = np.load(error_folder+file_).tolist()
		except TypeError:
			print(file_+' is zero, you should delete')
			continue
		misfit_list.append(misfit_value)
		model_size_list.append(model_size)
		toa_noise,process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise = parse_filename(file_)
		toa_list.append(toa_noise)
		process_position_list.append(process_position_noise)
		process_vel_list.append(process_vel_noise)
		depth_list.append(depth_noise)
		stream_list.append(stream_noise)
		interp_list.append(interp_noise)
	toa_list = np.array(toa_list, dtype='a5')

	process_position_list = np.array(process_position_list, dtype='a5')
	process_position_list[process_position_list==b'56.25']=['XXL']
	process_position_list[process_position_list==b'36.0']=['XL']
	process_position_list[process_position_list==b'20.25']=['L']
	process_position_list[process_position_list==b'9.0']=['M']
	process_position_list[process_position_list==b'2.25']=['S']
	process_position_list[process_position_list==b'0.562']=['XS']
	process_position_list[process_position_list==b'0.140']=['XXS']



	process_vel_list = np.array(process_vel_list, dtype='a5')
	process_vel_list[process_vel_list==b'56.25']=['XXL']
	process_vel_list[process_vel_list==b'36.0']=['XL']	
	process_vel_list[process_vel_list==b'20.25']=['L']
	process_vel_list[process_vel_list==b'9.0']=['M']
	process_vel_list[process_vel_list==b'2.25']=['S']
	process_vel_list[process_vel_list==b'0.562']=['XS']
	process_vel_list[process_vel_list==b'0.140']=['XXS']



	depth_list = np.array(depth_list, dtype='a5')
	depth_list[depth_list==b'20250']=['L']
	depth_list[depth_list==b'90000']=['M']
	depth_list[depth_list==b'22500']=['S']
	depth_list[depth_list==b'5625.']=['XS']
	depth_list[depth_list==b'1406.']=['XXS']


	stream_list = np.array(stream_list, dtype='a5')
	stream_list[stream_list==b'900.0']=['L']
	stream_list[stream_list==b'400.0']=['M']
	stream_list[stream_list==b'100.0']=['S']	
	stream_list[stream_list==b'25.0']=['XS']



	interp_list = np.array(interp_list, dtype='a5')
	interp_list[interp_list==b'32400']=['L']
	interp_list[interp_list==b'14400']=['M']
	interp_list[interp_list==b'3600.']=['S']
	interp_list[interp_list==b'900.0']=['XS']
	interp_list[interp_list==b'225.0']=['XXS']	

	dataframe_list = []
	misfit_list = np.array(misfit_list)*np.array(misfit_list)*8
	for token_name,token in [('TOA Noise',toa_list),
	('Position Noise',process_position_list),('Velocity Noise',process_vel_list)
	,('Depth Noise',depth_list),('Stream Noise',stream_list)]:
		dataframe = pd.DataFrame({'Error Type':token_name,'Condition':token,'Misfit':misfit_list,'Model Size':model_size_list})
		dataframe_list.append(dataframe)
	dataframe = pd.concat(dataframe_list)
	dataframe = dataframe.reset_index()
	dataframe['Condition'] = [x.decode('utf-8') for x in dataframe['Condition'].tolist()]
	dataframe = dataframe[dataframe['Error Type']!='TOA Noise']
	return dataframe




def compile_tuning_dataframe_toa(error_folder):
	misfit_list = []
	model_size_list = []
	toa_list = []
	process_position_list = []
	process_vel_list = []
	interp_list = []
	depth_list = []
	stream_list = []


	for file_ in os.listdir(error_folder):
		if not file_.endswith('.npy'):
			continue
		try: 
			misfit_value,model_size = np.load(error_folder+file_).tolist()
		except TypeError:
			print(file_+' is zero, you should delete')
			continue
		misfit_list.append(misfit_value)
		model_size_list.append(model_size)
		toa_noise,process_position_noise,process_vel_noise,interp_noise,depth_noise,stream_noise = parse_filename(file_)

		toa_list.append(toa_noise)
		process_position_list.append(process_position_noise)
		process_vel_list.append(process_vel_noise)
		depth_list.append(depth_noise)
		stream_list.append(stream_noise)
		interp_list.append(interp_noise)

	misfit_list = np.array(misfit_list)*np.array(misfit_list)*8

	dataframe = pd.DataFrame({'Toa Noise':toa_list,'Misfit List':misfit_list,"Position Noise":process_position_list,"Velocity Noise":process_vel_list,
		'Model Size':model_size_list,'Depth Noise':depth_list,"Stream Noise":stream_list,'Interp Noise':interp_list})
	return dataframe




file_handler = FilePathHandler(ROOT_DIR,'Tuning/Dimes')

plt.figure(figsize=(18,14))
plt.subplot(1,2,1)
dataframe = compile_tuning_dataframe_other(file_handler.out_file(''))

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

p_box = sns.boxplot(linewidth=6,hue_order=hue_order,**params)
handles, labels = p_box.get_legend_handles_labels()
plt.legend(handles[:4], labels[:4], title="Condition",
          handletextpad=0.5, columnspacing=1,ncol=7,bbox_to_anchor=(0.5, 1.15),
          loc="upper center", frameon=False)
plt.xticks(rotation=15)
p.set_yscale("log")
p.set_ylabel('Data Misfit, $T^2=\Vert\Sigma^{-1} j \Vert ^2$')

plt.subplot(1,2,2)
dataframe = compile_tuning_dataframe_toa(file_handler.out_file(''))
holder = dataframe[dataframe['Misfit List']==dataframe['Misfit List'].min()]
for k,data in holder[['Position Noise','Velocity Noise','Depth Noise','Stream Noise','Interp Noise']].iterrows():
	(pn,vn,dn,sn,interpn) = data

dm = dataframe['Depth Noise']==dn
sm = dataframe['Stream Noise']==sn
pm = dataframe['Position Noise']==pn
vm = dataframe['Velocity Noise']==vn
im = dataframe['Interp Noise']==interpn

best_mask = dm&sm&pm&im&vm
for k,data in dataframe[['Position Noise','Velocity Noise','Depth Noise','Stream Noise','Interp Noise']].drop_duplicates().iterrows():
	(pn,vn,dn,sn,interpn) = data
	mask = (dataframe['Position Noise'] == pn)&(dataframe['Velocity Noise']==vn)&(dataframe['Depth Noise']==dn)&(dataframe['Stream Noise']==sn)&(dataframe['Interp Noise']==interpn)
	holder = dataframe[mask].sort_values(by=['Toa Noise'])
	y = holder['Model Size']
	x = holder['Misfit List']
	plt.plot(x,y,'k',alpha=0.2)
for k,data in dataframe[best_mask][['Position Noise','Velocity Noise','Depth Noise','Stream Noise','Interp Noise']].drop_duplicates().iterrows():
	(pn,vn,dn,sn,interpn) = data
	mask = (dataframe['Position Noise'] == pn)&(dataframe['Velocity Noise']==vn)&(dataframe['Depth Noise']==dn)&(dataframe['Stream Noise']==sn)&(dataframe['Interp Noise']==interpn)
	holder = dataframe[mask].sort_values(by=['Toa Noise'])
	y = holder['Model Size']
	x = holder['Misfit List']
	plt.plot(x,y,'r',linewidth=7)
	plt.plot(x,y,'r',marker='o',markersize=12,linewidth=7)
plt.xscale('log')
point = (6.320554*10**4,6.123724)
plt.annotate('Best', point,
        xytext=(0.37, 0.1), textcoords='axes fraction',
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=25,
        horizontalalignment='left', verticalalignment='top')
plt.xlabel('Data Misfit, $T^2=\Vert\Sigma^{-1} j \Vert ^2$ ')
plt.ylabel('Model Norm, $\sum_{i=0}^n[x^s(t_i)-x^f(t_i)]^TP^{-1}_0[x^s(t_i)-x^f(t_i)]$')
plt.subplots_adjust(wspace = 0.3)
plt.savefig(plot_file_handler.out_file('Figure_9'))
plt.close()


file_handler = FilePathHandler(ROOT_DIR,'Tuning/Weddell')
plt.figure(figsize=(20,14))
plt.subplot(1,2,1)
dataframe = compile_tuning_dataframe_other(file_handler.out_file(''))

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
p_box = sns.boxplot(linewidth=6,hue_order=hue_order,**params)
handles, labels = p_box.get_legend_handles_labels()
plt.legend(handles[:4], labels[:4], title="Condition",
          handletextpad=0.5, columnspacing=1,ncol=7,bbox_to_anchor=(0.5, 1.13),
          loc="upper center", frameon=False)
plt.xticks(rotation=15)
p.set_yscale("log")
p.set_ylabel('Data Misfit, $T^2=\Vert\Sigma^{-1} j \Vert ^2$')
plt.subplot(1,2,2)
dataframe = compile_tuning_dataframe_toa(file_handler.out_file(''))
# holder = dataframe[dataframe['Misfit List']==dataframe['Misfit List'].min()]
# for k,data in holder[['Position Noise','Velocity Noise','Depth Noise','Stream Noise','Interp Noise']].iterrows():
# 	(pn,vn,dn,sn,interpn) = data

dm = dataframe['Depth Noise']==22500.0
sm = dataframe['Stream Noise']==900.0
pm = dataframe['Position Noise']==20.25
vm = dataframe['Velocity Noise']==0.5625
im = dataframe['Interp Noise']==14400.0

best_mask = dm&sm&pm&im&vm
for k,data in dataframe[['Position Noise','Velocity Noise','Depth Noise','Stream Noise','Interp Noise']].drop_duplicates().iterrows():
	(pn,vn,dn,sn,interpn) = data
	mask = (dataframe['Position Noise'] == pn)&(dataframe['Velocity Noise']==vn)&(dataframe['Depth Noise']==dn)&(dataframe['Stream Noise']==sn)&(dataframe['Interp Noise']==interpn)
	holder = dataframe[mask].sort_values(by=['Toa Noise'])
	y = holder['Model Size']
	x = holder['Misfit List']
	plt.plot(x,y,'k',alpha=0.2)
for k,data in dataframe[best_mask][['Position Noise','Velocity Noise','Depth Noise','Stream Noise','Interp Noise']].drop_duplicates().iterrows():
	(pn,vn,dn,sn,interpn) = data
	mask = (dataframe['Position Noise'] == pn)&(dataframe['Velocity Noise']==vn)&(dataframe['Depth Noise']==dn)&(dataframe['Stream Noise']==sn)&(dataframe['Interp Noise']==interpn)
	holder = dataframe[mask].sort_values(by=['Toa Noise'])
	y = holder['Model Size']
	x = holder['Misfit List']
	plt.plot(x,y,'r',linewidth=7)
	plt.plot(x,y,'r',marker='o',markersize=12,linewidth=7)

plt.xscale('log')
plt.yscale('log')
point = (6258.602989,6.508343)
plt.annotate('Best', point,
        xytext=(0.37, 0.1), textcoords='axes fraction',
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=25,
        horizontalalignment='left', verticalalignment='top')
plt.xlabel('Data Misfit, $\sum_{i=0}^n (\epsilon^s_i)^TR^{-1}_i\epsilon^s_i$')
plt.ylabel('Model Norm, $\sum_{i=0}^n[x^s(t_i)-x^f(t_i)]^TP^{-1}_0[x^s(t_i)-x^f(t_i)]$')
plt.subplots_adjust(wspace = 0.25)
plt.savefig(plot_file_handler.out_file('Figure_16'))
plt.close()
