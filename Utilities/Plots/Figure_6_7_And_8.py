import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from KalmanSmoother.Utilities.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
import pandas as pd
import seaborn as sns

file_handler = FilePathHandler(ROOT_DIR,'FinalFloatsPlot')
x_label_dict = {'GPS Number':'GPS Number (Observations)','TOA Error':'TOA Error (Seconds)','TOA Number':'TOA Number (Sources)'}

gps_list = []
smoother_pos_list = []
kalman_pos_list = []
ls_pos_list = []
smoother_toa_list = []
kalman_toa_list = []
ls_toa_list = []
toa_error_list = []
toa_number_list = []
percent_list = []
unique_percent_list = [0.1,0.3,0.7]
folder = FilePathHandler(ROOT_DIR,'Particle').out_file('')
for token in os.listdir(folder):
	if not token.endswith('.npy'):
		continue
	toa_error,toa_number,percent,gps_number, \
	smoother_pos_misfit,kalman_pos_misfit,ls_pos_misfit, \
	smoother_toa_misfit,kalman_toa_misfit,ls_toa_misfit = np.load(folder+token,allow_pickle=True)
	gps_list.append(gps_number)
	smoother_pos_list.append(smoother_pos_misfit)
	kalman_pos_list.append(kalman_pos_misfit)
	ls_pos_list.append(ls_pos_misfit)
	smoother_toa_list.append(smoother_toa_misfit)
	kalman_toa_list.append(kalman_toa_misfit)
	ls_toa_list.append(ls_toa_misfit)
	toa_error_list.append(toa_error)
	toa_number_list.append(toa_number)
	percent_list.append(percent)



dataframe = pd.concat([pd.DataFrame({'GPS Number':gps_list,'Pos Error':smoother_pos_list,'TOA Error':smoother_toa_list,
	'Type':'Smoother','TOA Uncertainty':toa_error_list,'TOA Number':toa_number_list,'SNR':percent_list}),
	pd.DataFrame({'GPS Number':gps_list,'Pos Error':kalman_pos_list,'TOA Error':kalman_toa_list,
		'Type':'Kalman','TOA Uncertainty':toa_error_list,'TOA Number':toa_number_list,'SNR':percent_list}),
	pd.DataFrame({'GPS Number':gps_list,'Pos Error':ls_pos_list,'TOA Error':ls_toa_list,
		'Type':'Least Squares','TOA Uncertainty':toa_error_list,'TOA Number':toa_number_list,'SNR':percent_list})])

dataframe = dataframe[dataframe.SNR<1]
dataframe.SNR[dataframe.SNR==0.7]='Low'
dataframe.SNR[dataframe.SNR==0.3]='Med'
dataframe.SNR[dataframe.SNR==0.1]='High'
dataframe = dataframe.reset_index()
SNR_ranking = ['High','Med','Low']

sns.set_theme(style="whitegrid")
sns.set(font_scale=2.2)
ax = sns.relplot(x="GPS Number", y="Pos Error",
                hue="Type",style="Type",s=20,col="SNR",col_order=['High','Med','Low'],alpha=0.5,
                data=dataframe,hue_order=['Least Squares','Kalman','Smoother']).set_titles('')
ax.set(yscale='log')
ax.axes[0][0].annotate('a', xy = (0.8,0.6),xycoords='axes fraction',zorder=10,size=52,bbox=dict(boxstyle="round", fc="0.8"),)
ax.axes[0][0].annotate('High', xy = (0.3,0.05),xycoords='axes fraction',zorder=10,size=52)

ax.axes[0][1].annotate('b', xy = (0.8,0.6),xycoords='axes fraction',zorder=10,size=52,bbox=dict(boxstyle="round", fc="0.8"),)
ax.axes[0][1].annotate('Med', xy = (0.3,0.05),xycoords='axes fraction',zorder=10,size=52)

ax.axes[0][2].annotate('c', xy = (0.8,0.6),xycoords='axes fraction',zorder=10,size=52,bbox=dict(boxstyle="round", fc="0.8"),)
ax.axes[0][2].annotate('Low', xy = (0.3,0.05),xycoords='axes fraction',zorder=10,size=52)

ax.set(ylabel='Error (km)')

leg = ax._legend
leg.set_bbox_to_anchor([0.71,0.85])
for lh in leg.legendHandles: 
    lh.set_alpha(1)
    lh.set_sizes([500])
ax.fig.set_size_inches(13,10)
ax.set(ylim=(0.1, 100))
plt.tight_layout()
plt.savefig(file_handler.out_file('Figure_6'))
plt.close()


sns.set_theme(style="whitegrid")
sns.set(font_scale=2.2)
ax = sns.lmplot(x="TOA Uncertainty", y="Pos Error",
                hue="Type",col="SNR",line_kws={'lw':10},
                scatter_kws={'alpha':0.07},col_order=['High','Med','Low'],hue_order=['Least Squares','Kalman','Smoother']
                ,data=dataframe,facet_kws=dict(legend_out=False)).set_titles('')
ax.set(yscale='log')
ax.set(ylim=(0.1, 100))
ax.axes[0][0].annotate('a', xy = (0.1,0.80),xycoords='axes fraction',zorder=10,size=52,bbox=dict(boxstyle="round", fc="0.8"),)
ax.axes[0][0].annotate('High', xy = (0.3,0.05),xycoords='axes fraction',zorder=10,size=52)

ax.axes[0][1].annotate('b', xy = (0.1,0.80),xycoords='axes fraction',zorder=10,size=52,bbox=dict(boxstyle="round", fc="0.8"),)
ax.axes[0][1].annotate('Med', xy = (0.3,0.05),xycoords='axes fraction',zorder=10,size=52)

ax.axes[0][2].annotate('c', xy = (0.1,0.80),xycoords='axes fraction',zorder=10,size=52,bbox=dict(boxstyle="round", fc="0.8"),)
ax.axes[0][2].annotate('Low', xy = (0.3,0.05),xycoords='axes fraction',zorder=10,size=52)
ax.set(ylabel='Error (km)')

h,l = ax.axes[0][0].get_legend_handles_labels()
ax.axes[0][0].legend_.remove()
leg = ax.fig.legend(h,l, ncol=3,loc='upper center')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
    lh.set_sizes([500])
ax.fig.set_size_inches(15,12)
plt.tight_layout()
plt.savefig(file_handler.out_file('Figure_7'))
plt.close()

f, ax = plt.subplots()
f.set_size_inches(15,15)
sns.despine(bottom=True, left=True)
sns.set_theme(style="whitegrid")
sns.set(font_scale=2.2)
p = sns.stripplot(x="TOA Number", y="Pos Error",
                hue="Type",hue_order=['Least Squares','Kalman','Smoother'],dodge=True, alpha=.25, zorder=1,
                data=dataframe)
sns.pointplot(x="TOA Number", y="Pos Error", hue="Type",
              data=dataframe,hue_order=['Least Squares','Kalman','Smoother'],dodge=.8 - .8 / 3,
              join=False, palette="dark",
              markers="d", scale=2, ci=None)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[3:], labels[3:], title="Type",
          handletextpad=0, columnspacing=1,
          loc="upper right", ncol=3, frameon=True)
ax.set_ylabel('Error (km)')
ax.set_xticklabels(['0','1','2','3','4','5'])
plt.yscale('log')
plt.tight_layout()
plt.savefig(file_handler.out_file('Figure_8'))
plt.close()
