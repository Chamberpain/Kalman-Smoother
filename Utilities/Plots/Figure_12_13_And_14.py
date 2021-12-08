import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from GeneralUtilities.Filepath.instance import FilePathHandler
from KalmanSmoother.Utilities.__init__ import ROOT_DIR
from KalmanSmoother.Utilities.Floats import WeddellAllFloats,DIMESAllFloats,AllFloats
from KalmanSmoother.Utilities.Filters import ObsHolder, Smoother
import copy
import datetime
from geopy.distance import GreatCircleDistance
from KalmanSmoother.Utilities.DataLibrary import dimes_position_process,dimes_velocity_process,dimes_depth_noise,dimes_stream_noise,dimes_toa_noise,dimes_interp_noise
from KalmanSmoother.Utilities.DataLibrary import weddell_position_process,weddell_velocity_process,weddell_depth_noise,weddell_stream_noise,weddell_toa_noise,weddell_interp_noise

file_handler = FilePathHandler(ROOT_DIR,'FinalFloatsPlot')

class SeabornFig2Grid():
    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())



def obs_date_diff_list(dummy):
    dates = np.sort(dummy.toa.date)
    date_diff_list = []
    toa_error_list = []
    toa_number_list = []
    pos_date = sorted(np.unique(dummy.pos_date))
    for date in dates[1:]:
        try:
            idx = pos_date.index(date)
            dummy.pos[idx]
            dummy.clock.set_date(date)
            AllFloats.sources.set_date(date)
            toa_list = dummy.toa.return_data()
            if not toa_list:
                continue
            dates_holder = dates[dates<date]
            diff = (date-max(dates_holder)).days
            date_diff_list.append(diff)

            toa_number_list.append(len(toa_list))
            holder = []
            for toa,source in toa_list:
                detrend_toa = dummy.clock.detrend_offset(toa)
                detrend_toa = source.clock.detrend_offset(detrend_toa)
                dist = GreatCircleDistance(source.position,dummy.pos[idx]).km
                dist_toa = source.toa_from_dist(dist)
                holder.append(abs(detrend_toa-dist_toa))
            toa_error_list.append(np.mean(holder))
        except ValueError:
            continue
        except IndexError:
            continue
    return (date_diff_list,toa_error_list,toa_number_list)

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
weddell_list = copy.deepcopy(WeddellAllFloats.list)

DIMESAllFloats.list=[]
all_floats = DIMESAllFloats()
for idx,dummy in enumerate(all_floats.list):
    print(idx)
    dummy.toa.set_observational_uncertainty(dimes_toa_noise)
    dummy.stream.set_observational_uncertainty(dimes_stream_noise)
    dummy.depth.set_observational_uncertainty(dimes_depth_noise)
    obs_holder = ObsHolder(dummy)
    smooth =Smoother(dummy,all_floats.sources,obs_holder,process_position_noise=dimes_position_process,process_vel_noise =dimes_velocity_process)
dimes_list = copy.deepcopy(DIMESAllFloats.list)
floatlist = weddell_list+dimes_list

data_df_list = []
float_df_list = []
for x in floatlist:
    try:
        toa_percent = x.percent_obs()
        date_diff_list,toa_error_list,toa_number_list = obs_date_diff_list(x)
        toa_number_list = np.array(toa_number_list)
        toa_number_list[toa_number_list>3]=3
        mean_error = np.mean(toa_error_list)
        print(x.floatname)
        data_df = pd.DataFrame({'Date Diff':date_diff_list,'Error':toa_error_list,
            'TOA Number':toa_number_list,'Float Type':x.type})
        float_df = pd.DataFrame({'TOA Percent':[(toa_percent*100)],
            'Error':[mean_error],'Float Type':[x.type]})
        data_df_list.append(data_df)
        float_df_list.append(float_df)
    except AttributeError:
        continue
data_df = pd.concat(data_df_list)

float_df = pd.concat(float_df_list)
float_df = float_df[float_df['TOA Percent']>5]

sns.set_theme(style="whitegrid")
sns.set(font_scale=2.2)
ax = sns.lmplot(x='Date Diff', y="Error",
                hue="Float Type",line_kws={'lw':6},
                scatter_kws={'alpha':0.55},x_ci='sd',
                data=data_df)
ax.set(yscale='log')
ax.set(xscale='log')
ax.set(ylabel='TOA Error (s)')
ax.set(xlabel='Days Since Last Positioned')

leg = ax._legend
leg.set_bbox_to_anchor([0.8,0.3])
for lh in leg.legendHandles: 
    lh.set_alpha(1)
    lh.set_sizes([500])
ax.fig.set_size_inches(12,12)
plt.ylim([0.1,data_df.Error.max()+10])
plt.tight_layout()
plt.savefig(file_handler.out_file('Figure_12'))
plt.close()

f = plt.figure(figsize=(15,15))
ax = f.add_subplot(1,1,1)
sns.set_theme(style="whitegrid")
sns.set(font_scale=2.2)
sns.stripplot(x="TOA Number", y="Error",
                hue='Float Type',dodge=True, alpha=.25, zorder=1,
                data=data_df)
sns.pointplot(x="TOA Number", y="Error", hue='Float Type',
              data=data_df, dodge=.8 - .8 / 3,
              join=False, palette="dark",
              markers="d", scale=2, ci=None)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[2:], labels[2:], title="Type",
          handletextpad=0, columnspacing=1,
          loc="upper right", ncol=3, frameon=True)
ax.set_xticklabels(['1','2','3+'])
ax.set_ylabel('TOA Error (s)')
ax.set_xlabel('Sound Sources Heard')
plt.ylim([0.01,data_df.Error.max()+10])

plt.yscale('log')
plt.tight_layout()
plt.savefig(file_handler.out_file('Figure_13'))
plt.close()

sns.set_theme(style="whitegrid")
sns.set(font_scale=2.2)
ax = sns.lmplot(
    data = float_df,
    x='TOA Percent',y='Error',hue='Float Type',line_kws={'lw':6},
                scatter_kws={'alpha':0.55},x_ci='sd')
ax.set(yscale='log')
ax.set(ylabel='Mean TOA Error (s)')
leg = ax._legend
leg.set_bbox_to_anchor([0.85,0.9])
for lh in leg.legendHandles: 
    lh.set_alpha(1)
    lh.set_sizes([500])
plt.ylim([float_df.Error.min(),110])
ax.fig.set_size_inches(12,12)
plt.tight_layout()
plt.savefig(file_handler.out_file('Figure_14'))
plt.close()
