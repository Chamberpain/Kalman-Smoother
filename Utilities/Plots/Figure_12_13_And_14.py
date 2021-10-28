import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from Projects.KalmanSmoother.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
from KalmanSmoother.Utilities.__init__ import ROOT_DIR as ROOT_DIR_PLOT
file_handler_dimes = FilePathHandler(ROOT_DIR,'Tuning/Dimes')

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


dimes_date_diff = np.load(base_dir+'dimes_date_diff.npy').tolist()
dimes_error = np.load(base_dir+'dimes_kalman_error_list.npy').tolist()
dimes_toa = np.load(base_dir+'dimes_toa_number.npy').tolist()
dimes_percent = np.load(base_dir+'dimes_percent.npy').tolist()
dimes_mean = np.load(base_dir+'dimes_mean.npy').tolist()


weddell_date_diff = np.load(base_dir+'weddell_date_diff.npy').tolist()
weddell_error = np.load(base_dir+'weddell_kalman_error_list.npy').tolist()
weddell_toa = np.load(base_dir+'weddell_toa_number.npy').tolist()
weddell_percent = np.load(base_dir+'weddell_percent.npy').tolist()
weddell_mean = np.load(base_dir+'weddell_mean.npy').tolist()

idx = -1
dimes_dataframe = pd.DataFrame({'Date Diff':dimes_date_diff[:idx],'Error':dimes_error[:idx],'TOA Number':dimes_toa[:idx],'Float Type':'DIMES'})
weddell_dataframe = pd.DataFrame({'Date Diff':weddell_date_diff[:idx],'Error':weddell_error[:idx],'TOA Number':weddell_toa[:idx],'Float Type':'Weddell'})
dataframe = pd.concat([dimes_dataframe,weddell_dataframe])
dataframe = dataframe[dataframe['TOA Number']<8]
dataframe = dataframe.reset_index()


sns.set_theme(style="whitegrid")
sns.set(font_scale=2.2)
ax = sns.lmplot(x='Date Diff', y="Error",
                hue="Float Type",line_kws={'lw':6},
                scatter_kws={'alpha':0.55},
                data=dataframe)
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
plt.ylim([0.1,dataframe.Error.max()+10])
plt.tight_layout()
plt.savefig(file_handler.out_file('date_diff_error'))
plt.close()


f = plt.figure(figsize=(15,15))
ax = f.add_subplot(1,1,1)
sns.set_theme(style="whitegrid")
sns.set(font_scale=2.2)
sns.stripplot(x="TOA Number", y="Error",
                hue='Float Type',dodge=True, alpha=.25, zorder=1,
                data=dataframe)
sns.pointplot(x="TOA Number", y="Error", hue='Float Type',
              data=dataframe, dodge=.8 - .8 / 3,
              join=False, palette="dark",
              markers="d", scale=2, ci=None)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[2:], labels[2:], title="Type",
          handletextpad=0, columnspacing=1,
          loc="upper right", ncol=3, frameon=True)
ax.set_xticklabels(['0','1','2','3','4','5'])
ax.set_ylabel('TOA Error (s)')
ax.set_xlabel('Sound Sources Heard')
plt.ylim([0.01,dataframe.Error.max()+10])

plt.yscale('log')
plt.tight_layout()
plt.savefig(file_handler.out_file('toa_number_error'))
plt.close()

weddell_dataframe = pd.DataFrame({'TOA Percent':[x*100 for x in weddell_percent],'Error':weddell_mean,'Float Type':'Wedell'})
weddell_dataframe = weddell_dataframe[weddell_dataframe['TOA Percent']>5]
dimes_dataframe = pd.DataFrame({'TOA Percent':[x*100 for x in dimes_percent],'Error':dimes_mean,'Float Type':'DIMES'})
dataframe = pd.concat([dimes_dataframe,weddell_dataframe])


sns.set_theme(style="whitegrid")
sns.set(font_scale=2.2)
ax = sns.lmplot(
    data = dataframe,
    x='TOA Percent',y='Error',hue='Float Type',line_kws={'lw':6},
                scatter_kws={'alpha':0.55})
ax.set(yscale='log')
ax.set(ylabel='Mean TOA Error (s)')
leg = ax._legend
leg.set_bbox_to_anchor([0.8,0.7])
for lh in leg.legendHandles: 
    lh.set_alpha(1)
    lh.set_sizes([500])
ax.fig.set_size_inches(12,12)
plt.tight_layout()
plt.savefig(file_handler.out_file('toa_percent_error'))
plt.close()
