import pandas as pd
import numpy as np
from plot_utilities.eulerian_plot import Basemap,basemap_setup


lllon = -60
urlon = 35
lllat = -75
urlat = -50

m = basemap_setup(urlat,lllat,urlon,lllon,0,spacing=15,aspect=False)

float_list = np.load('weddell_float_list.npy')
traj_df = pd.read_pickle('../../argo_traj_box/traj_df.pickle')



for float_ in float_list:
	df_holder = traj_df[traj_df.Cruise==str(float_)]