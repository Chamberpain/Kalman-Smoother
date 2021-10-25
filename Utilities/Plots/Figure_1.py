from GeneralUtilities.Plot.Cartopy.regional_plot import DrakePassageCartopy,WeddellSeaCartopy
from KalmanSmoother.Utilities.Floats import AllFloats
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from GeneralUtilities.Filepath.instance import FilePathHandler
from KalmanSmoother.Utilities.__init__ import ROOT_DIR

file_handler = FilePathHandler(ROOT_DIR,'FinalFloatsPlot')



def sound_source_number_plot():
	def pie_plot_list(list_):
		calc_array = np.array(list_)
		calc_array[calc_array>4]=4
		list_ = calc_array.tolist()
		return_list = []
		for num in np.unique(list_):
			np.where(num==np.array(list_))
			frac = len(np.where(num==np.array(list_))[0])/float(len(list_))
			return_list.append((num,frac))
		return return_list

	all_weddell = AllFloats('Weddell')
	all_dimes = AllFloats('DIMES')
	label = ['No Source','1 Source','2 Sources','3 Sources','4+ Sources']
	lats = np.arange(-90,91,1)
	lons = np.arange(-180,181,1)

	fig = plt.figure(figsize=(12,8))
	ax1 = fig.add_subplot(2,2,1, projection=ccrs.PlateCarree())
	plt.annotate('a', xy = (0.15,0.9),xycoords='axes fraction',zorder=10,size=16,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2 = fig.add_subplot(2,2,2, projection=ccrs.PlateCarree())	

	XX,YY,ax1 = DrakePassageCartopy(lat_grid=lats,lon_grid=lons,ax=ax1).get_map()
	ax1.streamline_plot()
	cf = ax1.bathy()
	lat_list = []
	lon_list = []
	for _,source in all_dimes.sources.array.items():
		if source.mission=='DIMES':
			lat_list.append(source.position.latitude)
			lon_list.append(source.position.longitude)
	ax1.scatter(lon_list,lat_list,c='r')


	XX,YY,ax2 = WeddellSeaCartopy(lat_grid=lats,lon_grid=lons,ax=ax2).get_map()
	ax2.streamline_plot()
	ax2.bathy()
	lat_list = []
	lon_list = []
	for _,source in all_weddell.sources.array.items():
		if source.mission=='Weddell':
			lat_list.append(source.position.latitude)
			lon_list.append(source.position.longitude)
	ax2.scatter(lon_list,lat_list,c='r')
	plt.annotate('b', xy = (0.15,0.9),xycoords='axes fraction',zorder=10,size=16,bbox=dict(boxstyle="round", fc="0.8"),)

	fig.colorbar(cf,ax=[ax1,ax2],pad=.1,label='Depth (km)',location='bottom',shrink=0.5)

	ax3 = fig.add_subplot(2,2,3)	
	dummy,frac = zip(*pie_plot_list(all_dimes.all_toa_number()))

	wedges, texts, autotexts = ax3.pie(frac,autopct='%1.1f%%', textprops={'fontsize':15})
	plt.annotate('c', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=16,bbox=dict(boxstyle="round", fc="0.8"),)

	ax4 = fig.add_subplot(2,2,4)	
	dummy,frac = zip(*pie_plot_list(all_weddell.all_toa_number()))
	ax4.pie(frac,autopct='%1.1f%%', textprops={'fontsize':15})	
	ax4.legend(wedges, label,
          title="Sources Heard",
          loc="upper right",
          bbox_to_anchor=(0.0, 0.8),
          prop={'size': 16})
	plt.annotate('d', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=16,bbox=dict(boxstyle="round", fc="0.8"),)

	plt.savefig(file_handler.out_file('Figure_1'))
	plt.close()
	