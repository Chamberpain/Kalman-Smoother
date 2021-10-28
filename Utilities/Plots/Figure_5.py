from KalmanSmoother.Utilities.Observations import SourceArray
from KalmanSmoother.Utilities.Floats import FloatGen
from KalmanSmoother.Utilities.Filters import ObsHolder,Smoother,LeastSquares
import matplotlib
import matplotlib.pyplot as plt
from KalmanSmoother.Utilities.Utilities import KalmanPoint
from GeneralUtilities.Plot.Cartopy.eulerian_plot import BaseCartopy
import cartopy.crs as ccrs

class ParticleCartopy(BaseCartopy):
	def __init__(self,x_list,y_list,*args,pad=0.2,**kwargs):
		super().__init__(*args,**kwargs)          
		llcrnrlon=min(x_list)-pad
		llcrnrlat=min(y_list)-pad
		urcrnrlon=max(x_list)+pad
		urcrnrlat=max(y_list)+pad
		self.ax.set_extent([llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat], crs=ccrs.PlateCarree())
		self.finish_map()

font = {'family' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)


sources = SourceArray()
sources.set_offset(0)
sources.set_drift(0)
sources.set_location(KalmanPoint(-64,-23.5))
a = FloatGen('test',sources,1,1,0.3,15,20,5)

q = ObsHolder(a)
LeastSquares(a,sources,q)
ls_x,ls_y = zip(*[(x.longitude,x.latitude) for x in a.pos])
Smoother(a,sources,q)

exact_x,exact_y = zip(*[(x.longitude,x.latitude) for x in a.exact_pos])

kalman_x,kalman_y = zip(*[(x.longitude,x.latitude) for x in a.kalman_pos])
smoother_x,smoother_y = zip(*[(x.longitude,x.latitude) for x in a.pos])

sources_x = []
sources_y = []
for name,source in sources.array.items():
	sources_x.append(source.position.longitude)
	sources_y.append(source.position.latitude)

x_list = smoother_x+kalman_x+ls_x
y_list = smoother_y+kalman_y+ls_y
XX,YY,ax = ParticleCartopy(x_list,y_list).get_map()


ax.scatter(sources_x,sources_y,c='r',label='Sound Sources')
ax.scatter(exact_x,exact_y,marker='D',c='yellow',label='True Position',zorder=11,alpha=0.8)
ax.plot(exact_x,exact_y,'--',c='yellow',alpha=0.8)
ax.scatter(ls_x,ls_y,marker='D',c='blue',label='Least Squares',alpha=0.5)
ax.plot(ls_x,ls_y,'--',c='blue',alpha=0.5)
ax.scatter(kalman_x,kalman_y,marker='D',c='orangered',label='Kalman Filter',alpha=0.5)
ax.plot(kalman_x,kalman_y,'--',c='orangered',alpha=0.5)
ax.scatter(smoother_x,smoother_y,marker='D',c='green',label='Kalman Smoother',alpha=0.5)
ax.plot(smoother_x,smoother_y,'--',c='green',alpha=0.5)

plt.legend()
plt.savefig('Figure_5')
plt.close()