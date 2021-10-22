from KalmanSmoother.Utilities.ArtificialFloats import FloatGen
from KalmanSmoother.Utilities.Observations import SourceArray
from KalmanSmoother.Utilities.Filters import ObsHolder,Smoother,LeastSquares
import matplotlib.pyplot as plt

sources = SourceArray()
sources.set_offset(0)
sources.set_drift(0)
a = FloatGen('test',sources,0.3,1,1,1000,50,1)
q = ObsHolder(a)
LeastSquares(a,sources,q)
Smoother(a,sources,q)

exact_x,exact_y = zip(*[(x.longitude,x.latitude) for x in a.exact_pos])
ls_x,ls_y = zip(*[(x.longitude,x.latitude) for x in a.ls_pos])
kalman_x,kalman_y = zip(*[(x.longitude,x.latitude) for x in a.kalman_pos])
smoother_x,smoother_y = zip(*[(x.longitude,x.latitude) for x in a.pos])


plt.scatter(exact_x,exact_y,label='Exact')
plt.scatter(ls_x,ls_y,label='Least Squares')
plt.scatter(kalman_x,kalman_y,label='Kalman')
plt.scatter(smoother_x,smoother_y,label='Smoother')
plt.legend()
plt.show()
