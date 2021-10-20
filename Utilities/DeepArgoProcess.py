import scipy.io
import pickle

file = '../Data/deep_argo/Paul_Data.mat'
mat = scipy.io.loadmat(file)

dict_41 = {'lat':mat['lat41'],
'lon':mat['lon41'],
'pos':mat['pos41'],
'mpr':mat['mpr41'],
'lat_g':mat['lat_g41'],
'lon_g':mat['lon_g41']}

dict_42 = {'lat':mat['lat42'],
'lon':mat['lon42'],
'pos':mat['pos42'],
'mpr':mat['mpr42'],
'lat_g':mat['lat_g42'],
'lon_g':mat['lon_g42']}

for filename,dictionary in [('../Data/deep_argo/41.pkl',dict_41),('../Data/deep_argo/42.pkl',dict_42)]:
	with open(filename, 'wb') as f:
		pickle.dump(dictionary, f)
		f.close()