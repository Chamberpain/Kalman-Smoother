class AllFloatsPlot(AllFloats):
	def plot_soso_errors_by_date(self):
		for _,soso in self.sources.array.items():
			if not soso.error_dataframe.empty:
				soso.plot_error_dataframe('Date')
				plt.show()

	def plot_soso_errors_by_dist(self):
		for _,soso in self.sources.array.items():
			if not soso.error_dataframe.empty:
				soso.plot_error_dataframe('Dist')
				print(soso.speed_of_sound)
				plt.show()

	def plot_gps_error_hist(self):
		kalman_error,smooth_error=self.all_gps_error_list()
		kalman_error = np.array(kalman_error)
		smooth_error = np.array(smooth_error)
		kalman_error = kalman_error[kalman_error<0.5]
		smooth_error = smooth_error[smooth_error<0.5]
		bins=np.histogram(np.hstack((smooth_error,kalman_error)), bins=50)[1]
		plt.figure()
		plt.hist(smooth_error,bins,label = 'Kalman Smoother',alpha=0.2)
		plt.hist(kalman_error,bins,label = 'Kalman Filter',alpha=0.2)
		plt.gca().set_yscale("log")
		plt.legend(prop={'size': 10})
		plt.title('Histogram of GPS Error')
		plt.savefig('gps_error')
		plt.close()
	def all_traj_plot(self,title):
		for all_dummy in self.list:
			try:
				all_dummy.traj_plot()
			except AttributeError:
				continue
		plt.savefig(title)
		plt.close

	def all_gps_error_list(self):
		kalman_list = []
		smoother_list = []
		for label,dummy_list in [('kalman',kalman_list),('smoother',smoother_list)]:
			for _ in self.list:
				dummy_1 = [item for sublist in _.gps.dx_error[label] for item in sublist]
				dummy_2 = [item for sublist in _.gps.dy_error[label] for item in sublist]
				dummy_list += [np.sqrt(_[0]**2+_[1]**2) for _ in list(zip(dummy_1,dummy_2))]
		return (kalman_list,smoother_list)

	def all_gps_date_list(self):
		dummy_list = []
		for _ in self.list:
			dummy_list+=_.gps.date
		return dummy_list

	def all_toa_error_list(self,error_type):
		dummy_list = []
		for _ in self.sources.array.items():
			dummy_list+=_[1].error[error_type]
		dummy_list = [item for sublist in dummy_list for item in sublist]
		return dummy_list

	def all_speed_list(self):
		dummy_list = []
		for _ in self.list:		
			dummy_list += (np.array([x.magnitude for x in np.diff(_.pos)])/np.array([x.days for x in np.diff(_.pos_date)])).tolist()
		return dummy_list


	def all_toa_date_list(self):
		dummy_list = []
		for _ in self.list:
			dummy_list+=_.toa.date
		return dummy_list

	def all_toa_number(self):
		dummy_list = []
		for _ in self.list:
			total_len = (_.gps.date[-7]-_.gps.date[6]).days
			for dummy_toa in _.toa.obs:
				dummy_list+=[len(dummy_toa)]
			nothing_heard = total_len-len(_.toa.obs)
			dummy_list+=[0]*nothing_heard
		return dummy_list

	def all_toa_list(self):
		dummy_list = []
		for _ in self.list:
			for dummy_toa in _.toa.obs:
				dummy_list+=dummy_toa
		return dummy_list


class WeddellAllFloatsPlot(WeddellAllFloats):
	def weddell_plot(self):
		plt.figure()
		lllon = -60
		urlon = 35
		lllat = -75
		urlat = -50
		m = Basemap.auto_map(urlat,lllat,urlon,lllon,0,spacing=15,aspect=False)
		first_lat_list= []
		first_lon_list = []
		last_lat_list = []
		last_lon_list = []

		for k,all_dummy in enumerate(self.list):	
			lat,lon = list(zip(*[(_.lat.decimal_degree,_.lon.decimal_degree) for _ in all_dummy.pos]))
			first_lat_list.append(lat[0])
			first_lon_list.append(lon[0])
			last_lat_list.append(lat[-1])
			last_lon_list.append(lon[-1])

			A,B,angle = list(zip(*[all_dummy.uncertainty_ellipse(_) for _ in all_dummy.pos_date]))
			m.scatter(lon,lat,s=1.5,c=np.array(A)*11.3,latlon=True,alpha=0.2)
		lat_list = []
		lon_list = []
		for _,source in self.sources.array.items():
			if source.mission=='Weddell':
				lat_list.append(source.position.latitude)
				lon_list.append(source.position.longitude)
		plt.colorbar(label='Uncertainty (km)')
		m.scatter(first_lon_list,first_lat_list,s=13,c='m',latlon=True,marker='^')
		m.scatter(last_lon_list,last_lat_list,s=13,c='m',latlon=True,marker='s')
		m.scatter(lon_list,lat_list,s=13,c='r',latlon=True)
		plt.savefig('weddell_plot')
		plt.close()
		plt.figure()
		dummy_list = self.all_toa_error_list('smoother')
		plt.hist(dummy_list,bins=75)
		plt.xlim([-20,20])
		plt.xlabel('Misfit (seconds)', fontsize=18)
		plt.savefig('Weddell_stats')

class DimesAllFloatsPlot(DimesAllFloats):
	def all_DIMES_error_list(self):
		dummy_list = []
		for _ in self.list:
			dummy_list+=_.toa.abs_error
		return dummy_list

	def all_DIMES_artoa_speed_list(self):
		dummy_list = []
		for _ in self.list:
			df = _.trj_df.drop_duplicates(subset=['Lat','Lon'])
			pos_list = [KalmanPoint(x_[0],x_[1]) for x_ in list(zip(df['Lat'],df['Lon']))] 
			dummy_list+=(np.array([x.magnitude for x in np.diff(pos_list)])/np.array([x.days for x in df.diff()['Date'][1:]])).tolist()
		return dummy_list

	def all_DIMES_kalman_speed_list(self):
		dummy_list = []
		for _ in self.list:
			mask = np.isin(_.pos_date,_.trj_date)
			dummy_pos = np.array(_.pos)[mask]
			dummy_pos_date = np.array(_.pos_date)[mask]
			dummy_list += (np.array([x.magnitude for x in np.diff(dummy_pos)])/np.array([x.days for x in np.diff(dummy_pos_date)])).tolist()

	def DIMES_speed_hist(self):
		kalman_speed = self.all_speed_list()
		dimes_speed = self.all_DIMES_artoa_speed_list()

		bins = np.linspace(0,41,100)
		plt.hist(kalman_speed,bins=bins,label='Kalman',alpha=0.2)
		plt.hist(dimes_speed,bins=bins,label='DIMES',alpha=0.2)
		plt.legend()
		plt.savefig('dimes_vel_stats')
		plt.close()

	def dimes_all_plot(self):
		lon_0 =0 
		for k,all_dummy in enumerate(self.list):
			print(k)
			if all_dummy.floatname ==837:
				continue
			if int(all_dummy.floatname)>868:
				print(all_dummy.floatname)
				continue
			if k == 0:
				lllon = -115
				urlon = -40
				lllat = -70
				urlat = -45
				m = Basemap(projection='cea',llcrnrlat=lllat,urcrnrlat=urlat,\
					llcrnrlon=lllon,urcrnrlon=urlon,resolution='l',lon_0=lon_0,\
					fix_aspect=False)
				# m.drawmapboundary(fill_color='darkgray')
				m.fillcontinents(color='dimgray')
				m.drawmeridians(np.arange(0,360,10),labels=[1,1,1,1])
				m.drawparallels(np.arange(-90,90,10),labels=[1,1,1,1])

			try:
				pos = np.array(all_dummy.pos)[np.isin(all_dummy.pos_date,all_dummy.trj_df['Date'].tolist())]
				lat,lon = list(zip(*[(_.latitude,_.longitude) for _ in pos]))
				m.plot(lon,lat,color='g',alpha=0.2,latlon=True)
				m.plot(all_dummy.trj_df['Lon'].tolist(),all_dummy.trj_df['Lat'].tolist(),color='b',alpha=0.2,latlon=True)
			except AttributeError:
				continue
		lats = []
		lons = []
		for _ in self.sources.array.items():
			if _[1].mission == 'DIMES':
				lat = _[1].position.latitude
				lon = _[1].position.longitude
				lats.append(lat)
				lons.append(lon)
		x,y = m(lons,lats)
		m.scatter(x,y,color='r',zorder=10)
		plt.savefig('all_dimes_plot')

	def all_dimes_error_hist(self):
		from scipy import stats		
		kalman_speed = self.all_speed_list()
		dimes_speed = self.all_DIMES_artoa_speed_list()
		trj_list = []
		trj_error_list = []
		kalman_list = []
		kalman_error_list = []
		kalman_date_list = []
		float_list = []
		diff_list = []
		long_diff_list = []
		std_list = []
		mean_list = []
		percent_list = []
		date_diff_list = []
		toa_number = []
		two_d_hist_diff = []
		for k,all_dummy in enumerate(self.list):
			print(all_dummy.floatname)
			try:
				trj_date = np.unique(all_dummy.trj_date)
				dummy_error,toa_error_list,dist_list,soso_list,date_return_list,pos_dist_list = all_dummy.toa.calculate_error_list(all_dummy.pos,all_dummy.pos_date)
				kalman_date_list+=date_return_list
				float_list+=[k]*len(date_return_list)
				kalman_error_list+=dummy_error
				mean_list.append(np.mean(np.abs(dummy_error)))
				std_list.append(np.std(dummy_error))
				percent_list.append(all_dummy.percent_obs())
				trj_error,toa_error_list,dist_list,soso_list,date_return_list,trj_dist_list = all_dummy.toa.calculate_error_list(all_dummy.trj_pos,trj_date)
				trj_error_list+=trj_error
				mask_trj = np.isin(trj_date,all_dummy.pos_date)
				pos = np.array(all_dummy.trj_pos)[mask_trj]
				mask_smooth = np.isin(all_dummy.pos_date,trj_date)
				smooth_pos = np.array(all_dummy.pos)[mask_smooth]
				diff = [(x[0]-x[1]).magnitude for x in list(zip(pos,smooth_pos))]
				diff_list+=diff
				two_d_hist_diff += np.array(diff)[np.isin(trj_date,all_dummy.toa.date)].tolist()
				mask = np.isin(all_dummy.toa.date,trj_date)
				date_diff_list += ([1]+[x.days for x in np.diff(np.array(all_dummy.toa.date)[mask])])
				toa_number += [len(x) for x in all_dummy.toa.obs]

			except:
				continue

		N = 1
		label_pad = 0
		x_date = np.convolve(date_diff_list,np.ones(N)/N, mode = 'valid')
		x_toa = np.convolve(toa_number,np.ones(N)/N, mode = 'valid')
		y = np.convolve(two_d_hist_diff,np.ones(N)/N, mode = 'valid')
		plt_date_diff = []

		bins = np.linspace(min(x_date),max(x_date),14)
		bin_means, bin_edges, binnumber = stats.binned_statistic(x_date,
			y, statistic='median', bins=bins)
		bin_std, bin_edges, binnumber = stats.binned_statistic(x_date,
			y, statistic='median', bins=bins)
		bin_width = (bin_edges[1] - bin_edges[0])
		bin_centers = bin_edges[1:] - bin_width/2
		mask = ~np.isnan(bin_means)

		# for i,date_diff_token in enumerate(np.unique(plt_date_diff_list)):
		# 	mask = (np.array(plt_date_diff_list)==date_diff_token)
		# 	plt_date_diff.append(plt_two_d_hist_diff[mask].mean())
		slope, intercept, r_value, p_value, std_err = stats.linregress(x_date, y)
		print('slope of date difference to uncertainty')
		print(slope)
		plt.subplot(3,1,1)
		plt.errorbar(bin_centers[mask], bin_means[mask], bin_std[mask],ecolor='r')
		plt.scatter(bin_centers[mask], bin_means[mask],color='b')
		plt.ylim([0,900])
		plt.xlabel('Time Without Positioning (days)',labelpad=label_pad)
		plt.annotate('a', xy = (0.8,0.75),xycoords='axes fraction',zorder=10,size=14,bbox=dict(boxstyle="round", fc="0.8"),)

		plt_toa = []
		for j,toa_number_token in enumerate(np.unique(toa_number)):
			mask = (np.array(toa_number)==toa_number_token)
			plt_toa.append(np.array(two_d_hist_diff)[mask].mean())
		slope, intercept, r_value, p_value, std_err = stats.linregress(x_toa, y)
		print('slope of toa heard to uncertainty')
		print(slope)
		plt.subplot(3,1,2)
		plt.scatter(np.unique(toa_number),plt_toa,alpha=0.2)
		plt.plot([min(toa_number),max(toa_number)],[intercept+slope*min(toa_number),intercept+slope*max(toa_number)])
		plt.ylim([0,100])
		plt.xlim([0.9,6.1])
		plt.xlabel('Sources Heard',labelpad=label_pad)
		plt.ylabel('Trajectory Difference (km)')
		plt.annotate('b', xy = (0.8,0.75),xycoords='axes fraction',zorder=10,size=14,bbox=dict(boxstyle="round", fc="0.8"),)
		
		plt.subplot(3,1,3)

		bins = np.linspace(min(percent_list)*100,max(percent_list)*100,7)
		bin_means, bin_edges, binnumber = stats.binned_statistic(np.array(percent_list)*100,
			mean_list, statistic='median', bins=bins)
		bin_std, bin_edges, binnumber = stats.binned_statistic(np.array(percent_list)*100,
			mean_list, statistic='median', bins=bins)
		bin_width = (bin_edges[1] - bin_edges[0])
		bin_centers = bin_edges[1:] - bin_width/2
		plt.errorbar(bin_centers, bin_means, bin_std,ecolor='r')
		plt.scatter(bin_centers, bin_means,color='b')
		plt.xlabel('Days Sources Heard (%)',labelpad=label_pad)
		plt.annotate('c', xy = (0.8,0.75),xycoords='axes fraction',zorder=10,size=14,bbox=dict(boxstyle="round", fc="0.8"),)



		plt.subplots_adjust(hspace=0.4)
		plt.savefig('uncertainty_increase')
		plt.close()

		from scipy.stats import norm
		plt.figure()
		plt.subplot(3,1,1)
		bins = np.linspace(-40,40,200)
		trj_n,dummy,dummy = plt.hist(trj_error_list,bins=bins,color='b',label='ARTOA',alpha=0.3)
		kalman_n,dummy,dummy = plt.hist(kalman_error_list,bins=bins,color='g',label='Smoother',alpha=0.3)
		x = np.arange(-30,30,0.01)
		trj_plot_list = np.array(trj_error_list)[(np.array(trj_error_list)<30)&(np.array(trj_error_list)>-30)]
		trj_plot_list = norm.pdf(x,trj_plot_list.mean(),trj_plot_list.std())
		trj_plot_list = trj_n.max()/trj_plot_list.max()*trj_plot_list
		plt.plot(x,trj_plot_list,'b',alpha=0.5,linewidth=3,linestyle='dashed')
		kalman_plot_list = np.array(kalman_error_list)[(np.array(kalman_error_list)<30)&(np.array(kalman_error_list)>-30)]
		kalman_plot_list = norm.pdf(x,kalman_plot_list.mean(),kalman_plot_list.std())
		kalman_plot_list = kalman_n.max()/kalman_plot_list.max()*kalman_plot_list

		plt.plot(x,kalman_plot_list,'g',alpha=0.5,linewidth=3,linestyle='dashed')
		print('mean of the trj_error_list')
		print(np.mean(trj_error_list))
		print('std of the trj_error_list')
		print(np.std(trj_error_list))

		print('mean of the kalman')
		print(np.mean(kalman_error_list))
		print('std of the kalman')
		print(np.std(kalman_error_list))


		plt.xlabel('Misfit (seconds)', fontsize=14)
		plt.xlim([-30,30])
		plt.legend(prop={'size': 10})
		plt.annotate('a', xy = (0.2,0.7),xycoords='axes fraction',zorder=10,size=12,bbox=dict(boxstyle="round", fc="0.8"),)
		plt.subplot(3,1,2)
		bins = np.linspace(0,200,40)
		plt.xlim([0,200])
		plt.hist(diff_list,bins=bins)
		plt.xlabel('Trajectory Difference (km)', fontsize=14)
		plt.annotate('b', xy = (0.2,0.7),xycoords='axes fraction',zorder=10,size=12,bbox=dict(boxstyle="round", fc="0.8"),)		
		print('the mean misfit is')
		print(np.mean(diff_list))
		print('the standard deviation of the misfit is ')
		print(np.std(diff_list))
		counts,bins = np.histogram(diff_list,bins)
		print ('misfit peaked at ')
		print(bins[counts.tolist().index(counts.max())])


		plt.subplot(3,1,3)
		bins = np.linspace(0,41,30)
		plt.hist(kalman_speed,bins=bins,color='g',label='Kalman',alpha=0.3)
		plt.hist(dimes_speed,bins=bins,color='b',label='DIMES',alpha=0.3)
		plt.annotate('c', xy = (0.2,0.7),xycoords='axes fraction',zorder=10,size=12,bbox=dict(boxstyle="round", fc="0.8"),)		
		plt.xlim([0,20])
		plt.xlabel('Speed (km $day^{-1}$)', fontsize=14)

		print('mean artoa speed')
		print(np.mean(dimes_speed))

		counts,bins = np.histogram(dimes_speed,bins)
		print ('artoa speed peaked at ')
		print(bins[counts.tolist().index(counts.max())])

		print('median artoa speed')
		print(np.median(dimes_speed))


		print('mean kalman speed')
		print(np.mean(kalman_speed))

		counts,bins = np.histogram(kalman_speed,bins)
		print ('kalman speed peaked at ')
		print(bins[counts.tolist().index(counts.max())])

		print('median kalman speed')
		print(np.median(kalman_speed))


		plt.tight_layout(pad=1.5)
		plt.savefig('all_dimes_error')
		plt.close()

class WeddellFloatPlot(WeddellFloat):

	def traj_plot(self):
		lllon = -60
		urlon = 35
		lllat = -75
		urlat = -50
		self.plot(urlat,lllat,urlon,lllon)

	def uncertainty_ellipse(self,date):
		date_idx = self.pos_date.index(date)
		P_ = self.P[date_idx]
		pos = self.pos[date_idx]
		C = P_[[0,0,2,2],[0,2,0,2]].reshape([2,2])
		w,v = np.linalg.eig(C)
		angle = np.degrees(np.arctan(v[1,np.argmax(w)]/v[0,np.argmax(w)]))
		A = 2*max(w)*np.sqrt(9.210)/abs(degree_dist*np.cos(np.deg2rad(pos.lat.decimal_degree)))
		B = 2*min(w)*np.sqrt(9.210)/degree_dist
		return A,B,angle

	def ellipse_plot(self,sources,P):
		lat,lon = list(zip(*[(_.lat.decimal_degree,_.lon.decimal_degree) for _ in self.pos]))
		lllon = min(lon)-2
		urlon = max(lon)+2
		lllat = min(lat)-2
		urlat = max(lat)+2


		for k,idx in enumerate(np.where([len(_)>2 for _ in self.toa.obs])[0][::5]):
			print(str(self.floatname)+'_toa_'+str(k))
			fig, ax = plt.subplots()
			m = self.plot(urlat,lllat,urlon,lllon)
			date = self.toa.date[idx]
			pos_idx = self.pos_date.index(date)
			A,B,angle = self.uncertainty_ellipse(date)
			m.ellipse(lon[pos_idx],lat[pos_idx],A,B,60,m,phi=angle,line=False,ax=ax)
			m.plot(lon[pos_idx],lat[pos_idx],latlon=True,marker='X',color='lime',markersize=12,zorder=10)
			m.plot(lon[0],lat[0],latlon=True,marker='^',color='m',markersize=12)
			m.plot(lon[-1],lat[-1],latlon=True,marker='s',color='m',markersize=12)
			m.plot(lon[0],lat[0],latlon=True,color='k',markersize=12)
			self.clock.set_date(date) 
			sources.set_date(date)
			# colorlist = ['m','b','g','y']
			for data in self.toa.return_data()[0]:
				# color = colorlist.pop()
				(toa,source) = data
				print(source.ID)
				detrend_toa = source.clock.detrend_offset(toa)
				dist = source.dist_from_toa(detrend_toa)
				print(source.position)
				print(dist)
				source_lat = source.position.lat.decimal_degree
				source_lon = source.position.lon.decimal_degree
				m.scatter(source_lon,source_lat,latlon=True,s=20,color='r')
				coords = m.ellipse(source_lon,source_lat,dist/(np.cos(np.deg2rad(source_lat))*degree_dist),dist/degree_dist,50,m)
				lons,lats = list(zip(*coords))
				plt.annotate(source.ID,m(source_lon+0.5,source_lat-0.3))
				m.plot(lons,lats,color='r',latlon=True)
			plt.savefig(str(self.floatname)+'_toa_'+str(k))
			plt.close()

class DIMESFloatPlot(DIMESFloat):
	def traj_plot(self):
		plt.subplot(2,1,1)
		lllon = -125
		urlon = -20
		lllat = -70
		urlat = -40
		m = self.plot(urlat,lllat,urlon,lllon)
		m.plot(self.trj_df['Lon'].tolist(),self.trj_df['Lat'].tolist(),latlon=True)
		plt.subplot(2,1,2)

		dist_error_list,toa_error_list,dist_list,soso_list,date_return_list,pos_dist_list = self.toa.calculate_error_list(self.pos,self.pos_date)
		soso_misfit_dict = {}
		for soso_token in np.unique(soso_list):
			soso_misfit_dict[soso_token]=[]

		for k,date in enumerate(self.pos_date):
			if date in date_return_list:
				idxs = np.where([x==date for x in date_return_list])[0]
				for idx in idxs:
					soso_misfit_dict[soso_list[idx]].append((k,toa_error_list[idx]))
			else:
				continue
		for soso_token in soso_misfit_dict.keys():
			x,y = list(zip(*soso_misfit_dict[soso_token]))
			plt.plot(x,y,label=soso_token)
		plt.legend()
