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
