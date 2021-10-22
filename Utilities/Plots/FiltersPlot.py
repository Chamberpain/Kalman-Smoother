class SmootherPlot(Smoother):
	def error_calc(self,pos_list,error_label):
		innovation_list = []
		innovation_label_list = []
		self.set_date(self.float.gps.date[0])
		while self.date<=self.float.gps.date[-1]:
			assert self.date == self.float.clock.date
			gps,toa,depth,stream,interp = self.float.return_data()
			pos = self.float.return_pos()
			Z,label = self.Z_constructor(gps,toa,depth,stream,interp,error_label)
			h = self.h_constructor(gps,toa,depth,stream,interp,pos)
			Y = np.array(Z)-np.array(h) #innovation
			Y = Y.reshape(len(Z),1)
			for dummy in zip(label,Y.tolist()):
				dummy[0].append(dummy[1])
			self.increment_date()
	def X_m_minus_X_p_diagnostic(self):
		date_list,innovation = zip(*self.variable_check['x_m-x_p'])
		date_diff_list = self.obs_date_diff_list(date_list)
		innovation = np.array([np.sqrt(_[0]**2+_[2]**2) for _ in innovation]).flatten()

		plt.scatter(range(len(innovation)),innovation,s=0.3,c=np.array(date_diff_list),cmap=plt.cm.get_cmap("winter"))
		plt.colorbar(label='days since position')
		plt.xlabel('time step')
		plt.ylabel('innovation (km)')
		plt.savefig(str(self.float.floatname)+'-innovation-date-diagnostic')
		plt.close()

	def diagnostic_plot(self,innovation_list,label_list,label):
		flat_label_list = [item for sublist in label_list for item in sublist]
		flat_innovation_list = [item for sublist in innovation_list for item in sublist]
		for variable_label in np.unique(flat_label_list):
			dummy_list = []
			for _ in zip(label_list,innovation_list):
				try:
					idx = _[0].index(variable_label)
					flat_value_list = [item for sublist in _[1] for item in sublist]
					dummy_list.append(flat_value_list[idx])
				except ValueError:
					dummy_list.append(np.nan)
			plt.scatter(range(len(dummy_list)),dummy_list)
			plt.title(str(self.float.floatname)+' '+label+' '+variable_label)
			plt.savefig(str(self.float.floatname)+'_'+label+'_'+variable_label)
			plt.close()
	def innovation_diagnostic(self):
		self.diagnostic_plot(self.variable_check['innovation'],self.variable_check['innovation label'],'innovation')
	