from PyQt5.QtWidgets import QWidget, QLineEdit, QMessageBox, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QComboBox, \
	QCheckBox, QRadioButton, QButtonGroup
from celldetective.gui.gui_utils import FigureCanvas, center_window, color_from_class
import numpy as np
import matplotlib.pyplot as plt
from superqt import QLabeledSlider,QLabeledDoubleSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from PyQt5.QtCore import Qt, QSize
import os
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from celldetective.gui import Styles
from math import ceil
from celldetective.utils import extract_cols_from_query

def step_function(t, t_shift, dt):
	return 1/(1+np.exp(-(t-t_shift)/dt))

class ClassifierWidget(QWidget, Styles):

	def __init__(self, parent_window):

		super().__init__()

		self.parent_window = parent_window
		self.screen_height = self.parent_window.parent_window.parent_window.screen_height
		self.screen_width = self.parent_window.parent_window.parent_window.screen_width
		self.currentAlpha = 1.0

		self.setWindowTitle("Custom classification")

		self.mode = self.parent_window.mode
		self.df = self.parent_window.df

		is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
		is_number_test = is_number(self.df.dtypes)
		self.cols = [col for t,col in zip(is_number_test,self.df.columns) if t]

		self.class_name = 'custom'
		self.name_le = QLineEdit(self.class_name)
		self.init_class()

		# Create the QComboBox and add some items
		center_window(self)

		
		layout = QVBoxLayout(self)
		layout.setContentsMargins(30,30,30,30)

		name_layout = QHBoxLayout()
		name_layout.addWidget(QLabel('class name: '), 33)
		name_layout.addWidget(self.name_le, 66)
		layout.addLayout(name_layout)

		fig_btn_hbox = QHBoxLayout()
		fig_btn_hbox.addWidget(QLabel(''), 95)
		self.project_times_btn = QPushButton('')
		self.project_times_btn.setStyleSheet(self.parent_window.parent_window.parent_window.button_select_all)
		self.project_times_btn.setIcon(icon(MDI6.math_integral,color="black"))
		self.project_times_btn.setToolTip("Project measurements at all times.")
		self.project_times_btn.setIconSize(QSize(20, 20))
		self.project_times = False
		self.project_times_btn.clicked.connect(self.switch_projection)
		fig_btn_hbox.addWidget(self.project_times_btn, 5)
		layout.addLayout(fig_btn_hbox)

		# Figure
		self.initalize_props_scatter()
		layout.addWidget(self.propscanvas)

		# slider
		self.frame_slider = QLabeledSlider()
		self.frame_slider.setSingleStep(1)
		self.frame_slider.setOrientation(1)
		self.frame_slider.setRange(0,int(self.df.FRAME.max()) - 1)
		self.frame_slider.setValue(0)
		self.currentFrame = 0

		slider_hbox = QHBoxLayout()
		slider_hbox.addWidget(QLabel('frame: '), 10)
		slider_hbox.addWidget(self.frame_slider, 90)
		layout.addLayout(slider_hbox)


		# transparency slider
		self.alpha_slider = QLabeledDoubleSlider()
		self.alpha_slider.setSingleStep(0.001)
		self.alpha_slider.setOrientation(1)
		self.alpha_slider.setRange(0,1)
		self.alpha_slider.setValue(1.0)
		self.alpha_slider.setDecimals(3)

		slider_alpha_hbox = QHBoxLayout()
		slider_alpha_hbox.addWidget(QLabel('transparency: '), 10)
		slider_alpha_hbox.addWidget(self.alpha_slider, 90)
		layout.addLayout(slider_alpha_hbox)



		self.features_cb = [QComboBox() for i in range(2)]
		self.log_btns = [QPushButton() for i in range(2)]

		for i in range(2):
			hbox_feat = QHBoxLayout()
			hbox_feat.addWidget(QLabel(f'feature {i}: '), 20)
			hbox_feat.addWidget(self.features_cb[i], 75)
			hbox_feat.addWidget(self.log_btns[i], 5)
			layout.addLayout(hbox_feat)

			self.features_cb[i].clear()
			self.features_cb[i].addItems(sorted(list(self.cols),key=str.lower))
			self.features_cb[i].currentTextChanged.connect(self.update_props_scatter)
			self.features_cb[i].setCurrentIndex(i)

			self.log_btns[i].setIcon(icon(MDI6.math_log,color="black"))
			self.log_btns[i].setStyleSheet(self.button_select_all)
			self.log_btns[i].clicked.connect(lambda ch, i=i: self.switch_to_log(i))

		hbox_classify = QHBoxLayout()
		hbox_classify.addWidget(QLabel('classify: '), 10)
		self.property_query_le = QLineEdit()
		self.property_query_le.setPlaceholderText('classify points using a query such as: area > 100 or eccentricity > 0.95')
		hbox_classify.addWidget(self.property_query_le, 70)
		self.submit_query_btn = QPushButton('Submit...')
		self.submit_query_btn.clicked.connect(self.apply_property_query)
		hbox_classify.addWidget(self.submit_query_btn, 20)
		layout.addLayout(hbox_classify)

		self.time_corr = QCheckBox('Time correlated')
		self.time_corr.toggled.connect(self.activate_time_corr_options)
		if "TRACK_ID" in self.df.columns:
			self.time_corr.setEnabled(True)
		else:
			self.time_corr.setEnabled(False)
		layout.addWidget(self.time_corr,alignment=Qt.AlignCenter)

		self.irreversible_event_btn = QRadioButton('irreversible event')
		self.unique_state_btn = QRadioButton('unique state')
		time_corr_btn_group = QButtonGroup()
		self.unique_state_btn.click()
		self.time_corr_options = [self.irreversible_event_btn, self.unique_state_btn]

		for btn in self.time_corr_options:
			time_corr_btn_group.addButton(btn)
			btn.setEnabled(False)

		time_corr_layout = QHBoxLayout()
		time_corr_layout.addWidget(self.unique_state_btn, 50, alignment=Qt.AlignCenter)
		time_corr_layout.addWidget(self.irreversible_event_btn, 50,alignment=Qt.AlignCenter)
		layout.addLayout(time_corr_layout)

		self.r2_slider = QLabeledDoubleSlider()
		self.r2_slider.setValue(0.75)
		self.r2_slider.setRange(0,1)
		self.r2_slider.setSingleStep(0.01)
		self.r2_slider.setOrientation(1)
		self.r2_label = QLabel('R2 tolerance:')
		self.r2_label.setToolTip('Minimum R2 between the fit sigmoid and the binary response to the filters to accept the event.')
		r2_threshold_layout = QHBoxLayout()
		r2_threshold_layout.addWidget(QLabel(''), 50)
		r2_threshold_layout.addWidget(self.r2_label, 15)
		r2_threshold_layout.addWidget(self.r2_slider, 35)
		layout.addLayout(r2_threshold_layout)	
		
		self.irreversible_event_btn.clicked.connect(self.activate_r2)
		self.unique_state_btn.clicked.connect(self.activate_r2)

		for wg in [self.r2_slider, self.r2_label]:
			wg.setEnabled(False)

		layout.addWidget(QLabel())


		self.submit_btn = QPushButton('apply')
		self.submit_btn.setStyleSheet(self.button_style_sheet)
		self.submit_btn.clicked.connect(self.submit_classification)
		layout.addWidget(self.submit_btn, 30)

		self.frame_slider.valueChanged.connect(self.set_frame)
		self.alpha_slider.valueChanged.connect(self.set_transparency)

	def activate_r2(self):
		if self.irreversible_event_btn.isChecked() and self.time_corr.isChecked():
			for wg in [self.r2_slider, self.r2_label]:
				wg.setEnabled(True)
		else:
			for wg in [self.r2_slider, self.r2_label]:
				wg.setEnabled(False)				

	def activate_time_corr_options(self):

		if self.time_corr.isChecked():
			for btn in self.time_corr_options:
				btn.setEnabled(True)
			if self.irreversible_event_btn.isChecked():
				for wg in [self.r2_slider, self.r2_label]:
					wg.setEnabled(True)
			else:
				for wg in [self.r2_slider, self.r2_label]:
					wg.setEnabled(False)				
		else:
			for btn in self.time_corr_options:
				btn.setEnabled(False)
			for wg in [self.r2_slider, self.r2_label]:
				wg.setEnabled(False)

	def init_class(self):

		self.class_name = 'custom'
		i=1
		while self.class_name in self.df.columns:
			self.class_name = f'custom_{i}'
			i+=1
		self.name_le.setText(self.class_name)
		self.df.loc[:,self.class_name] = 1

	def initalize_props_scatter(self):

		"""
		Define properties scatter.
		"""

		self.fig_props, self.ax_props = plt.subplots(figsize=(4,4),tight_layout=True)
		self.propscanvas = FigureCanvas(self.fig_props, interactive=True)
		self.fig_props.set_facecolor('none')
		self.fig_props.canvas.setStyleSheet("background-color: transparent;")
		self.scat_props = self.ax_props.scatter([],[], color="k", alpha=self.currentAlpha)
		self.propscanvas.canvas.draw_idle()
		self.propscanvas.canvas.setMinimumHeight(self.screen_height//5)

	def update_props_scatter(self, feature_changed=True):

		if not self.project_times:
			self.scat_props.set_offsets(self.df.loc[self.df['FRAME']==self.currentFrame,[self.features_cb[1].currentText(),self.features_cb[0].currentText()]].to_numpy())
			colors = [color_from_class(c) for c in self.df.loc[self.df['FRAME']==self.currentFrame,self.class_name].to_numpy()]
			self.scat_props.set_facecolor(colors)
			self.scat_props.set_alpha(self.currentAlpha)
			self.ax_props.set_xlabel(self.features_cb[1].currentText())
			self.ax_props.set_ylabel(self.features_cb[0].currentText())
		else:
			self.scat_props.set_offsets(self.df[[self.features_cb[1].currentText(),self.features_cb[0].currentText()]].to_numpy())
			colors = [color_from_class(c) for c in self.df[self.class_name].to_numpy()]
			self.scat_props.set_facecolor(colors)
			self.scat_props.set_alpha(self.currentAlpha)
			self.ax_props.set_xlabel(self.features_cb[1].currentText())
			self.ax_props.set_ylabel(self.features_cb[0].currentText())


		feat_x = self.features_cb[1].currentText()
		feat_y = self.features_cb[0].currentText()
		min_x = self.df.dropna(subset=feat_x)[feat_x].min()
		max_x = self.df.dropna(subset=feat_x)[feat_x].max()
		min_y = self.df.dropna(subset=feat_y)[feat_y].min()
		max_y = self.df.dropna(subset=feat_y)[feat_y].max()

		if min_x==min_x and max_x==max_x:
			self.ax_props.set_xlim(min_x, max_x)
		if min_y==min_y and max_y==max_y:
			self.ax_props.set_ylim(min_y, max_y)
		
		if feature_changed:
			self.propscanvas.canvas.toolbar.update()
		self.propscanvas.canvas.draw_idle()

	def apply_property_query(self):

		query = self.property_query_le.text()
		self.df[self.class_name] = 1

		cols = extract_cols_from_query(query) 
		print(cols)
		cols_in_df = np.all([c in list(self.df.columns) for c in cols], axis=0)
		print(f'Testing if columns from query are in the dataframe: {cols_in_df}...')

		if query=='':
			print('empty query')
		else:
			try:
				if cols_in_df:
					self.selection = self.df.dropna(subset=cols).query(query).index
				else:
					self.selection = self.df.query(query).index
				self.df.loc[self.selection, self.class_name] = 0
			except Exception as e:
				print(e)
				print(self.df.columns)
				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Warning)
				msgBox.setText(f"The query could not be understood. No filtering was applied. {e}")
				msgBox.setWindowTitle("Warning")
				msgBox.setStandardButtons(QMessageBox.Ok)
				returnValue = msgBox.exec()
				if returnValue == QMessageBox.Ok:
					return None

		self.update_props_scatter()

	def set_frame(self, value):
		xlim=self.ax_props.get_xlim()
		ylim=self.ax_props.get_ylim()
		self.currentFrame = value
		self.update_props_scatter(feature_changed=False)
		self.ax_props.set_xlim(xlim)
		self.ax_props.set_ylim(ylim)


	def set_transparency(self, value):
		xlim=self.ax_props.get_xlim()
		ylim=self.ax_props.get_ylim()
		self.currentAlpha = value
		#fc = self.scat_props.get_facecolors()
		#fc[:, 3] = value
		#self.scat_props.set_facecolors(fc)
		#self.propscanvas.canvas.draw_idle()
		self.update_props_scatter(feature_changed=False)
		self.ax_props.set_xlim(xlim)
		self.ax_props.set_ylim(ylim)

	def switch_projection(self):
		if self.project_times:
			self.project_times = False
			self.project_times_btn.setIcon(icon(MDI6.math_integral,color="black"))
			self.project_times_btn.setIconSize(QSize(20, 20))
			self.frame_slider.setEnabled(True)
		else:
			self.project_times = True
			self.project_times_btn.setIcon(icon(MDI6.math_integral_box,color="black"))
			self.project_times_btn.setIconSize(QSize(20, 20))
			self.frame_slider.setEnabled(False)
		self.update_props_scatter()

	def submit_classification(self):
		
		print('submit')
		self.apply_property_query()

		if self.time_corr.isChecked():
			self.class_name_user = 'class_'+self.name_le.text()
			print(f'User defined class name: {self.class_name_user}.')
			if self.class_name_user in self.df.columns:

				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Information)
				msgBox.setText(f"The class column {self.class_name_user} already exists in the table.\nProceeding will reclassify. Do you want to continue?")
				msgBox.setWindowTitle("Warning")
				msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
				returnValue = msgBox.exec()
				if returnValue == QMessageBox.Yes:
					pass
				else:
					return None

			name_map = {self.class_name: self.class_name_user}
			self.df = self.df.drop(list(set(name_map.values()) & set(self.df.columns)), axis=1).rename(columns=name_map)
			self.df.reset_index(inplace=True, drop=True)

			#self.df.reset_index(inplace=True)
			if 'TRACK_ID' in self.df.columns:
				print('Tracks detected... save a status column...')
				stat_col = self.class_name_user.replace('class','status')
				self.df.loc[:,stat_col] = 1 - self.df[self.class_name_user].values
				for tid,track in self.df.groupby(['position','TRACK_ID']):
					indices = track[self.class_name_user].index
					status_values = track[stat_col].to_numpy()
					if self.irreversible_event_btn.isChecked():
						if np.all([s==0 for s in status_values]):
							self.df.loc[indices, self.class_name_user] = 1
						elif np.all([s==1 for s in status_values]):
							self.df.loc[indices, self.class_name_user] = 2
							self.df.loc[indices, self.class_name_user.replace('class','status')] = 2
						else:
							self.df.loc[indices, self.class_name_user] = 2
					elif self.unique_state_btn.isChecked():
						frames = track['FRAME'].to_numpy()
						t_first = track['t_firstdetection'].to_numpy()[0]
						median_status = np.nanmedian(status_values[frames>=t_first])
						if median_status==median_status:
							c = ceil(median_status)
							if c==0:
								self.df.loc[indices, self.class_name_user] = 1
								self.df.loc[indices, self.class_name_user.replace('class','t')] = -1
							elif c==1:
								self.df.loc[indices, self.class_name_user] = 2
								self.df.loc[indices, self.class_name_user.replace('class','t')] = -1
				if self.irreversible_event_btn.isChecked():
					self.df.loc[self.df[self.class_name_user]!=2, self.class_name_user.replace('class', 't')] = -1
					self.estimate_time()
		else:
			self.group_name_user = 'group_' + self.name_le.text()
			print(f'User defined characteristic group name: {self.group_name_user}.')
			if self.group_name_user in self.df.columns:

				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Information)
				msgBox.setText(
					f"The group column {self.group_name_user} already exists in the table.\nProceeding will reclassify. Do you want to continue?")
				msgBox.setWindowTitle("Warning")
				msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
				returnValue = msgBox.exec()
				if returnValue == QMessageBox.Yes:
					pass
				else:
					return None

			name_map = {self.class_name: self.group_name_user}
			self.df = self.df.drop(list(set(name_map.values()) & set(self.df.columns)), axis=1).rename(columns=name_map)
			print(self.df.columns)
			self.df[self.group_name_user] = self.df[self.group_name_user].replace({0: 1, 1: 0})
			self.df.reset_index(inplace=True, drop=True)


		for pos,pos_group in self.df.groupby('position'):
			pos_group.to_csv(pos+os.sep.join(['output', 'tables', f'trajectories_{self.mode}.csv']), index=False)

		# reset
		#self.init_class()
		#self.update_props_scatter()
		self.parent_window.parent_window.update_position_options()
		self.close()

	def switch_to_log(self, i):

		"""
		Switch threshold histogram to log scale. Auto adjust.
		"""

		if i==1:
			try:
				if self.ax_props.get_xscale()=='linear':
					self.ax_props.set_xscale('log')
					self.log_btns[i].setIcon(icon(MDI6.math_log,color="#1565c0"))
				else:
					self.ax_props.set_xscale('linear')
					self.log_btns[i].setIcon(icon(MDI6.math_log,color="black"))
			except Exception as e:
				print(e)
		elif i==0:
			try:
				if self.ax_props.get_yscale()=='linear':
					self.ax_props.set_yscale('log')
					self.log_btns[i].setIcon(icon(MDI6.math_log,color="#1565c0"))
				else:
					self.ax_props.set_yscale('linear')
					self.log_btns[i].setIcon(icon(MDI6.math_log,color="black"))
			except Exception as e:
				print(e)

		self.ax_props.autoscale()
		self.propscanvas.canvas.draw_idle()

	def estimate_time(self):

		self.df = self.df.sort_values(by=['position','TRACK_ID'],ignore_index=True)
		for tid,group in self.df.loc[self.df[self.class_name_user]==2].groupby(['position','TRACK_ID']):
			indices = group.index
			status_col = self.class_name_user.replace('class','status')
			status_signal = group[status_col].values
			timeline = group['FRAME'].values
			
			try:
				popt, pcov = curve_fit(step_function, timeline.astype(int), status_signal, p0=[self.df['FRAME'].max()//2, 0.8],maxfev=30000)
				values = [step_function(t, *popt) for t in timeline]
				r2 = r2_score(status_signal,values)
			except Exception as e:
				print(e)
				self.df.loc[indices, self.class_name_user] = 2.0
				self.df.loc[indices, self.class_name_user.replace('class','t')] = -1
				continue

			if r2 > float(self.r2_slider.value()):
				t0 = popt[0]
				self.df.loc[indices, self.class_name_user.replace('class','t')] = t0
				self.df.loc[indices, self.class_name_user] = 0.0
			else:
				self.df.loc[indices, self.class_name_user.replace('class','t')] = -1
				self.df.loc[indices, self.class_name_user] = 2.0

		print('Done.')









