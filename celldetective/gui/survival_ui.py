from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QScrollArea, QButtonGroup, QComboBox, QFrame, QCheckBox, QFileDialog, QGridLayout, QTextEdit, QLineEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton, QRadioButton
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QDoubleValidator
from celldetective.gui.gui_utils import center_window, FeatureChoice, ListWidget, QHSeperationLine, FigureCanvas, GeometryChoice, OperationChoice
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider,QLabeledSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from celldetective.utils import extract_experiment_channels, get_software_location, _extract_labels_from_config
from celldetective.io import interpret_tracking_configuration, load_frames, auto_load_number_of_frames, load_experiment_tables
from celldetective.measure import compute_haralick_features, contour_of_instance_segmentation
import numpy as np
from tifffile import imread
import json
from shutil import copyfile
import os
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
from natsort import natsorted
from tifffile import imread
from pathlib import Path, PurePath
import gc
import pandas as pd
from tqdm import tqdm
from lifelines import KaplanMeierFitter
import matplotlib.cm as mcm
import math
from celldetective.events import switch_to_events_v2

class ConfigSurvival(QWidget):
	
	"""
	UI to set survival instructions.

	"""

	def __init__(self, parent=None):
		
		super().__init__()
		self.parent = parent
		self.setWindowTitle("Configure survival")
		self.setWindowIcon(QIcon(os.sep.join(['celldetective','icons','mexican-hat.png'])))

		self.exp_dir = self.parent.exp_dir
		self.soft_path = get_software_location()		
		self.exp_config = self.exp_dir +"config.ini"
		self.wells = np.array(self.parent.parent.wells,dtype=str)
		self.well_labels = _extract_labels_from_config(self.exp_config,len(self.wells))
		self.FrameToMin = self.parent.parent.FrameToMin
		self.float_validator = QDoubleValidator()

		print('Parent wells: ', self.wells)


		self.well_option = self.parent.parent.well_list.currentIndex()
		self.position_option = self.parent.parent.position_list.currentIndex()
		self.interpret_pos_location()
		#self.config_path = self.exp_dir + self.config_name

		self.screen_height = self.parent.parent.parent.screen_height
		center_window(self)

		self.setMinimumWidth(350)
		#self.setMinimumHeight(int(0.8*self.screen_height))
		#self.setMaximumHeight(int(0.8*self.screen_height))
		self.populate_widget()
		#self.load_previous_measurement_instructions()

	def interpret_pos_location(self):
		
		"""
		Read the well/position selection from the control panel to decide which data to load
		Set position_indices to None if all positions must be taken

		"""
		
		if self.well_option==len(self.wells):
			self.well_indices = np.arange(len(self.wells))
		else:
			self.well_indices = np.array([self.well_option],dtype=int)

		if self.position_option==0:
			self.position_indices = None
		else:
			self.position_indices = np.array([self.position_option],dtype=int)


	def populate_widget(self):

		"""
		Create the multibox design.

		"""
		
		# Create button widget and layout
		main_layout = QVBoxLayout()
		self.setLayout(main_layout)
		main_layout.setContentsMargins(30,30,30,30)

		panel_title = QLabel('Options')
		panel_title.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		main_layout.addWidget(panel_title, alignment=Qt.AlignCenter)


		labels = [QLabel('population: '), QLabel('time of\ninterest: '), QLabel('time of\nreference: '), QLabel('class: '), QLabel('cmap: ')]
		self.cb_options = [['targets','effectors'],['t0','first detection'], ['0','first detection', 't0'], ['class'], list(plt.colormaps())]
		self.cbs = [QComboBox() for i in range(len(labels))]
		self.cbs[0].currentIndexChanged.connect(self.set_classes_and_times)

		choice_layout = QVBoxLayout()
		choice_layout.setContentsMargins(20,20,20,20)
		for i in range(len(labels)):
			hbox = QHBoxLayout()
			hbox.addWidget(labels[i], 33)
			hbox.addWidget(self.cbs[i],66)
			self.cbs[i].addItems(self.cb_options[i])
			choice_layout.addLayout(hbox)
		main_layout.addLayout(choice_layout)

		self.cbs[0].setCurrentIndex(1)
		self.cbs[0].setCurrentIndex(0)

		time_calib_layout = QHBoxLayout()
		time_calib_layout.setContentsMargins(20,20,20,20)
		time_calib_layout.addWidget(QLabel('time calibration\n(frame to min)'), 33)
		self.time_calibration_le = QLineEdit(str(self.FrameToMin).replace('.',','))
		self.time_calibration_le.setValidator(self.float_validator)
		time_calib_layout.addWidget(self.time_calibration_le, 66)
		#time_calib_layout.addWidget(QLabel(' min'))
		main_layout.addLayout(time_calib_layout)

		self.submit_btn = QPushButton('Submit')
		self.submit_btn.setStyleSheet(self.parent.parent.parent.button_style_sheet)
		self.submit_btn.clicked.connect(self.process_survival)
		main_layout.addWidget(self.submit_btn)

		#self.populate_left_panel()
		#grid.addLayout(self.left_side, 0, 0, 1, 1)

		# self.setCentralWidget(self.scroll_area)
		# self.show()

	def set_classes_and_times(self):

		# Look for all classes and times
		tables = glob(self.exp_dir+os.sep.join(['W*','*','output','tables',f'trajectories_*']))
		self.all_columns = []
		for tab in tables:
			cols = pd.read_csv(tab, nrows=1).columns.tolist()
			self.all_columns.extend(cols)
		self.all_columns = np.unique(self.all_columns)
		class_idx = np.array([s.startswith('class_') for s in self.all_columns])
		time_idx = np.array([s.startswith('t_') for s in self.all_columns])
		class_columns = list(self.all_columns[class_idx])
		for c in ['class_id', 'class_color']:
			if c in class_columns:
				class_columns.remove(c)

		time_columns = list(self.all_columns[time_idx])
		
		self.cbs[1].clear()
		self.cbs[1].addItems(np.unique(self.cb_options[1]+time_columns))

		self.cbs[2].clear()
		self.cbs[2].addItems(np.unique(self.cb_options[2]+time_columns))

		self.cbs[3].clear()
		self.cbs[3].addItems(np.unique(self.cb_options[3]+class_columns))
		

	def process_survival(self):

		print('you clicked!!')
		self.FrameToMin = float(self.time_calibration_le.text().replace(',','.'))
		print(self.FrameToMin, 'set')

		# read instructions from combobox options
		self.load_available_tables_local()
		if self.df is not None:
			self.compute_survival_functions()
			# prepare survival

			# plot survival
			self.survivalWidget = QWidget()
			self.survivalWidget.setMinimumHeight(int(0.8*self.screen_height))
			self.survivalWidget.setWindowTitle('survival')
			self.plotvbox = QVBoxLayout(self.survivalWidget)
			self.plotvbox.setContentsMargins(30,30,30,30)
			self.survival_title = QLabel('Survival function')
			self.survival_title.setStyleSheet("""
				font-weight: bold;
				padding: 0px;
				""")
			self.plotvbox.addWidget(self.survival_title, alignment=Qt.AlignCenter)

			plot_buttons_hbox = QHBoxLayout()
			plot_buttons_hbox.addWidget(QLabel(''),90, alignment=Qt.AlignLeft)

			self.legend_btn = QPushButton('')
			self.legend_btn.setIcon(icon(MDI6.text_box,color="black"))
			self.legend_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
			self.legend_btn.setToolTip('Show or hide the legend')
			self.legend_visible = True
			self.legend_btn.clicked.connect(self.show_hide_legend)
			plot_buttons_hbox.addWidget(self.legend_btn, 5,alignment=Qt.AlignRight)


			self.log_btn = QPushButton('')
			self.log_btn.setIcon(icon(MDI6.math_log,color="black"))
			self.log_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
			self.log_btn.clicked.connect(self.switch_to_log)
			self.log_btn.setToolTip('Enable or disable log scale')
			plot_buttons_hbox.addWidget(self.log_btn, 5, alignment=Qt.AlignRight)

			self.fig, self.ax = plt.subplots(1,1,figsize=(4,3))
			self.survival_window = FigureCanvas(self.fig, title="Survival")
			self.survival_window.setContentsMargins(0,0,0,0)
			if self.df is not None:
				self.initialize_axis()
			plt.tight_layout()
			self.fig.set_facecolor('none')  # or 'None'
			self.fig.canvas.setStyleSheet("background-color: transparent;")
			self.survival_window.canvas.draw()

			#self.survival_window.layout.addWidget(QLabel('WHAAAAATTT???'))

			self.plot_options = [QRadioButton() for i in range(3)]
			self.radio_labels = ['well', 'pos', 'both']
			radio_hbox = QHBoxLayout()
			radio_hbox.setContentsMargins(30,30,30,30)
			self.plot_btn_group = QButtonGroup()
			for i in range(3):
				self.plot_options[i].setText(self.radio_labels[i])
				#self.plot_options[i].toggled.connect(self.plot_survivals)
				self.plot_btn_group.addButton(self.plot_options[i])
				radio_hbox.addWidget(self.plot_options[i], 33, alignment=Qt.AlignCenter)
			self.plot_btn_group.buttonClicked[int].connect(self.plot_survivals)

			if self.position_indices is not None:
				if len(self.well_indices)>1 and len(self.position_indices)==1:
					self.plot_btn_group.buttons()[0].click()
					for i in [1,2]:
						self.plot_options[i].setEnabled(False)
				elif len(self.well_indices)>1:
					self.plot_btn_group.buttons()[0].click()
				elif len(self.well_indices)==1 and len(self.position_indices)==1:
					self.plot_btn_group.buttons()[1].click()
					for i in [0,2]:
						self.plot_options[i].setEnabled(False)
			else:
				if len(self.well_indices)>1:
					self.plot_btn_group.buttons()[0].click()
				elif len(self.well_indices)==1:
					self.plot_btn_group.buttons()[2].click()


			# elif len(self.well_indices)>1:		
			# 	self.plot_btn_group.buttons()[0].click()
			# else:
			# 	self.plot_btn_group.buttons()[1].click()

			# if self.position_indices is not None:
			# 	for i in [0,2]:
			# 		self.plot_options[i].setEnabled(False)


			#self.plot_options[0].setChecked(True)
			self.plotvbox.addLayout(radio_hbox)

			self.plotvbox.addLayout(plot_buttons_hbox)
			self.plotvbox.addWidget(self.survival_window)

			self.select_pos_label = QLabel('Select positions')
			self.select_pos_label.setStyleSheet("""
				font-weight: bold;
				padding: 0px;
				""")
			self.plotvbox.addWidget(self.select_pos_label, alignment=Qt.AlignCenter)

			self.select_option = [QRadioButton() for i in range(2)]
			self.select_label = ['name', 'spatial']
			select_hbox = QHBoxLayout()
			select_hbox.setContentsMargins(30,30,30,30)
			self.select_btn_group = QButtonGroup()
			for i in range(2):
				self.select_option[i].setText(self.select_label[i])
				#self.select_option[i].toggled.connect(self.switch_selection_mode)
				self.select_btn_group.addButton(self.select_option[i])
				select_hbox.addWidget(self.select_option[i],33, alignment=Qt.AlignCenter)
			self.select_btn_group.buttonClicked[int].connect(self.switch_selection_mode)
			self.plotvbox.addLayout(select_hbox)

			self.look_for_metadata()
			if self.metadata_found:
				self.fig_scatter, self.ax_scatter = plt.subplots(1,1,figsize=(4,3))
				self.position_scatter = FigureCanvas(self.fig_scatter)
				self.load_coordinates()
				self.plot_spatial_location()
				#self.plot_positions()
				self.ax_scatter.spines['top'].set_visible(False)
				self.ax_scatter.spines['right'].set_visible(False)
				self.ax_scatter.set_aspect('equal')
				self.ax_scatter.set_xticks([])
				self.ax_scatter.set_yticks([])
				plt.tight_layout()
				self.fig_scatter.set_facecolor('none')  # or 'None'
				self.fig_scatter.canvas.setStyleSheet("background-color: transparent;")
				self.plotvbox.addWidget(self.position_scatter)

			self.generate_pos_selection_widget()

			# if self.df is not None and len(self.ks_estimators_per_position)>0:
			# 	self.plot_survivals()
			self.select_btn_group.buttons()[0].click()
			self.survivalWidget.show()

	def generate_pos_selection_widget(self):

		self.well_names = self.df['well_name'].unique()
		self.pos_names = self.df_pos_info['pos_name'].unique() #pd.DataFrame(self.ks_estimators_per_position)['position_name'].unique()
		print(f'POSITION NAMES: ',self.pos_names)
		self.usable_well_labels = []
		for name in self.well_names:
			for lbl in self.well_labels:
				if name+':' in lbl:
					self.usable_well_labels.append(lbl)

		self.line_choice_widget = QWidget()
		self.line_check_vbox = QVBoxLayout()
		self.line_choice_widget.setLayout(self.line_check_vbox)
		if len(self.well_indices)>1:
			self.well_display_options = [QCheckBox(self.usable_well_labels[i]) for i in range(len(self.usable_well_labels))]
			for i in range(len(self.well_names)):
				self.line_check_vbox.addWidget(self.well_display_options[i], alignment=Qt.AlignLeft)
				self.well_display_options[i].setChecked(True)
				self.well_display_options[i].toggled.connect(self.select_survival_lines)
		else:
			self.pos_display_options = [QCheckBox(self.pos_names[i]) for i in range(len(self.pos_names))]
			for i in range(len(self.pos_names)):
				self.line_check_vbox.addWidget(self.pos_display_options[i], alignment=Qt.AlignLeft)
				self.pos_display_options[i].setChecked(True)
				self.pos_display_options[i].toggled.connect(self.select_survival_lines)

		self.plotvbox.addWidget(self.line_choice_widget, alignment=Qt.AlignCenter)


	def load_available_tables_local(self):

		"""
		Load the tables of the selected wells/positions from the control Panel for the population of interest

		"""

		self.well_option = self.parent.parent.well_list.currentIndex()
		if self.well_option==len(self.wells):
			wo = '*'
		else:
			wo = self.well_option
		self.position_option = self.parent.parent.position_list.currentIndex()
		if self.position_option==0:
			po = '*'
		else:
			po = self.position_option - 1

		self.df, self.df_pos_info = load_experiment_tables(self.exp_dir, well_option=wo, position_option=po, population=self.cbs[0].currentText(), return_pos_info=True)
		if self.df is None:
			print('no table could be found...')
		else:
			self.df_well_info = self.df_pos_info.loc[:,['well_path', 'well_index', 'well_name', 'well_number', 'well_alias']].drop_duplicates()

	def compute_survival_functions(self):

		# Per position survival
		left_censored = False
		for block,movie_group in self.df.groupby(['well','position']):
			classes = movie_group.groupby('TRACK_ID')[self.cbs[3].currentText()].min().values
			times = movie_group.groupby('TRACK_ID')[self.cbs[1].currentText()].min().values
			max_times = movie_group.groupby('TRACK_ID')['FRAME'].max().values
			first_detections = None
			
			if self.cbs[2].currentText()=='first detection':
				left_censored = True

				first_detections = []
				for tid,track_group in movie_group.groupby('TRACK_ID'):
					if 'area' in self.df.columns:
						area = track_group['area'].values
						timeline = track_group['FRAME'].values
						if np.any(area==area):
							first_det = timeline[area==area][0]
							first_detections.append(first_det)
						else:
							# think about assymmetry with class and times
							continue
					else:
						continue

			elif self.cbs[2].currentText().startswith('t'):
				left_censored = True
				first_detections = movie_group.groupby('TRACK_ID')[self.cbs[2].currentText()].max().values
				print(first_detections)


			if self.cbs[2].currentText()=='first detection' or self.cbs[2].currentText().startswith('t'):
				left_censored = True
			else:
				left_censored = False
			events, survival_times = switch_to_events_v2(classes, times, max_times, first_detections, left_censored=left_censored, FrameToMin=self.FrameToMin)
			ks = KaplanMeierFitter()
			if len(events)>0:
				ks.fit(survival_times, event_observed=events)
				self.df_pos_info.loc[self.df_pos_info['pos_path']==block[1],'survival_fit'] = ks

		# Per well survival
		left_censored = False
		for well,well_group in self.df.groupby('well'):

			well_classes = []
			well_times = []
			well_max_times = []
			well_first_detections = []

			for block,movie_group in well_group.groupby('position'):
				classes = movie_group.groupby('TRACK_ID')[self.cbs[3].currentText()].min().values
				times = movie_group.groupby('TRACK_ID')[self.cbs[1].currentText()].min().values
				max_times = movie_group.groupby('TRACK_ID')['FRAME'].max().values
				first_detections = None
				if self.cbs[2].currentText()=='first detection':
					
					left_censored = True
					first_detections = []
					for tid,track_group in movie_group.groupby('TRACK_ID'):
						if 'area' in self.df.columns:
							area = track_group['area'].values
							timeline = track_group['FRAME'].values
							if np.any(area==area):
								first_det = timeline[area==area][0]
								first_detections.append(first_det)
							else:
								# think about assymmetry with class and times
								continue

				elif self.cbs[2].currentText().startswith('t'):
					left_censored = True
					first_detections = movie_group.groupby('TRACK_ID')[self.cbs[2].currentText()].max().values

				else:
					continue

				well_classes.extend(classes)
				well_times.extend(times)
				well_max_times.extend(max_times)
				if first_detections is not None:
					well_first_detections.extend(first_detections)  
			
			if len(well_first_detections)==0:
				well_first_detections = None
		  
			events, survival_times = switch_to_events_v2(well_classes, well_times, well_max_times, well_first_detections,left_censored=left_censored, FrameToMin=self.FrameToMin)
			ks = KaplanMeierFitter()
			if len(survival_times)>0:
				ks.fit(survival_times, event_observed=events)
				print(ks.survival_function_)
			else:
				ks = None
			self.df_well_info.loc[self.df_well_info['well_path']==well,'survival_fit'] = ks

		self.df_pos_info.loc[:,'select'] = True
		self.df_well_info.loc[:,'select'] = True

	def initialize_axis(self):

		self.ax.clear()
		self.ax.plot([],[])
		self.ax.spines['top'].set_visible(False)
		self.ax.spines['right'].set_visible(False)
		#self.ax.set_ylim(0.001,1.05)
		self.ax.set_xlim(0,self.df['FRAME'].max()*self.FrameToMin)
		self.ax.set_xlabel('time [min]')
		self.ax.set_ylabel('survival')

	def plot_survivals(self, id):

		for i in range(3):
			if self.plot_options[i].isChecked():
				self.plot_mode = self.radio_labels[i]

		cmap_lbl = self.cbs[4].currentText()
		self.cmap = getattr(mcm, cmap_lbl)

		colors = np.array([self.cmap(i / len(self.df_pos_info)) for i in range(len(self.df_pos_info))])
		well_color = [self.cmap(i / len(self.df_well_info)) for i in range(len(self.df_well_info))]

		if self.plot_mode=='pos':
			self.initialize_axis()
			lines = self.df_pos_info.loc[self.df_pos_info['select'],'survival_fit'].values
			pos_labels = self.df_pos_info.loc[self.df_pos_info['select'],'pos_name'].values
			pos_indices = self.df_pos_info.loc[self.df_pos_info['select'],'pos_index'].values
			well_index = self.df_pos_info.loc[self.df_pos_info['select'],'well_index'].values
			for i in range(len(lines)):
				if (len(self.well_indices)<=1) and lines[i]==lines[i]:
					try:
						lines[i].plot_survival_function(ax=self.ax, legend=None, color=colors[pos_indices[i]],label=pos_labels[i], xlabel='timeline [min]')
					except Exception as e:
						print(f'error {e}')
						pass
				elif lines[i]==lines[i]:
					try:
						lines[i].plot_survival_function(ax=self.ax, legend=None, color=well_color[well_index[i]],label=pos_labels[i], xlabel='timeline [min]')
					except Exception as e:
						print(f'error {e}')
						pass
				else:
					pass

		elif self.plot_mode=='well':
			self.initialize_axis()
			lines = self.df_well_info.loc[self.df_well_info['select'],'survival_fit'].values	
			well_index = self.df_well_info.loc[self.df_well_info['select'],'well_index'].values
			well_labels = self.df_well_info.loc[self.df_well_info['select'],'well_name'].values
			for i in range(len(lines)):		
				if len(self.well_indices)<=1 and lines[i]==lines[i]:

					try:
						lines[i].plot_survival_function(ax=self.ax, label=well_labels[i], color="k", xlabel='timeline [min]')
					except Exception as e:
						print(f'error {e}')
						pass
				elif lines[i]==lines[i]:
					try:
						lines[i].plot_survival_function(ax=self.ax, label=well_labels[i], color=well_color[well_index[i]], xlabel='timeline [min]')
					except Exception as e:
						print(f'error {e}')
						pass
				else:
					pass

		elif self.plot_mode=='both':
			self.initialize_axis()
			lines_pos = self.df_pos_info.loc[self.df_pos_info['select'],'survival_fit'].values
			lines_well = self.df_well_info.loc[self.df_well_info['select'],'survival_fit'].values	

			pos_indices = self.df_pos_info.loc[self.df_pos_info['select'],'pos_index'].values
			well_index_pos = self.df_pos_info.loc[self.df_pos_info['select'],'well_index'].values
			well_index = self.df_well_info.loc[self.df_well_info['select'],'well_index'].values
			well_labels = self.df_well_info.loc[self.df_well_info['select'],'well_name'].values
			pos_labels = self.df_pos_info.loc[self.df_pos_info['select'],'pos_name'].values

			for i in range(len(lines_pos)):
				if len(self.well_indices)<=1 and lines_pos[i]==lines_pos[i]:
					
					try:
						lines_pos[i].plot_survival_function(ax=self.ax, label=pos_labels[i], alpha=0.25, color=colors[pos_indices[i]], xlabel='timeline [min]')
					except Exception as e:
						print(f'error {e}')
						pass
				elif lines_pos[i]==lines_pos[i]:
					try:
						lines_pos[i].plot_survival_function(ci_show=False, ax=self.ax, legend=None, alpha=0.25, color=well_color[well_index_pos[i]], xlabel='timeline [min]')
					except Exception as e:
						print(f'error {e}')
						pass
				else:
					pass

			for i in range(len(lines_well)):		
				if len(self.well_indices)<=1 and lines_well[i]==lines_well[i]:
					try:
						lines_well[i].plot_survival_function(ax=self.ax, label='pool', color="k")
					except Exception as e:
						print(f'error {e}')
						pass
				elif lines_well[i]==lines_well[i]:
					try:
						lines_well[i].plot_survival_function(ax=self.ax, label=well_labels[i], color=well_color[well_index[i]])
					except Exception as e:
						print(f'error {e}')
						pass
				else:
					pass

		
		self.survival_window.canvas.draw()

	def switch_to_log(self):

		"""
		Switch threshold histogram to log scale. Auto adjust.
		"""

		if self.ax.get_yscale()=='linear':
			self.ax.set_yscale('log')
			#self.ax.set_ylim(0.01,1.05)
		else:
			self.ax.set_yscale('linear')
			#self.ax.set_ylim(0.01,1.05)

		#self.ax.autoscale()
		self.survival_window.canvas.draw_idle()

	def show_hide_legend(self):
		if self.legend_visible:
			self.ax.legend().set_visible(False)
			self.legend_visible = False
			self.legend_btn.setIcon(icon(MDI6.text_box_outline,color="black"))
		else:
			self.ax.legend().set_visible(True)
			self.legend_visible = True
			self.legend_btn.setIcon(icon(MDI6.text_box,color="black"))

		self.survival_window.canvas.draw_idle()		

	def look_for_metadata(self):

		self.metadata_found = False
		self.metafiles = glob(self.exp_dir+os.sep.join([f'W*','*','movie','*metadata.txt'])) \
					+ glob(self.exp_dir+os.sep.join([f'W*','*','*metadata.txt'])) \
					+ glob(self.exp_dir+os.sep.join([f'W*','*metadata.txt'])) \
					+ glob(self.exp_dir+'*metadata.txt')
		print(f'Found {len(self.metafiles)} metadata files...')
		if len(self.metafiles)>0:
			self.metadata_found = True

	def switch_selection_mode(self, id):
		print(f'button {id} was clicked')
		for i in range(2):
			if self.select_option[i].isChecked():
				self.selection_mode = self.select_label[i]
		if self.selection_mode=='name':
			if len(self.metafiles)>0:
				self.position_scatter.hide()
			self.line_choice_widget.show()
		else:
			if len(self.metafiles)>0:
				self.position_scatter.show()
			self.line_choice_widget.hide()


	def load_coordinates(self):

		"""
		Read metadata and try to extract position coordinates
		"""

		self.no_meta = False
		try:
			with open(self.metafiles[0], 'r') as f:
				data = json.load(f)
				positions = data['Summary']['InitialPositionList']
		except Exception as e:
			print(f'Trouble loading metadata: error {e}...')
			return None

		for k in range(len(positions)):
			pos_label = positions[k]['Label']
			try:
				coords = positions[k]['DeviceCoordinatesUm']['XYStage']
			except:
				try:
					coords = positions[k]['DeviceCoordinatesUm']['PIXYStage']
				except:
					self.no_meta = True

			if not self.no_meta:
				self.df_pos_info = self.df_pos_info.dropna(subset=['stack_path'])
				files = self.df_pos_info['stack_path'].values
				print(files)
				pos_loc = [pos_label in f for f in files]
				self.df_pos_info.loc[pos_loc, 'x'] = coords[0]
				self.df_pos_info.loc[pos_loc, 'y'] = coords[1]
				self.df_pos_info.loc[pos_loc, 'metadata_tag'] = pos_label


	def update_annot(self, ind):
		
		pos = self.sc.get_offsets()[ind["ind"][0]]
		self.annot.xy = pos
		text = self.scat_labels[ind["ind"][0]]
		self.annot.set_text(text)
		self.annot.get_bbox_patch().set_facecolor('k')
		self.annot.get_bbox_patch().set_alpha(0.4)

	def hover(self, event):
		vis = self.annot.get_visible()
		if event.inaxes == self.ax_scatter:
			cont, ind = self.sc.contains(event)
			if cont:
				self.update_annot(ind)
				self.annot.set_visible(True)
				self.fig_scatter.canvas.draw_idle()
			else:
				if vis:
					self.annot.set_visible(False)
					self.fig_scatter.canvas.draw_idle()

	def unselect_position(self, event):
		
		ind = event.ind # index of selected position
		well_idx = self.df_pos_info.iloc[ind]['well_index'].values[0]
		selectedPos = self.df_pos_info.iloc[ind]['pos_path'].values[0]
		currentSelState = self.df_pos_info.iloc[ind]['select'].values[0]
		if self.plot_options[0].isChecked() or self.plot_options[2].isChecked():
			self.df_pos_info.loc[self.df_pos_info['well_index']==well_idx,'select'] = not currentSelState
			self.df_well_info.loc[self.df_well_info['well_index']==well_idx, 'select'] = not currentSelState
			if len(self.well_indices)>1:
				self.well_display_options[well_idx].setChecked(not currentSelState)
			else:
				for p in self.pos_display_options:
					p.setChecked(not currentSelState)
		else:
			self.df_pos_info.loc[self.df_pos_info['pos_path']==selectedPos,'select'] = not currentSelState
			if len(self.well_indices)<=1:
				self.pos_display_options[ind[0]].setChecked(not currentSelState)

		self.sc.set_color(self.select_color(self.df_pos_info["select"].values))
		self.position_scatter.canvas.draw_idle()
		self.plot_survivals(0)

	def select_survival_lines(self):
		
		if len(self.well_indices)>1:
			for i in range(len(self.well_display_options)):
				self.df_well_info.loc[self.df_well_info['well_index']==i,'select'] = self.well_display_options[i].isChecked()
				self.df_pos_info.loc[self.df_pos_info['well_index']==i,'select'] = self.well_display_options[i].isChecked()
		else:
			for i in range(len(self.pos_display_options)):
				self.df_pos_info.loc[self.df_pos_info['pos_index']==i,'select'] = self.pos_display_options[i].isChecked()

		if len(self.metafiles)>0:
			try:
				self.sc.set_color(self.select_color(self.df_pos_info["select"].values))
				self.position_scatter.canvas.draw_idle()
			except:
				pass
		self.plot_survivals(0)				


	def select_color(self, selection):
		colors = [self.cmap(0) if s else self.cmap(0.1) for s in selection]
		return colors

	def plot_spatial_location(self):

		try:
			self.sc = self.ax_scatter.scatter(self.df_pos_info["x"].values, self.df_pos_info["y"].values, picker=True, pickradius=1, color=self.select_color(self.df_pos_info["select"].values))
			self.scat_labels = self.df_pos_info['metadata_tag'].values
			self.ax_scatter.invert_xaxis()
			self.annot = self.ax_scatter.annotate("", xy=(0,0), xytext=(10,10),textcoords="offset points",
								bbox=dict(boxstyle="round", fc="w"),
								arrowprops=dict(arrowstyle="->"))
			self.annot.set_visible(False)
			self.fig_scatter.canvas.mpl_connect("motion_notify_event", self.hover)
			self.fig_scatter.canvas.mpl_connect("pick_event", self.unselect_position)
		except Exception as e:
			pass

	# def plot_positions(self):
		
	# 	# Load metadata, read X-Y positions
	# 	coordinates = []
	# 	line_index = []
	# 	label_annotations = []
	# 	coords_wells = []
	# 	for m in self.metafiles[:1]:
	# 		with open(m, 'r') as f:
	# 			data = json.load(f)
	# 			positions = data['Summary']['InitialPositionList']
	# 			for k in range(len(positions)):
	# 				pos_label = positions[k]['Label']
	# 				coords = positions[k]['DeviceCoordinatesUm']['XYStage']
	# 				for k,ks in enumerate(self.ks_estimators_per_position):
	# 					pos = ks['position']
	# 					pos_blocks = pos.split(os.sep)
	# 					pos_blocks.remove('')
	# 					well_id = pos_blocks[-3]
	# 					movies = glob(pos+f'movie{os.sep}{self.parent.parent.movie_prefix}*.tif')
	# 					if len(movies)>0:
	# 						file = movies[0]
	# 						if pos_label in file:
	# 							print(f"match for index {k} between position {pos} and coordinates {pos_label}")
	# 							coordinates.append(coords)
	# 							line_index.append(k)
	# 							label_annotations.append(pos_label)
	# 							coords_wells.append({'x': coords[0], 'y': coords[1], 'well': well_id})

	# 	coords_wells = pd.DataFrame(coords_wells)
	# 	well_coordinates = coords_wells.groupby('well').mean()[['x','y']].to_numpy()

	# 	coordinates = np.array(coordinates)
		
	# 	if self.plot_options[0].isChecked():
	# 		label_annotations_scat = list(coords_wells.groupby('well').mean().index)
	# 		self.position_colors = [tab10(0) for i in range(len(well_coordinates))]
	# 		self.sc = self.ax_scatter.scatter(well_coordinates[:,0], well_coordinates[:,1],picker=True, pickradius=1, color=self.position_colors)

	# 	elif self.plot_options[1].isChecked():
	# 		label_annotations_scat = label_annotations
	# 		self.position_colors = [tab10(0) for i in range(len(coordinates))]
	# 		self.sc = self.ax_scatter.scatter(coordinates[:,0], coordinates[:,1],picker=True, pickradius=1, color=self.position_colors)
		
	# 	self.ax_scatter.invert_xaxis()
	# 	annot = self.ax_scatter.annotate("", xy=(0,0), xytext=(10,10),textcoords="offset points",
	# 						bbox=dict(boxstyle="round", fc="w"),
	# 						arrowprops=dict(arrowstyle="->"))
	# 	annot.set_visible(False)

	# 	def update_annot(ind):
			
	# 		pos = self.sc.get_offsets()[ind["ind"][0]]
	# 		annot.xy = pos
	# 		text = label_annotations_scat[ind["ind"][0]]
	# 		# text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
	# 		# 					   " ".join([label_annotations[n] for n in ind["ind"]]))
	# 		annot.set_text(text)
	# 		annot.get_bbox_patch().set_facecolor('k')
	# 		annot.get_bbox_patch().set_alpha(0.4)

	# 	def hover(event):
	# 		vis = annot.get_visible()
	# 		if event.inaxes == self.ax_scatter:
	# 			cont, ind = self.sc.contains(event)
	# 			if cont:
	# 				update_annot(ind)
	# 				annot.set_visible(True)
	# 				self.fig_scatter.canvas.draw_idle()
	# 			else:
	# 				if vis:
	# 					annot.set_visible(False)
	# 					self.fig_scatter.canvas.draw_idle()

	# 	def unselect_position(event):
	# 		print(event)
	# 		ind = event.ind
	# 		print(ind)
	# 		if len(ind)>0:
	# 			ind = ind[0]
	# 			if self.position_colors[ind]==tab10(0.1):
	# 				self.position_colors[ind] = tab10(0)
	# 				if self.plot_options[1].isChecked():
	# 					self.line_to_plot[ind] = True # reselect line
	# 				elif self.plot_options[0].isChecked():
	# 					self.line_to_plot_well[ind] = True
	# 			else:
	# 				self.position_colors[ind] = tab10(0.1)
	# 				if self.plot_options[1].isChecked():
	# 					self.line_to_plot[ind] = False # unselect line
	# 				elif self.plot_options[0].isChecked():
	# 					self.line_to_plot_well[ind] = False
	# 			self.sc.set_color(self.position_colors)
	# 			self.position_scatter.canvas.draw_idle()
	# 			self.plot_survivals(0)

	# 	self.fig_scatter.canvas.mpl_connect("motion_notify_event", hover)
	# 	self.fig_scatter.canvas.mpl_connect("pick_event", unselect_position)

