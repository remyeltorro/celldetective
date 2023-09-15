from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QScrollArea, QButtonGroup, QComboBox, QFrame, QCheckBox, QFileDialog, QGridLayout, QTextEdit, QLineEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton, QRadioButton
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from celldetective.gui.gui_utils import center_window, FeatureChoice, ListWidget, QHSeperationLine, FigureCanvas, GeometryChoice, OperationChoice
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider,QLabeledSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from celldetective.utils import extract_experiment_channels, get_software_location, _extract_labels_from_config
from celldetective.io import interpret_tracking_configuration, load_frames, auto_load_number_of_frames
from celldetective.measure import compute_haralick_features, contour_of_instance_segmentation
import numpy as np
from tifffile import imread
import json
from shutil import copyfile
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
from natsort import natsorted
from tifffile import imread
from pathlib import Path, PurePath
import gc
import pandas as pd
from tqdm import tqdm
from lifelines import KaplanMeierFitter
from matplotlib.cm import viridis, tab10


def switch_to_events(classes, times, max_times, first_detections=None):
	
	events = []
	survival_times = []
	if first_detections is None:
		first_detections = np.zeros_like(max_times)
		
	for c,t,mt,ft in zip(classes, times, max_times, first_detections):
		if c==0:
			if t>0:
				events.append(1)
				survival_times.append(t - ft)
		elif c==1:
			events.append(0)
			survival_times.append(mt - ft)
		else:
			pass
	return events, survival_times
	  
	



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


		labels = [QLabel('population: '), QLabel('time of\ninterest: '), QLabel('time of\nreference: '), QLabel('class: ')]
		cb_options = [['targets','effectors'],['t0','first detection'], ['0','first detection'], ['class']]
		self.cbs = [QComboBox() for i in range(len(labels))]

		choice_layout = QVBoxLayout()
		choice_layout.setContentsMargins(20,20,20,20)
		for i in range(len(labels)):
			hbox = QHBoxLayout()
			hbox.addWidget(labels[i], 33)
			hbox.addWidget(self.cbs[i],66)
			self.cbs[i].addItems(cb_options[i])
			choice_layout.addLayout(hbox)
		main_layout.addLayout(choice_layout)

		self.submit_btn = QPushButton('Submit')
		self.submit_btn.setStyleSheet(self.parent.parent.parent.button_style_sheet)
		self.submit_btn.clicked.connect(self.process_survival)
		main_layout.addWidget(self.submit_btn)

		#self.populate_left_panel()
		#grid.addLayout(self.left_side, 0, 0, 1, 1)

		# self.setCentralWidget(self.scroll_area)
		# self.show()

	def process_survival(self):

		print('you clicked!!')

		# read instructions from combobox options
		self.load_available_tables()
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
		self.log_btn = QPushButton('')
		self.log_btn.setIcon(icon(MDI6.math_log,color="black"))
		self.log_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.log_btn.clicked.connect(self.switch_to_log)
		plot_buttons_hbox.addWidget(self.log_btn, alignment=Qt.AlignRight)

		self.fig, self.ax = plt.subplots(1,1,figsize=(4,3))
		self.survival_window = FigureCanvas(self.fig, title="Survival")
		self.survival_window.setContentsMargins(0,0,0,0)
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
		if len(self.well_indices)>1:		
			self.plot_btn_group.buttons()[0].click()
		else:
			self.plot_btn_group.buttons()[1].click()

		if self.position_indices is not None:
			for i in [0,2]:
				self.plot_options[i].setEnabled(False)


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
		self.pos_names = pd.DataFrame(self.ks_estimators_per_position)['position_name'].unique()
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
		else:
			self.pos_display_options = [QCheckBox(self.pos_names[i]) for i in range(len(self.pos_names))]
			for i in range(len(self.pos_names)):
				self.line_check_vbox.addWidget(self.pos_display_options[i], alignment=Qt.AlignLeft)
				self.pos_display_options[i].setChecked(True)
		self.plotvbox.addWidget(self.line_choice_widget, alignment=Qt.AlignCenter)

	def load_available_tables(self):

		"""
		Load the tables of the selected wells/positions from the control Panel for the population of interest

		"""

		self.df = []
		self.df_pos_info = []

		for widx,well_path in enumerate(tqdm(self.wells[self.well_indices])):

			well_index = widx
			split_well_path = well_path.split(os.sep)
			split_well_path = list(filter(None, split_well_path))
			well_name = split_well_path[-1]
			well_number = int(split_well_path[-1].replace('W',''))
			well_alias = self.well_labels[widx]

			positions = np.array(natsorted(glob(well_path+'*'+os.sep)),dtype=str)
			if self.position_indices is not None:
				positions = positions[self.position_indices]

			for pidx,pos_path in enumerate(positions):

				split_pos_path = pos_path.split(os.sep)
				split_pos_path = list(filter(None, split_pos_path))
				pos_name = split_pos_path[-1]
				table = os.sep.join([pos_path,'output','tables',f'trajectories_{self.cbs[0].currentText()}.csv'])

				movies = glob(pos_path+os.sep.join(['movie',self.parent.parent.movie_prefix+'*.tif']))
				if len(movies)>0:
					stack_path = movies[0]
				else:
					stack_path = np.nan

				if os.path.exists(table):
					df_pos = pd.read_csv(table, low_memory=False)
					df_pos['position'] = pos_path
					df_pos['well'] = well_path
					df_pos['well_index'] = well_number
					df_pos['well_name'] = well_name
					df_pos['pos_name'] = pos_name
					self.df.append(df_pos)

					self.df_pos_info.append({'pos_path': pos_path, 'pos_index': pidx, 'pos_name': pos_name, 'table_path': table, 'stack_path': stack_path,
											'well_path': well_path, 'well_index': well_index, 'well_name': well_name, 'well_number': well_number, 'well_alias': well_alias})

		self.df_pos_info = pd.DataFrame(self.df_pos_info)
		self.df_well_info = self.df_pos_info.loc[:,['well_path', 'well_index', 'well_name', 'well_number', 'well_alias']].drop_duplicates()
		self.df_well_info.to_csv(self.exp_dir+'exp_info_well.csv')

		if len(self.df)>0:
			self.df = pd.concat(self.df)
		else:
			print('No table could be found to compute survival...')
			return None

		print('End of new function...')



	# def load_available_tables(self):

	# 	"""
	# 	Load the tables of the selected wells/positions from the control Panel for the population of interest

	# 	"""

	# 	self.load_available_tables_v2()
	# 	self.df = []
	# 	for widx,w in enumerate(tqdm(self.wells[self.well_indices])):
			
	# 		split_w = w.split(os.sep)
	# 		split_w = list(filter(None, split_w))
	# 		well_nbr = int(split_w[-1].replace('W','')) - 1
	# 		well_name = split_w[-1]

	# 		positions = np.array(natsorted(glob(w+'*'+os.sep)),dtype=str)
	# 		if self.position_indices is not None:
	# 			# load only selected position
	# 			positions = positions[self.position_indices]

	# 		for pidx,pos in enumerate(positions):
	# 			split_pos = pos.split(os.sep)
	# 			split_pos = list(filter(None, split_pos))
	# 			pos_name = split_pos[-1]
	# 			table = os.sep.join([pos,'output','tables',f'trajectories_{self.cbs[0].currentText()}.csv'])
	# 			if os.path.exists(table):
	# 				df_pos = pd.read_csv(table, low_memory=False)
	# 				df_pos['position'] = pos
	# 				df_pos['well'] = w
	# 				df_pos['well_index'] = well_nbr
	# 				df_pos['well_name'] = well_name
	# 				df_pos['pos_name'] = pos_name
	# 				self.df.append(df_pos)

	# 	if len(self.df)>0:
	# 		self.df = pd.concat(self.df)
	# 	else:
	# 		print('No table could be found to compute survival...')
	# 		return None

	def compute_survival_functions(self):

		# Per position survival
		self.ks_estimators_per_position = []
		for block,movie_group in self.df.groupby(['well','position']):
			pos_blocks = block[1].split(os.sep)
			pos_blocks.remove('')
			well_id = int(pos_blocks[-3].replace('W','')) - 1
			classes = movie_group.groupby('TRACK_ID')[self.cbs[3].currentText()].min().values
			times = movie_group.groupby('TRACK_ID')[self.cbs[1].currentText()].min().values
			max_times = movie_group.groupby('TRACK_ID')['FRAME'].max().values
			first_detections = None
			split_pos = block[1].split(os.sep)
			split_pos = list(filter(None, split_pos))
			pos_name = split_pos[-1]
			if self.cbs[2].currentText()=='first detection':
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

			events, survival_times = switch_to_events(classes, times, max_times, first_detections)
			ks = KaplanMeierFitter()
			if len(events)>0:
				ks.fit(survival_times, event_observed=events)
				self.df_pos_info.loc[self.df_pos_info['pos_path']==block[1],'survival_fit'] = ks
				self.ks_estimators_per_position.append({'ks_estimator': ks, 'well': block[0], 'position': block[1], 'well_id': well_id, 'position_name': pos_name})

		self.line_to_plot = [True for i in range(len(self.ks_estimators_per_position))]
		# Per well survival

		self.ks_estimators_per_well = []

		for well,well_group in self.df.groupby('well'):

			split_w = well.split(os.sep)
			split_w = list(filter(None, split_w))
			well_id = int(split_w[-1].replace('W','')) - 1

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
				well_classes.extend(classes)
				well_times.extend(times)
				well_max_times.extend(max_times)
				if first_detections is not None:
					well_first_detections.extend(first_detections)  
			
			if len(well_first_detections)==0:
				well_first_detections = None
		  
			events, survival_times = switch_to_events(well_classes, well_times, well_max_times, well_first_detections)
			ks = KaplanMeierFitter()
			ks.fit(survival_times, event_observed=events)
			self.df_well_info.loc[self.df_well_info['well_path']==well,'survival_fit'] = ks
			self.ks_estimators_per_well.append({'ks_estimator': ks, 'well': well, 'well_id': well_id})	

		self.line_to_plot_well = [True for i in range(len(self.ks_estimators_per_well))]
		self.df_pos_info.loc[:,'select'] = True
		self.df_well_info.loc[:,'select'] = True

	def initialize_axis(self):

		self.ax.clear()
		self.ax.plot([],[])
		self.ax.spines['top'].set_visible(False)
		self.ax.spines['right'].set_visible(False)
		self.ax.set_ylim(0.001,1.05)
		self.ax.set_xlim(0,self.df['FRAME'].max())
		self.ax.set_xlabel('time [frame]')
		self.ax.set_ylabel('survival')

	def plot_survivals(self, id):

		for i in range(3):
			if self.plot_options[i].isChecked():
				self.plot_mode = self.radio_labels[i]

		colors = np.array([tab10(i / len(self.ks_estimators_per_position)) for i in range(len(self.ks_estimators_per_position))])
		well_color = [tab10(i / len(self.well_indices)) for i in range(len(self.well_indices) + 5)]

		if self.plot_mode=='pos':
			self.initialize_axis()
			for z,ks in enumerate([line for line,lp in zip(self.ks_estimators_per_position,self.line_to_plot) if lp]):
				if len(self.well_indices)<=1:
					sub_colors = colors[np.array(self.line_to_plot)]
					ks['ks_estimator'].plot_survival_function(ax=self.ax, legend=None, color=sub_colors[z])
				else:
					ks['ks_estimator'].plot_survival_function(ax=self.ax, legend=None, color=well_color[ks['well_id']])

		elif self.plot_mode=='well':
			self.initialize_axis()
			for ks in [line for line,lp in zip(self.ks_estimators_per_well,self.line_to_plot_well) if lp]:
				if len(self.well_indices)<=1:
					ks['ks_estimator'].plot_survival_function(ax=self.ax, legend=None, color="k")
				else:
					ks['ks_estimator'].plot_survival_function(ax=self.ax, legend=None, color=well_color[ks['well_id']])


		elif self.plot_mode=='both':
			self.initialize_axis()
			for z,ks in enumerate([line for line,lp in zip(self.ks_estimators_per_position,self.line_to_plot) if lp]):
				if len(self.well_indices)<=1:
					sub_colors = colors[np.array(self.line_to_plot)]
					ks['ks_estimator'].plot_survival_function(ax=self.ax, legend=None, alpha=0.25, color=sub_colors[z])
				else:
					ks['ks_estimator'].plot_survival_function(ci_show=False, ax=self.ax, legend=None, alpha=0.25, color=well_color[ks['well_id']])

			for ks in [line for line,lp in zip(self.ks_estimators_per_well,self.line_to_plot_well) if lp]:
				if len(self.well_indices)<=1:
					ks['ks_estimator'].plot_survival_function(ax=self.ax, legend=None,color="k")
				else:
					ks['ks_estimator'].plot_survival_function(ax=self.ax, legend=None, color=well_color[ks['well_id']])


		self.survival_window.canvas.draw()

	def switch_to_log(self):

		"""
		Switch threshold histogram to log scale. Auto adjust.
		"""

		if self.ax.get_yscale()=='linear':
			self.ax.set_yscale('log')
			self.ax.set_ylim(0.01,1.05)
		else:
			self.ax.set_yscale('linear')
			self.ax.set_ylim(0.01,1.05)

		#self.ax.autoscale()
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
			self.position_scatter.hide()
			self.line_choice_widget.show()
		else:
			self.position_scatter.show()
			self.line_choice_widget.hide()


	def load_coordinates(self):

		"""
		Read metadata and try to extract position coordinates
		"""

		try:
			with open(self.metafiles[0], 'r') as f:
				data = json.load(f)
				positions = data['Summary']['InitialPositionList']
		except Exception as e:
			print(f'Trouble loading metadata: error {e}...')
			return None

		self.position_coords = []
		for k in range(len(positions)):
			pos_label = positions[k]['Label']
			coords = positions[k]['DeviceCoordinatesUm']['XYStage']

			# Loop over each position survival curve
			for k,ks in enumerate(self.ks_estimators_per_position):
				well_id = ks['well_id']
				pos = ks['position']
				movies = glob(pos+os.sep.join(['movie',self.parent.parent.movie_prefix+'*.tif']))
				if len(movies)>0:
					file = movies[0]
					if pos_label in file:
						self.position_coords.append({'x': coords[0], 'y': coords[1], 'label': pos_label,'position': pos, 'well': well_id, 'select': True})

		self.position_coords = pd.DataFrame(self.position_coords)


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
		if len(ind)>0:
			ind = ind[0]
			if len(self.well_indices)>1:
				# auto switch all positions
				currentWell = self.position_coords.iloc[ind]['well']
				currentSelection = self.position_coords.loc[self.position_coords['well']==currentWell, 'select'].values
				self.position_coords.loc[self.position_coords['well']==currentWell, 'select'] = [not v for v in currentSelection]
				self.line_to_plot = self.position_coords["select"].values
			else:
				currentValue = bool(self.position_coords.iloc[ind]['select'])
				currentLabel = self.position_coords.iloc[ind]['label']
				self.position_coords.loc[self.position_coords['label']==currentLabel, 'select'] = not currentValue
				self.line_to_plot = self.position_coords["select"].values

			self.sc.set_color(self.select_color(self.position_coords["select"].values))
			self.position_scatter.canvas.draw_idle()
			self.plot_survivals(0)			

			# if self.position_colors[ind]==tab10(0.1):
			# 	self.position_colors[ind] = tab10(0)
			# 	if self.plot_options[1].isChecked():
			# 		self.line_to_plot[ind] = True # reselect line
			# 	elif self.plot_options[0].isChecked():
			# 		self.line_to_plot_well[ind] = True
			# else:
			# 	self.position_colors[ind] = tab10(0.1)
			# 	if self.plot_options[1].isChecked():
			# 		self.line_to_plot[ind] = False # unselect line
			# 	elif self.plot_options[0].isChecked():
			# 		self.line_to_plot_well[ind] = False
			# self.sc.set_color(self.position_colors)
			# self.position_scatter.canvas.draw_idle()
			# self.plot_survivals(0)


	def select_color(self, selection):
		colors = [tab10(0) if s else tab10(0.1) for s in selection]
		return colors

	def plot_spatial_location(self):

		self.sc = self.ax_scatter.scatter(self.position_coords["x"].values, self.position_coords["y"].values,picker=True, pickradius=1, color=self.select_color(self.position_coords["select"].values))
		self.scat_labels = self.position_coords['label'].values

		self.ax_scatter.invert_xaxis()
		self.annot = self.ax_scatter.annotate("", xy=(0,0), xytext=(10,10),textcoords="offset points",
							bbox=dict(boxstyle="round", fc="w"),
							arrowprops=dict(arrowstyle="->"))
		self.annot.set_visible(False)
		self.fig_scatter.canvas.mpl_connect("motion_notify_event", self.hover)
		self.fig_scatter.canvas.mpl_connect("pick_event", self.unselect_position)

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

