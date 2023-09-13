from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QScrollArea, QComboBox, QFrame, QCheckBox, QFileDialog, QGridLayout, QTextEdit, QLineEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from celldetective.gui.gui_utils import center_window, FeatureChoice, ListWidget, QHSeperationLine, FigureCanvas, GeometryChoice, OperationChoice
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider,QLabeledSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from celldetective.utils import extract_experiment_channels, get_software_location
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
		#self.config_path = self.exp_dir + self.config_name

		self.screen_height = self.parent.parent.parent.screen_height
		center_window(self)

		self.setMinimumWidth(350)
		#self.setMinimumHeight(int(0.3*self.screen_height))
		#self.setMaximumHeight(int(0.8*self.screen_height))
		self.populate_widget()
		#self.load_previous_measurement_instructions()

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

		self.fig, self.ax = plt.subplots(1,1,figsize=(4,3))
		self.survival_window = FigureCanvas(self.fig, title="Survival")
		self.ax.clear()
		self.ax.plot([],[])
		self.ax.spines['top'].set_visible(False)
		self.ax.spines['right'].set_visible(False)
		self.ax.set_ylim(0.001,1.05)
		self.ax.set_xlim(0,self.df['FRAME'].max())
		self.ax.set_xlabel('time [frame]')
		self.ax.set_ylabel('survival')
		plt.tight_layout()
		self.fig.set_facecolor('none')  # or 'None'
		self.fig.canvas.setStyleSheet("background-color: transparent;")
		self.survival_window.canvas.draw()

		self.survival_window.layout.addWidget(QLabel('WHAAAAATTT???'))

		if self.df is not None and len(self.ks_estimators_per_position)>0:
			self.plot_survivals()


		self.survival_window.show()

	def load_available_tables(self):

		self.well_paths = natsorted(glob(self.exp_dir+'W*'+os.sep))
		self.df = []
		for widx,w in enumerate(tqdm(self.well_paths)):
			split_w = w.split(os.sep)
			split_w = list(filter(None, split_w))
			well_nbr = int(split_w[-1].replace('W','')) - 1
			positions = natsorted(glob(w+'*'+os.sep))
			for pidx,pos in enumerate(positions):
				table = os.sep.join([pos,'output','tables',f'trajectories_{self.cbs[0].currentText()}.csv'])
				if os.path.exists(table):
					df_pos = pd.read_csv(table)
					df_pos['position'] = pos
					df_pos['well'] = w
					df_pos['well_index'] = well_nbr
					self.df.append(df_pos)
		if len(self.df)>0:
			self.df = pd.concat(self.df)
			print(self.df.head(10))
		else:
			print('No table could be found to compute survival...')
			return None

	def compute_survival_functions(self):

		# Per position survival
		self.ks_estimators_per_position = []
		for block,movie_group in self.df.groupby(['well','position']):

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

			events, survival_times = switch_to_events(classes, times, max_times, first_detections)
			ks = KaplanMeierFitter()
			if len(events)>0:
				ks.fit(survival_times, event_observed=events)
				self.ks_estimators_per_position.append({'ks_estimator': ks, 'well': block[0], 'position': block[1]})

		# Per well survival

		self.ks_estimators_per_well = []

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
			self.ks_estimators_per_well.append({'ks_estimator': ks, 'well': well})		

	def plot_survivals(self):
		self.plot_mode = 'well'
		if self.plot_mode=='pos':
			for ks in self.ks_estimators_per_position:
				ks['ks_estimator'].plot_survival_function(ax=self.ax, legend=None)
		else:
			for ks in self.ks_estimators_per_well:
				ks['ks_estimator'].plot_survival_function(ax=self.ax, legend=None)			
		self.survival_window.canvas.draw()


