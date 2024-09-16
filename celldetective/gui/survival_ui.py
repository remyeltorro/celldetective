from PyQt5.QtWidgets import QMessageBox, QComboBox, QLineEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QDoubleValidator
from celldetective.gui.gui_utils import center_window
from superqt import QColormapComboBox
from celldetective.gui.generic_signal_plot import SurvivalPlotWidget
from celldetective.utils import get_software_location, _extract_labels_from_config
from celldetective.io import load_experiment_tables
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from glob import glob
import pandas as pd
from lifelines import KaplanMeierFitter
from celldetective.events import switch_to_events
from celldetective.gui import Styles
from matplotlib import colormaps

class ConfigSurvival(QWidget, Styles):
	
	"""
	UI to set survival instructions.

	"""

	def __init__(self, parent_window=None):
		
		super().__init__()
		self.parent_window = parent_window
		self.setWindowTitle("Configure survival")
		self.setWindowIcon(QIcon(os.sep.join(['celldetective','icons','logo.png'])))

		self.exp_dir = self.parent_window.exp_dir
		self.soft_path = get_software_location()		
		self.exp_config = self.exp_dir +"config.ini"
		self.wells = np.array(self.parent_window.parent_window.wells,dtype=str)
		self.well_labels = _extract_labels_from_config(self.exp_config,len(self.wells))
		self.FrameToMin = self.parent_window.parent_window.FrameToMin
		self.float_validator = QDoubleValidator()
		self.auto_close = False

		self.well_option = self.parent_window.parent_window.well_list.currentIndex()
		self.position_option = self.parent_window.parent_window.position_list.currentIndex()
		self.interpret_pos_location()
		#self.config_path = self.exp_dir + self.config_name

		self.screen_height = self.parent_window.parent_window.parent_window.screen_height
		center_window(self)

		self.setMinimumWidth(350)
		#self.setMinimumHeight(int(0.8*self.screen_height))
		#self.setMaximumHeight(int(0.8*self.screen_height))
		self.populate_widget()
		#self.load_previous_measurement_instructions()
		if self.auto_close:
			self.close()
		self.setAttribute(Qt.WA_DeleteOnClose)

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


		labels = [QLabel('population: '), QLabel('time of\nreference: '), QLabel('time of\ninterest: '), QLabel('cmap: ')] #QLabel('class: '), 
		self.cb_options = [['targets','effectors'], ['0'], [], list(plt.colormaps())] #['class'], 
		self.cbs = [QComboBox() for i in range(len(labels))]

		self.cbs[-1] = QColormapComboBox()
		self.cbs[0].currentIndexChanged.connect(self.set_classes_and_times)

		choice_layout = QVBoxLayout()
		choice_layout.setContentsMargins(20,20,20,20)
		for i in range(len(labels)):
			hbox = QHBoxLayout()
			hbox.addWidget(labels[i], 33)
			hbox.addWidget(self.cbs[i],66)
			if i < len(labels)-1:
				self.cbs[i].addItems(self.cb_options[i])
			choice_layout.addLayout(hbox)

		for cm in list(colormaps):
			try:
				self.cbs[-1].addColormap(cm)
			except:
				pass

		main_layout.addLayout(choice_layout)

		select_layout = QHBoxLayout()
		select_layout.addWidget(QLabel('select cells\nwith query: '), 33)
		self.query_le = QLineEdit()
		select_layout.addWidget(self.query_le, 66)
		main_layout.addLayout(select_layout)

		self.cbs[0].setCurrentIndex(0)
		self.cbs[1].setCurrentText('t_firstdetection')

		time_calib_layout = QHBoxLayout()
		time_calib_layout.setContentsMargins(20,20,20,20)
		time_calib_layout.addWidget(QLabel('time calibration\n(frame to min)'), 33)
		self.time_calibration_le = QLineEdit(str(self.FrameToMin).replace('.',','))
		self.time_calibration_le.setValidator(self.float_validator)
		time_calib_layout.addWidget(self.time_calibration_le, 66)
		#time_calib_layout.addWidget(QLabel(' min'))
		main_layout.addLayout(time_calib_layout)

		self.submit_btn = QPushButton('Submit')
		self.submit_btn.setStyleSheet(self.button_style_sheet)
		self.submit_btn.clicked.connect(self.process_survival)
		main_layout.addWidget(self.submit_btn)

		#self.populate_left_panel()
		#grid.addLayout(self.left_side, 0, 0, 1, 1)

		# self.setCentralWidget(self.scroll_area)
		# self.show()

	def set_classes_and_times(self):

		# Look for all classes and times
		tables = glob(self.exp_dir+os.sep.join(['W*','*','output','tables',f'trajectories_*.csv']))
		self.all_columns = []
		for tab in tables:
			cols = pd.read_csv(tab, nrows=1,encoding_errors='ignore').columns.tolist()
			self.all_columns.extend(cols)
		self.all_columns = np.unique(self.all_columns)
		#class_idx = np.array([s.startswith('class_') for s in self.all_columns])
		time_idx = np.array([s.startswith('t_') for s in self.all_columns])
		class_idx = np.array([s.startswith('class') for s in self.all_columns])

		# class_columns = list(self.all_columns[class_idx])
		# for c in ['class_id', 'class_color']:
		# 	if c in class_columns:
		# 		class_columns.remove(c)

		try:
			time_columns = list(self.all_columns[time_idx])
		except:
			print('no column starts with t')
			self.auto_close = True
			return None

		try:
			class_columns = list(self.all_columns[class_idx])
			self.cbs[3].clear()
			self.cbs[3].addItems(np.unique(self.cb_options[3]+class_columns))
		except:
			print('no column starts with class')
			self.auto_close = True
			return None

		self.cbs[2].clear()
		self.cbs[2].addItems(np.unique(self.cb_options[2]+time_columns))

		self.cbs[1].clear()
		self.cbs[1].addItems(np.unique(self.cb_options[1]+time_columns))
		self.cbs[1].setCurrentText('t_firstdetection')

		# self.cbs[3].clear()
		# self.cbs[3].addItems(np.unique(self.cb_options[3]+class_columns))
		

	def process_survival(self):

		print('you clicked!!')
		self.FrameToMin = float(self.time_calibration_le.text().replace(',','.'))
		print(self.FrameToMin, 'set')

		self.time_of_interest = self.cbs[2].currentText()
		if self.time_of_interest=="t0":
			self.class_of_interest = "class"
		else:
			self.class_of_interest = self.time_of_interest.replace('t_','class_')

		# read instructions from combobox options
		self.load_available_tables_local()

		if self.df is not None:
			
			try:
				query_text = self.query_le.text()
				if query_text != '':
					self.df = self.df.query(query_text)
			except Exception as e:
				print(e, ' The query is misunderstood and will not be applied...')

			self.compute_survival_functions()
			# prepare survival
			self.interpret_pos_location()
			self.plot_window = SurvivalPlotWidget(parent_window=self, df=self.df, df_pos_info = self.df_pos_info, df_well_info = self.df_well_info, title='plot survivals')
			self.plot_window.show()

	def load_available_tables_local(self):

		"""
		Load the tables of the selected wells/positions from the control Panel for the population of interest

		"""

		self.well_option = self.parent_window.parent_window.well_list.currentIndex()
		if self.well_option==len(self.wells):
			wo = '*'
		else:
			wo = self.well_option
		self.position_option = self.parent_window.parent_window.position_list.currentIndex()
		if self.position_option==0:
			po = '*'
		else:
			po = self.position_option - 1

		self.df, self.df_pos_info = load_experiment_tables(self.exp_dir, well_option=wo, position_option=po, population=self.cbs[0].currentText(), return_pos_info=True)
		if self.df is None:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("No table could be found.. Abort.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				self.close()
				return None		
			print('no table could be found...')
		else:
			self.df_well_info = self.df_pos_info.loc[:,['well_path', 'well_index', 'well_name', 'well_number', 'well_alias']].drop_duplicates()
			#print(f"{self.df_well_info=}")

	def compute_survival_functions(self):

		# Per position survival
		left_censored = False
		for block,movie_group in self.df.groupby(['well','position']):
			try:
				classes = movie_group.groupby('TRACK_ID')[self.class_of_interest].min().values
				times = movie_group.groupby('TRACK_ID')[self.cbs[2].currentText()].min().values
			except Exception as e:
				print(e)
				continue
			max_times = movie_group.groupby('TRACK_ID')['FRAME'].max().values
			first_detections = None
			
			if self.cbs[1].currentText()=='first detection':
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

			elif self.cbs[1].currentText().startswith('t'):
				left_censored = True
				first_detections = movie_group.groupby('TRACK_ID')[self.cbs[1].currentText()].max().values
				print(first_detections)


			if self.cbs[1].currentText()=='first detection' or self.cbs[1].currentText().startswith('t'):
				left_censored = True
			else:
				left_censored = False
			events, survival_times = switch_to_events(classes, times, max_times, first_detections, left_censored=left_censored, FrameToMin=self.FrameToMin)
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
				try:
					classes = movie_group.groupby('TRACK_ID')[self.class_of_interest].min().values
					times = movie_group.groupby('TRACK_ID')[self.cbs[2].currentText()].min().values
				except Exception as e:
					print(e)
					continue
				max_times = movie_group.groupby('TRACK_ID')['FRAME'].max().values
				first_detections = None

				if self.cbs[1].currentText()=='first detection':
					
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

				elif self.cbs[1].currentText().startswith('t'):
					left_censored = True
					first_detections = movie_group.groupby('TRACK_ID')[self.cbs[1].currentText()].max().values

				else:
					pass

				well_classes.extend(classes)
				well_times.extend(times)
				well_max_times.extend(max_times)
				if first_detections is not None:
					well_first_detections.extend(first_detections)  
			
			if len(well_first_detections)==0:
				well_first_detections = None
			
			print(f"{well_classes=}; {well_times=}")
			events, survival_times = switch_to_events(well_classes, well_times, well_max_times, well_first_detections,left_censored=left_censored, FrameToMin=self.FrameToMin)
			print(f"{events=}; {survival_times=}")
			ks = KaplanMeierFitter()
			if len(survival_times)>0:
				ks.fit(survival_times, event_observed=events)
				print(ks.survival_function_)
			else:
				ks = None
			print(f"{ks=}")
			self.df_well_info.loc[self.df_well_info['well_path']==well,'survival_fit'] = ks

		self.df_pos_info.loc[:,'select'] = True
		self.df_well_info.loc[:,'select'] = True