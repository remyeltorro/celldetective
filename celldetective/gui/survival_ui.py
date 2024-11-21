from PyQt5.QtWidgets import QMessageBox, QComboBox, QLineEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
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
from celldetective.gui import Styles
from matplotlib import colormaps
from celldetective.events import compute_survival

class ConfigSurvival(QWidget, Styles):
	
	"""
	UI to set survival instructions.

	"""

	def __init__(self, parent_window=None):
		
		super().__init__()
		self.parent_window = parent_window
		self.setWindowTitle("Configure survival")
		self.setWindowIcon(self.celldetective_icon)

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
		self.populate_widget()
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
		self.cb_options = [['targets','effectors'], ['0'], [], []] #['class'], 
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

		all_cms = list(colormaps)
		for cm in all_cms:
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

		time_cut_layout = QHBoxLayout()
		cut_time_lbl = QLabel('cut obs.\ntime [min]: ')
		cut_time_lbl.setToolTip('Filter out later events from\nthe analysis (in absolute time).')
		time_cut_layout.addWidget(cut_time_lbl, 33)
		self.query_time_cut = QLineEdit()
		self.query_time_cut.setValidator(self.float_validator)
		time_cut_layout.addWidget(self.query_time_cut, 66)
		main_layout.addLayout(time_cut_layout)

		self.set_classes_and_times()
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

		try:
			time_columns = list(self.all_columns[time_idx])
		except:
			print('no column starts with t')
			self.auto_close = True
			return None
		if 't0' in self.all_columns:
			time_columns.append('t0')

		self.cbs[1].clear()
		self.cbs[1].addItems(np.unique(self.cb_options[1]+time_columns))
		self.cbs[1].setCurrentText('t_firstdetection')

		self.cbs[2].clear()
		self.cbs[2].addItems(np.unique(self.cb_options[2]+time_columns))


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
			
			self.interpret_pos_location()
			if self.class_of_interest in list(self.df.columns) and self.cbs[2].currentText() in list(self.df.columns):
				self.compute_survival_functions()
			else:
				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Warning)
				msgBox.setText("The class and/or event time of interest is not found in the dataframe...")
				msgBox.setWindowTitle("Warning")
				msgBox.setStandardButtons(QMessageBox.Ok)
				returnValue = msgBox.exec()
				if returnValue == QMessageBox.Ok:
					return None
			if 'survival_fit' in list(self.df_pos_info.columns):
				self.plot_window = SurvivalPlotWidget(parent_window=self, df=self.df, df_pos_info = self.df_pos_info, df_well_info = self.df_well_info, title='plot survivals')
				self.plot_window.show()
			else:
				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Warning)
				msgBox.setText("No survival function was successfully computed...\nCheck your parameter choice.")
				msgBox.setWindowTitle("Warning")
				msgBox.setStandardButtons(QMessageBox.Ok)
				returnValue = msgBox.exec()
				if returnValue == QMessageBox.Ok:
					return None	

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
				return None		
		else:
			self.df_well_info = self.df_pos_info.loc[:,['well_path', 'well_index', 'well_name', 'well_number', 'well_alias']].drop_duplicates()
			#print(f"{self.df_well_info=}")

	def compute_survival_functions(self):

		cut_observation_time = None
		try:
			cut_observation_time = float(self.query_time_cut.text().replace(',','.')) / self.FrameToMin
			if not 0<cut_observation_time<=(self.df['FRAME'].max()):
				print('Invalid cut time (larger than movie length)... Not applied.')
				cut_observation_time = None		
		except Exception as e:
			pass
		print(f"{cut_observation_time=}")

		# Per position survival
		for block,movie_group in self.df.groupby(['well','position']):

			ks = compute_survival(movie_group, self.class_of_interest, self.cbs[2].currentText(), t_reference=self.cbs[1].currentText(), FrameToMin=self.FrameToMin, cut_observation_time=cut_observation_time)
			if ks is not None:
				self.df_pos_info.loc[self.df_pos_info['pos_path']==block[1],'survival_fit'] = ks

		# Per well survival
		for well,well_group in self.df.groupby('well'):

			ks = compute_survival(well_group, self.class_of_interest, self.cbs[2].currentText(), t_reference=self.cbs[1].currentText(), FrameToMin=self.FrameToMin, cut_observation_time=cut_observation_time)
			if ks is not None:
				self.df_well_info.loc[self.df_well_info['well_path']==well,'survival_fit'] = ks

		self.df_pos_info.loc[:,'select'] = True
		self.df_well_info.loc[:,'select'] = True