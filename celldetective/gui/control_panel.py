from PyQt5.QtWidgets import QMainWindow, QComboBox, QPushButton, QLabel, QWidget, QGridLayout, QFrame, QTabWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QSize
from celldetective.gui.gui_utils import center_window, QHSeperationLine
from celldetective.utils import _extract_labels_from_config, ConfigSectionMap
from celldetective.gui import ConfigEditor, ProcessPanel
from natsort import natsorted
from glob import glob
import os
import numpy as np
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import gc

class ControlPanel(QMainWindow):

	def __init__(self, parent=None, exp_dir=""):
		
		super().__init__()
		self.exp_dir = exp_dir
		if not self.exp_dir.endswith('/'):
			self.exp_dir = self.exp_dir+'/'
		self.setWindowTitle("celldetective")
		self.parent = parent
		center_window(self)

		self.init_wells_and_positions()
		self.load_configuration()

		self.w = QWidget()
		self.grid = QGridLayout(self.w)
		self.grid.setSpacing(5)
		self.grid.setContentsMargins(20,20,20,20) #left top right bottom

		self.to_disable = []
		self.generate_header()
		self.ProcessEffectors = ProcessPanel(self,'effectors')
		self.ProcessTargets = ProcessPanel(self,'targets')
		
		ProcessFrame = QFrame()
		grid_process = QVBoxLayout(ProcessFrame)
		grid_process.setContentsMargins(20,50,20,20)

		grid_process.addWidget(self.ProcessEffectors)
		grid_process.addWidget(self.ProcessTargets)


		tabWidget = QTabWidget()
		tab_index_process = tabWidget.addTab(ProcessFrame, "Process")
		tabWidget.setTabIcon(tab_index_process, icon(MDI6.cog_outline, color='black'))

		#tabWidget.addTab(CheckFrame, "Control")
		#tab_index_analyze = tabWidget.addTab(AnalyzeFrame, "Analyze")
		#tabWidget.setTabIcon(tab_index_analyze, icon(MDI6.chart_bell_curve, color='black'))
		tabWidget.setStyleSheet(self.parent.qtab_style)

		self.grid.addWidget(tabWidget, 7,0,1,3, alignment=Qt.AlignTop)
		self.grid.setSpacing(5)

		self.setCentralWidget(self.w)

	def init_wells_and_positions(self):

		"""
		Detect the wells in the experiment folder and the associated positions.
		"""
		
		self.wells = natsorted(glob(self.exp_dir + "W*/"))
		self.positions = []
		for w in self.wells:
			w = os.path.split(w[:-1])
			root = w[0]
			w = w[1]
			positions_path = natsorted(glob(root+os.sep+w+os.sep+f"{w[1]}*/"))
			self.positions.append([os.path.split(pos[:-1])[1] for pos in positions_path])

	def generate_header(self):
		
		"""
		Show the experiment name, create two QComboBox for respectively the 
		biological condition (well) and position of interest, access the experiment config.
		"""

		condition_label = QLabel("condition: ")
		position_label = QLabel("position: ")

		name = self.exp_dir.split("/")[-2]
		experiment_label = QLabel(f"Experiment:")
		experiment_label.setStyleSheet("""
			font-weight: bold;
			""")
		self.grid.addWidget(experiment_label, 0,0,1,1, alignment=Qt.AlignRight)
		self.grid.addWidget(QLabel(name), 0,1,1,1, alignment=Qt.AlignLeft)

		self.edit_config_button = QPushButton()
		self.edit_config_button.setIcon(icon(MDI6.cog_outline,color="black"))
		self.edit_config_button.setIconSize(QSize(20, 20))
		self.edit_config_button.setToolTip("Configuration file")
		self.edit_config_button.clicked.connect(self.open_config_editor)
		self.edit_config_button.setStyleSheet(self.parent.button_select_all)
		self.grid.addWidget(self.edit_config_button, 0,0,1,3, alignment=Qt.AlignRight)

		self.well_list = QComboBox()
		self.well_list.addItems(self.well_labels)
		self.well_list.addItems(["*"])
		self.well_list.activated.connect(self.display_positions)
		self.to_disable.append(self.well_list)

		self.position_list = QComboBox()
		self.position_list.addItems(["*"])
		self.position_list.addItems(self.positions[0])
		self.position_list.activated.connect(self.update_position_options)
		self.to_disable.append(self.position_list)


		self.grid.addWidget(QLabel("Well:"), 1, 0, 1,1, alignment=Qt.AlignRight)
		self.grid.addWidget(self.well_list, 1, 1, 1, 2)
		self.grid.addWidget(QLabel("Position:"),2,0,1,1, alignment=Qt.AlignRight)
		self.grid.addWidget(self.position_list, 2,1,1,2)

		
		hsep = QHSeperationLine()
		self.grid.addWidget(hsep, 5, 0, 1, 3)

	def load_configuration(self):

		'''
		This methods load the configuration read in the config.ini file of the experiment.
		'''

		self.exp_config = self.exp_dir + "config.ini"
		self.PxToUm = float(ConfigSectionMap(self.exp_config,"MovieSettings")["pxtoum"])
		self.FrameToMin = float(ConfigSectionMap(self.exp_config,"MovieSettings")["frametomin"])
		self.len_movie = int(ConfigSectionMap(self.exp_config,"MovieSettings")["len_movie"])
		self.shape_x = int(ConfigSectionMap(self.exp_config,"MovieSettings")["shape_x"])
		self.shape_y = int(ConfigSectionMap(self.exp_config,"MovieSettings")["shape_y"])
		self.movie_prefix = ConfigSectionMap(self.exp_config,"MovieSettings")["movie_prefix"]

		# Read channels
		self.nbr_channels = 0
		self.exp_channels = []
		try:
			self.brightfield_channel = int(ConfigSectionMap(self.exp_config,"MovieSettings")["brightfield_channel"])
			self.nbr_channels += 1
			self.exp_channels.append("brightfield_channel")
		except:
			self.brightfield_channel = None

		try:
			self.live_nuclei_channel = int(ConfigSectionMap(self.exp_config,"MovieSettings")["live_nuclei_channel"])
			self.nbr_channels += 1
			self.exp_channels.append("live_nuclei_channel")
		except:
			self.live_nuclei_channel = None

		try:
			self.dead_nuclei_channel = int(ConfigSectionMap(self.exp_config,"MovieSettings")["dead_nuclei_channel"])
			self.nbr_channels +=1
			self.exp_channels.append("dead_nuclei_channel")
		except:
			self.dead_nuclei_channel = None

		try:
			self.effector_fluo_channel = int(ConfigSectionMap(self.exp_config,"MovieSettings")["effector_fluo_channel"])
			self.nbr_channels +=1
			self.exp_channels.append("effector_fluo_channel")
		except:
			self.effector_fluo_channel = None

		try:
			self.adhesion_channel = int(ConfigSectionMap(self.exp_config,"MovieSettings")["adhesion_channel"])
			self.nbr_channels += 1
			self.exp_channels.append("adhesion_channel")
		except:
			self.adhesion_channel = None

		try:
			self.fluo_channel_1 = int(ConfigSectionMap(self.exp_config,"MovieSettings")["fluo_channel_1"])
			self.nbr_channels += 1
			self.exp_channels.append("fluo_channel_1")
		except:
			self.fluo_channel_1 = None	
	
		try:
			self.fluo_channel_2 = int(ConfigSectionMap(self.exp_config,"MovieSettings")["fluo_channel_2"])
			self.nbr_channels += 1
			self.exp_channels.append("fluo_channel_2")
		except:
			self.fluo_channel_2 = None			

		self.search_radius_tc = int(ConfigSectionMap(self.exp_config,"SearchRadii")["search_radius_tc"])
		self.search_radius_nk = int(ConfigSectionMap(self.exp_config,"SearchRadii")["search_radius_nk"])
		self.time_dilation = int(ConfigSectionMap(self.exp_config,"BinningParameters")["time_dilation"])

		self.intensity_measurement_radius = int(ConfigSectionMap(self.exp_config,"Thresholds")["intensity_measurement_radius"])
		self.intensity_measurement_radius_nk = int(ConfigSectionMap(self.exp_config,"Thresholds")["intensity_measurement_radius_nk"])
		self.model_signal_length = int(ConfigSectionMap(self.exp_config,"Thresholds")["model_signal_length"])

		try:
			self.hide_frames_for_tracking = np.array([int(s) for s in ConfigSectionMap(config,"Thresholds")["hide_frames_for_tracking"].split(",")])
		except:
			self.hide_frames_for_tracking = np.array([])

		number_of_wells = len(self.wells)
		self.well_labels = _extract_labels_from_config(self.exp_config,number_of_wells)

		self.concentrations = ConfigSectionMap(self.exp_config,"Labels")["concentrations"].split(",")
		if number_of_wells != len(self.concentrations):
			self.concentrations = [str(s) for s in np.linspace(0,number_of_wells-1,number_of_wells)]		

		#self.antibodies = extract_labels_from_config(config, number_of_wells)

		self.cell_types = ConfigSectionMap(self.exp_config,"Labels")["cell_types"].split(",")
		if number_of_wells != len(self.cell_types):
			self.cell_types = [str(s) for s in np.linspace(0,number_of_wells-1,number_of_wells)]		

		try:
			self.antibodies = ConfigSectionMap(self.exp_config,"Labels")["antibodies"].split(",")
			if number_of_wells != len(self.antibodies):
				self.antibodies = [str(s) for s in np.linspace(0,number_of_wells-1,number_of_wells)]
		except:
			print("Warning... antibodies not found...")
			self.antibodies = [str(s) for s in np.linspace(0,number_of_wells-1,number_of_wells)]

		try:
			self.pharmaceutical_agents = ConfigSectionMap(self.exp_config,"Labels")["pharmaceutical_agents"].split(",")
			if number_of_wells != len(self.pharmaceutical_agents):
				self.pharmaceutical_agents = [str(s) for s in np.linspace(0,number_of_wells-1,number_of_wells)]
		except:
			print("Warning... pharmaceutical agents not found...")
			self.pharmaceutical_agents = [str(s) for s in np.linspace(0,number_of_wells-1,number_of_wells)]

		#self.modelpath = abs_path+"/models/" #ConfigSectionMap(config,"Paths")["modelpath"]

	def closeEvent(self, event):

		"""
		Close child windows if closed.
		"""
		
		try:
			if self.cfg_editor:
				self.cfg_editor.close()
		except:
			pass

		try:
			if self.ProcessTargets.SegModelLoader:
				self.ProcessTargets.SegModelLoader.close()
		except:
			pass

		try:
			if self.ProcessEffectors.SegModelLoader:
				self.ProcessEffectors.SegModelLoader.close()
		except:
			pass

		gc.collect()


	def display_positions(self):

		"""
		Show the positions as the well is changed.
		"""

		if self.well_list.currentText()=="*":
			self.position_list.clear()
			self.position_list.addItems(["*"])
			position_linspace = np.linspace(0,len(self.positions[0])-1,len(self.positions[0]),dtype=int)
			position_linspace = [str(s) for s in position_linspace]
			self.position_list.addItems(position_linspace)
		else:
			pos_index = self.well_labels.index(str(self.well_list.currentText()))
			self.position_list.clear()
			self.position_list.addItems(["*"])
			self.position_list.addItems(self.positions[pos_index])
	
	def open_config_editor(self):
		self.cfg_editor = ConfigEditor(self)
		self.cfg_editor.show()

	def locate_selected_position(self):

		"""
		Set the current position if the option one well, one positon is selected
		Display error messages otherwise.

		"""

		if self.well_list.currentText()=="*":
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Critical)
			msgBox.setText("Please select a single well...")
			msgBox.setWindowTitle("Error")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None
		else:
			self.well_index = [self.well_labels.index(str(self.well_list.currentText()))]

		for w_idx in self.well_index:

			pos = self.positions[w_idx]
			if self.position_list.currentText()=="*":
				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Critical)
				msgBox.setText("Please select a single position...")
				msgBox.setWindowTitle("Error")
				msgBox.setStandardButtons(QMessageBox.Ok)
				returnValue = msgBox.exec()
				if returnValue == QMessageBox.Ok:
					return None
			else:
				pos_indices = natsorted([pos.index(self.position_list.currentText())])

			well = self.wells[w_idx]

			for pos_idx in pos_indices:
				self.pos = natsorted(glob(well+f"{well[-2]}*/"))[pos_idx]

	def update_position_options(self):
		
		self.pos = self.position_list.currentText()
		panels = [self.ProcessEffectors, self.ProcessTargets]
		if self.position_list.currentText()=="*":
			for p in panels:
				p.check_seg_btn.setEnabled(False)
		else:
			if not self.well_list.currentText()=="*":
				self.locate_selected_position()
				if os.path.exists(self.pos+'/labels_effectors/'):
					self.ProcessEffectors.check_seg_btn.setEnabled(True)
				else:
					self.ProcessEffectors.check_seg_btn.setEnabled(False)
				if os.path.exists(self.pos+'/labels_targets/'):
					self.ProcessTargets.check_seg_btn.setEnabled(True)
				else:
					self.ProcessTargets.check_seg_btn.setEnabled(False)
