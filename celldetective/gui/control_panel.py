from PyQt5.QtWidgets import QMainWindow, QComboBox, QPushButton,QHBoxLayout, QLabel, QWidget, QGridLayout, QFrame, QTabWidget, QVBoxLayout, QMessageBox
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from celldetective.gui.gui_utils import center_window, QHSeperationLine
from celldetective.utils import _extract_labels_from_config, ConfigSectionMap
from celldetective.gui import ConfigEditor, ProcessPanel, AnalysisPanel
from natsort import natsorted
from glob import glob
import os
import numpy as np
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import gc
import subprocess

class ControlPanel(QMainWindow):

	def __init__(self, parent=None, exp_dir=""):
		
		super().__init__()
		self.exp_dir = exp_dir
		if not self.exp_dir.endswith(os.sep):
			self.exp_dir = self.exp_dir+os.sep
		self.setWindowTitle("celldetective")
		self.setWindowIcon(QIcon(os.sep.join(['celldetective','icons','mexican-hat.png'])))
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

		AnalyzeFrame = QFrame()
		grid_analyze = QVBoxLayout(AnalyzeFrame)
		grid_analyze.setContentsMargins(20,50,20,20)
		self.SurvivalBlock = AnalysisPanel(self,title='Survival')


		grid_process.addWidget(self.ProcessEffectors)
		grid_process.addWidget(self.ProcessTargets)

		grid_analyze.addWidget(self.SurvivalBlock)


		tabWidget = QTabWidget()
		tab_index_process = tabWidget.addTab(ProcessFrame, "Process")
		tabWidget.setTabIcon(tab_index_process, icon(MDI6.cog_outline, color='black'))

		tab_index_analyze = tabWidget.addTab(AnalyzeFrame, "Analyze")
		tabWidget.setTabIcon(tab_index_analyze, icon(MDI6.poll, color='black'))
		tabWidget.setStyleSheet(self.parent.qtab_style)

		self.grid.addWidget(tabWidget, 7,0,1,3, alignment=Qt.AlignTop)
		self.grid.setSpacing(5)

		self.setCentralWidget(self.w)
		self.create_config_dir()

	def init_wells_and_positions(self):

		"""
		Detect the wells in the experiment folder and the associated positions.
		"""
		
		self.wells = natsorted(glob(self.exp_dir + "W*" + os.sep))
		self.positions = []
		for w in self.wells:
			w = os.path.split(w[:-1])
			root = w[0]
			w = w[1]
			positions_path = natsorted(glob(os.sep.join([root, w, f"{w[-1]}*{os.sep}"])))
			self.positions.append([os.path.split(pos[:-1])[1] for pos in positions_path])

	def generate_header(self):
		
		"""
		Show the experiment name, create two QComboBox for respectively the 
		biological condition (well) and position of interest, access the experiment config.
		"""

		condition_label = QLabel("condition: ")
		position_label = QLabel("position: ")

		name = self.exp_dir.split(os.sep)[-2]
		experiment_label = QLabel(f"Experiment:")
		experiment_label.setStyleSheet("""
			font-weight: bold;
			""")

		self.folder_exp_btn = QPushButton()
		self.folder_exp_btn.setIcon(icon(MDI6.folder,color="black"))
		self.folder_exp_btn.setIconSize(QSize(20, 20))
		self.folder_exp_btn.setToolTip("Open experiment folder on disk")
		self.folder_exp_btn.clicked.connect(self.open_experiment_folder)
		self.folder_exp_btn.setStyleSheet(self.parent.button_select_all)


		self.edit_config_button = QPushButton()
		self.edit_config_button.setIcon(icon(MDI6.cog_outline,color="black"))
		self.edit_config_button.setIconSize(QSize(20, 20))
		self.edit_config_button.setToolTip("Configuration file")
		self.edit_config_button.clicked.connect(self.open_config_editor)
		self.edit_config_button.setStyleSheet(self.parent.button_select_all)

		self.exp_options_layout = QHBoxLayout()
		self.exp_options_layout.addWidget(experiment_label, 32, alignment=Qt.AlignRight)
		self.exp_options_layout.addWidget(QLabel(name), 65, alignment=Qt.AlignLeft)
		self.exp_options_layout.addWidget(self.folder_exp_btn, 5, alignment=Qt.AlignRight)
		self.exp_options_layout.addWidget(self.edit_config_button, 5, alignment=Qt.AlignRight)
		self.grid.addLayout(self.exp_options_layout, 0,0,1,3)

		self.well_list = QComboBox()
		thresh = 40
		self.well_truncated = [w[:thresh - 3]+'...' if len(w)>thresh else w for w in self.well_labels]		
		self.well_list.addItems(self.well_truncated) #self.well_labels
		for i in range(len(self.well_labels)):
			self.well_list.setItemData(i, self.well_labels[i], Qt.ToolTipRole)
		self.well_list.addItems(["*"])
		self.well_list.activated.connect(self.display_positions)
		self.to_disable.append(self.well_list)

		self.position_list = QComboBox()
		self.position_list.addItems(["*"])
		self.position_list.addItems(self.positions[0])
		self.position_list.activated.connect(self.update_position_options)
		self.to_disable.append(self.position_list)
		#self.locate_selected_position()

		self.grid.addWidget(QLabel("Well:"), 1, 0, 1,1, alignment=Qt.AlignRight)
		self.grid.addWidget(self.well_list, 1, 1, 1, 2)

		self.grid.addWidget(QLabel("Position:"),2,0,1,1, alignment=Qt.AlignRight)
		self.grid.addWidget(self.position_list, 2,1,1,2)

		
		hsep = QHSeperationLine()
		self.grid.addWidget(hsep, 5, 0, 1, 3)

	def open_experiment_folder(self):

		try:
			subprocess.Popen(f'explorer {os.path.realpath(self.exp_dir)}')
		except:
			try:
				os.system('xdg-open "%s"' % self.exp_dir)
			except:
				return None		

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

		try:
			self.search_radius_targets = int(ConfigSectionMap(self.exp_config,"SearchRadii")["search_radius_targets"])
			self.search_radius_effectors = int(ConfigSectionMap(self.exp_config,"SearchRadii")["search_radius_effectors"])
		except:
			self.search_radius_targets = int(ConfigSectionMap(self.exp_config,"SearchRadii")["search_radius_tc"])
			self.search_radius_effectors = int(ConfigSectionMap(self.exp_config,"SearchRadii")["search_radius_nk"])

		# self.time_dilation = int(ConfigSectionMap(self.exp_config,"BinningParameters")["time_dilation"])

		# self.intensity_measurement_radius = int(ConfigSectionMap(self.exp_config,"Thresholds")["intensity_measurement_radius"])
		# self.intensity_measurement_radius_nk = int(ConfigSectionMap(self.exp_config,"Thresholds")["intensity_measurement_radius_nk"])
		# self.model_signal_length = int(ConfigSectionMap(self.exp_config,"Thresholds")["model_signal_length"])

		# try:
		# 	self.hide_frames_for_tracking = np.array([int(s) for s in ConfigSectionMap(config,"Thresholds")["hide_frames_for_tracking"].split(",")])
		# except:
		# 	self.hide_frames_for_tracking = np.array([])

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

		try:
			if self.ProcessTargets.ConfigTracking:
				self.ProcessTargets.ConfigTracking.close()
		except:
			pass

		try:
			if self.ProcessEffectors.ConfigTracking:
				self.ProcessEffectors.ConfigTracking.close()
		except:
			pass

		try:
			if self.ProcessTargets.ConfigSignalTrain:
				self.ProcessTargets.ConfigSignalTrain.close()
		except:
			pass

		try:
			if self.ProcessEffectors.ConfigSignalTrain:
				self.ProcessEffectors.ConfigSignalTrain.close()
		except:
			pass

		try:
			if self.ProcessTargets.ConfigMeasurements:
				self.ProcessTargets.ConfigMeasurements.close()
		except:
			pass

		try:
			if self.ProcessEffectors.ConfigMeasurements:
				self.ProcessEffectors.ConfigMeasurements.close()
		except:
			pass

		try:
			if self.ProcessTargets.ConfigSignalAnnotator:
				self.ProcessTargets.ConfigSignalAnnotator.close()
		except:
			pass

		try:
			if self.ProcessEffectors.ConfigSignalAnnotator:
				self.ProcessEffectors.ConfigSignalAnnotator.close()
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
			pos_index = self.well_list.currentIndex()
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
				return False
		else:
			self.well_index = [self.well_list.currentIndex()]

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
					return False
			else:
				pos_indices = natsorted([pos.index(self.position_list.currentText())])

			well = self.wells[w_idx]

			for pos_idx in pos_indices:
				self.pos = natsorted(glob(well+f"{well[-2]}*{os.sep}"))[pos_idx]
				if not os.path.exists(self.pos + 'output'):
					os.mkdir(self.pos + 'output')
				if not os.path.exists(self.pos + os.sep.join(['output','tables'])):
					os.mkdir(self.pos + os.sep.join(['output','tables']))

		return True

	def create_config_dir(self):
		self.config_folder = self.exp_dir+'configs'+os.sep
		if not os.path.exists(self.config_folder):
			os.mkdir(self.config_folder)

	def update_position_options(self):
		
		self.pos = self.position_list.currentText()
		panels = [self.ProcessEffectors, self.ProcessTargets]
		if self.position_list.currentText()=="*":
			for p in panels:
				p.check_seg_btn.setEnabled(False)
				p.check_tracking_result_btn.setEnabled(False)
			#self.ProcessTargets.view_tab_btn.setEnabled(False)
			#self.ProcessEffectors.view_tab_btn.setEnabled(False)
			self.ProcessTargets.check_seg_btn.setEnabled(False)
			self.ProcessEffectors.check_seg_btn.setEnabled(False)
			self.ProcessTargets.check_tracking_result_btn.setEnabled(False)
			self.ProcessEffectors.check_tracking_result_btn.setEnabled(False)
			#self.ProcessTargets.signal_analysis_action.setEnabled(False)
			#self.ProcessEffectors.signal_analysis_action.setEnabled(False)
			self.ProcessTargets.check_signals_btn.setEnabled(False)
			self.ProcessEffectors.check_signals_btn.setEnabled(False)
		else:
			if not self.well_list.currentText()=="*":
				self.locate_selected_position()
				if os.path.exists(os.sep.join([self.pos,'labels_effectors', os.sep])):
					self.ProcessEffectors.check_seg_btn.setEnabled(True)
				else:
					self.ProcessEffectors.check_seg_btn.setEnabled(False)
				if os.path.exists(os.sep.join([self.pos,'labels_targets', os.sep])):
					self.ProcessTargets.check_seg_btn.setEnabled(True)
				else:
					self.ProcessTargets.check_seg_btn.setEnabled(False)
				if os.path.exists(os.sep.join([self.pos,'output','tables','napari_target_trajectories.npy'])):
					self.ProcessTargets.check_tracking_result_btn.setEnabled(True)
				else:
					self.ProcessTargets.check_tracking_result_btn.setEnabled(False)
				if os.path.exists(os.sep.join([self.pos,'output','tables','napari_effector_trajectories.npy'])):
					self.ProcessEffectors.check_tracking_result_btn.setEnabled(True)
				else:
					self.ProcessEffectors.check_tracking_result_btn.setEnabled(False)

				if os.path.exists(os.sep.join([self.pos,'output','tables','trajectories_effectors.csv'])):
					self.ProcessEffectors.check_signals_btn.setEnabled(True)
					#self.ProcessEffectors.signal_analysis_action.setEnabled(True)
					self.ProcessEffectors.view_tab_btn.setEnabled(True)

				else:
					self.ProcessEffectors.check_signals_btn.setEnabled(False)
					#self.ProcessEffectors.signal_analysis_action.setEnabled(False)
					self.ProcessEffectors.view_tab_btn.setEnabled(False)

				if os.path.exists(os.sep.join([self.pos,'output','tables','trajectories_targets.csv'])):
					self.ProcessTargets.check_signals_btn.setEnabled(True)
					#self.ProcessTargets.signal_analysis_action.setEnabled(True)
					self.ProcessTargets.view_tab_btn.setEnabled(True)

				else:
					self.ProcessTargets.check_signals_btn.setEnabled(False)
					#self.ProcessTargets.signal_analysis_action.setEnabled(False)
					self.ProcessTargets.view_tab_btn.setEnabled(False)

