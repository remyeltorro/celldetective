from PyQt5.QtWidgets import QFrame, QGridLayout, QComboBox, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QCheckBox, QMessageBox
from PyQt5.QtCore import Qt, QSize
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import gc
from celldetective.io import get_segmentation_models_list, control_segmentation_napari, get_signal_models_list, control_tracking_btrack, load_experiment_tables
from celldetective.gui import SegmentationModelLoader, ClassifierWidget, ConfigSegmentationModelTraining, ConfigTracking, SignalAnnotator, ConfigSignalModelTraining, ConfigMeasurements, ConfigSignalAnnotator, TableUI
from celldetective.gui.gui_utils import QHSeperationLine
from celldetective.segmentation import segment_at_position, segment_from_threshold_at_position
from celldetective.tracking import track_at_position
from celldetective.measure import measure_at_position
from celldetective.signals import analyze_signals_at_position
import numpy as np
from glob import glob
from natsort import natsorted
import os
import pandas as pd
from tqdm import tqdm

class ProcessPanel(QFrame):
	def __init__(self, parent, mode):

		super().__init__()		
		self.parent = parent
		self.mode = mode
		self.exp_channels = self.parent.exp_channels
		self.exp_dir = self.parent.exp_dir
		self.threshold_config_targets = None
		self.threshold_config_effectors = None
		self.wells = np.array(self.parent.wells,dtype=str)

		self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.grid = QGridLayout(self)
		self.generate_header()
	
	def generate_header(self):

		"""
		Read the mode and prepare a collapsable block to process a specific cell population.

		"""

		panel_title = QLabel(f"PROCESS {self.mode.upper()}")
		panel_title.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")

		self.grid.addWidget(panel_title, 0, 0, 1, 4, alignment=Qt.AlignCenter)

		self.select_all_btn = QPushButton()
		self.select_all_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
		self.select_all_btn.setIconSize(QSize(20, 20))
		self.all_ticked = False
		self.select_all_btn.clicked.connect(self.tick_all_actions)
		self.select_all_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.grid.addWidget(self.select_all_btn, 0, 0, 1, 4, alignment=Qt.AlignLeft)
		#self.to_disable.append(self.all_tc_actions)
		
		self.collapse_btn = QPushButton()
		self.collapse_btn.setIcon(icon(MDI6.chevron_down,color="black"))
		self.collapse_btn.setIconSize(QSize(25, 25))
		self.collapse_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.grid.addWidget(self.collapse_btn, 0, 0, 1, 4, alignment=Qt.AlignRight)

		self.populate_contents()

		self.grid.addWidget(self.ContentsFrame, 1, 0, 1, 4, alignment=Qt.AlignTop)
		self.collapse_btn.clicked.connect(lambda: self.ContentsFrame.setHidden(not self.ContentsFrame.isHidden()))
		self.collapse_btn.clicked.connect(self.collapse_advanced)
		self.ContentsFrame.hide()

	def collapse_advanced(self):
		if self.ContentsFrame.isHidden():
			self.collapse_btn.setIcon(icon(MDI6.chevron_down,color="black"))
			self.collapse_btn.setIconSize(QSize(20, 20))
			self.parent.w.adjustSize()
			self.parent.adjustSize()
		else:
			self.collapse_btn.setIcon(icon(MDI6.chevron_up,color="black"))
			self.collapse_btn.setIconSize(QSize(20, 20))
			self.parent.w.adjustSize()
			self.parent.adjustSize()

	def populate_contents(self):

		self.ContentsFrame = QFrame()
		self.grid_contents = QGridLayout(self.ContentsFrame)
		self.grid_contents.setContentsMargins(0,0,0,0)
		self.generate_segmentation_options()
		self.generate_tracking_options()
		self.generate_measure_options()
		self.generate_signal_analysis_options()

		self.grid_contents.addWidget(QHSeperationLine(), 9, 0, 1, 4)
		self.view_tab_btn = QPushButton("View table")
		self.view_tab_btn.setStyleSheet(self.parent.parent.button_style_sheet_2)
		self.view_tab_btn.clicked.connect(self.view_table_ui)
		self.view_tab_btn.setToolTip('poop twice a day for a healthy gut')
		self.view_tab_btn.setIcon(icon(MDI6.table,color="#1565c0"))
		self.view_tab_btn.setIconSize(QSize(20, 20))
		#self.view_tab_btn.setEnabled(False)
		self.grid_contents.addWidget(self.view_tab_btn, 10, 0, 1, 4)

		self.grid_contents.addWidget(QHSeperationLine(), 9, 0, 1, 4)
		self.submit_btn = QPushButton("Submit")
		self.submit_btn.setStyleSheet(self.parent.parent.button_style_sheet_2)
		self.submit_btn.clicked.connect(self.process_population)
		self.grid_contents.addWidget(self.submit_btn, 11, 0, 1, 4)

	def generate_measure_options(self):
		
		measure_layout = QHBoxLayout()

		self.measure_action = QCheckBox("MEASURE")
		self.measure_action.setStyleSheet("""
			font-size: 10px;
			padding-left: 10px;
			padding-top: 5px;
			""")
		self.measure_action.setIcon(icon(MDI6.eyedropper,color="black"))
		self.measure_action.setIconSize(QSize(20, 20))
		self.measure_action.setToolTip("Measure the intensity of the cells, \ndetect death events using the selected pre-trained model, \nformat the data for visualization, \nremove cells that are already dead and \nsave the result in a table.")
		measure_layout.addWidget(self.measure_action, 90)
		#self.to_disable.append(self.measure_action_tc)

		self.classify_btn = QPushButton()
		self.classify_btn.setIcon(icon(MDI6.scatter_plot,color="black"))
		self.classify_btn.setIconSize(QSize(20, 20))
		self.classify_btn.setToolTip("Classify data.")
		self.classify_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.classify_btn.clicked.connect(self.open_classifier_ui)
		measure_layout.addWidget(self.classify_btn, 5) #4,2,1,1, alignment=Qt.AlignRight

		self.measurements_config_btn = QPushButton()
		self.measurements_config_btn.setIcon(icon(MDI6.cog_outline,color="black"))
		self.measurements_config_btn.setIconSize(QSize(20, 20))
		self.measurements_config_btn.setToolTip("Configure measurements.")
		self.measurements_config_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.measurements_config_btn.clicked.connect(self.open_measurement_configuration_ui)
		measure_layout.addWidget(self.measurements_config_btn, 5) #4,2,1,1, alignment=Qt.AlignRight

		self.grid_contents.addLayout(measure_layout,5,0,1,4)

	def generate_signal_analysis_options(self):

		signal_layout = QVBoxLayout()
		signal_hlayout = QHBoxLayout()
		self.signal_analysis_action = QCheckBox("SIGNAL ANALYSIS")
		self.signal_analysis_action.setStyleSheet("""
			font-size: 10px;
			padding-left: 10px;
			padding-top: 5px;
			""")
		self.signal_analysis_action.setIcon(icon(MDI6.chart_bell_curve_cumulative,color="black"))
		self.signal_analysis_action.setIconSize(QSize(20, 20))
		self.signal_analysis_action.setToolTip("Analyze cell signals using deep learning or a fit procedure.")
		self.signal_analysis_action.toggled.connect(self.enable_signal_model_list)
		signal_hlayout.addWidget(self.signal_analysis_action, 90)

		self.check_signals_btn = QPushButton()
		self.check_signals_btn.setIcon(icon(MDI6.eye_check_outline,color="black"))
		self.check_signals_btn.setIconSize(QSize(20, 20))
		self.check_signals_btn.clicked.connect(self.check_signals)
		self.check_signals_btn.setStyleSheet(self.parent.parent.button_select_all)
		signal_hlayout.addWidget(self.check_signals_btn, 6)

		self.config_signal_annotator_btn = QPushButton()
		self.config_signal_annotator_btn.setIcon(icon(MDI6.cog_outline,color="black"))
		self.config_signal_annotator_btn.setIconSize(QSize(20, 20))
		self.config_signal_annotator_btn.setToolTip("Configure the animation of the annotation tool.")
		self.config_signal_annotator_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.config_signal_annotator_btn.clicked.connect(self.open_signal_annotator_configuration_ui)
		signal_hlayout.addWidget(self.config_signal_annotator_btn, 6)

		#self.to_disable.append(self.measure_action_tc)
		signal_layout.addLayout(signal_hlayout)

		signal_model_vbox = QVBoxLayout()
		signal_model_vbox.setContentsMargins(25,0,25,0)
		
		model_zoo_layout = QHBoxLayout()
		model_zoo_layout.addWidget(QLabel("Model zoo:"),90)

		self.signal_models_list = QComboBox()
		self.signal_models_list.setEnabled(False)
		self.refresh_signal_models()
		#self.to_disable.append(self.cell_models_list)

		self.train_signal_model_btn = QPushButton("TRAIN")
		self.train_signal_model_btn.setToolTip("Open a dialog box to create a new target segmentation model.")
		self.train_signal_model_btn.setIcon(icon(MDI6.redo_variant,color='black'))
		self.train_signal_model_btn.setIconSize(QSize(20, 20)) 
		self.train_signal_model_btn.setStyleSheet(self.parent.parent.button_style_sheet_3)
		model_zoo_layout.addWidget(self.train_signal_model_btn, 5)
		self.train_signal_model_btn.clicked.connect(self.open_signal_model_config_ui)
		
		signal_model_vbox.addLayout(model_zoo_layout)
		signal_model_vbox.addWidget(self.signal_models_list)

		signal_layout.addLayout(signal_model_vbox)

		self.grid_contents.addLayout(signal_layout,6,0,1,4)

	def refresh_signal_models(self):
		signal_models = get_signal_models_list()
		self.signal_models_list.clear()
		self.signal_models_list.addItems(signal_models)

	def generate_tracking_options(self):
		grid_track = QHBoxLayout()

		self.track_action = QCheckBox("TRACK")
		self.track_action.setIcon(icon(MDI6.chart_timeline_variant,color="black"))
		self.track_action.setIconSize(QSize(20, 20))
		self.track_action.setToolTip("Track the target cells using bTrack.")
		self.track_action.setStyleSheet("""
			font-size: 10px;
			padding-left: 10px;
			padding-top: 5px;
			""")
		grid_track.addWidget(self.track_action, 80)
		#self.to_disable.append(self.track_action_tc)

		# self.show_track_table_btn = QPushButton()
		# self.show_track_table_btn.setIcon(icon(MDI6.table,color="black"))
		# self.show_track_table_btn.setIconSize(QSize(20, 20))
		# self.show_track_table_btn.setToolTip("Show trajectories table.")
		# self.show_track_table_btn.setStyleSheet(self.parent.parent.button_select_all)
		# #self.show_track_table_btn.clicked.connect(self.display_trajectory_table)
		# self.show_track_table_btn.setEnabled(False)
		# grid_track.addWidget(self.show_track_table_btn, 6)  #4,3,1,1, alignment=Qt.AlignLeft

		self.check_tracking_result_btn = QPushButton()
		self.check_tracking_result_btn.setIcon(icon(MDI6.eye_check_outline,color="black"))
		self.check_tracking_result_btn.setIconSize(QSize(20, 20))
		self.check_tracking_result_btn.setToolTip("Control dynamically the trajectories.")
		self.check_tracking_result_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.check_tracking_result_btn.clicked.connect(self.open_napari_tracking)
		self.check_tracking_result_btn.setEnabled(False)
		grid_track.addWidget(self.check_tracking_result_btn, 6)  #4,3,1,1, alignment=Qt.AlignLeft

		self.track_config_btn = QPushButton()
		self.track_config_btn.setIcon(icon(MDI6.cog_outline,color="black"))
		self.track_config_btn.setIconSize(QSize(20, 20))
		self.track_config_btn.setToolTip("Configure tracking.")
		self.track_config_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.track_config_btn.clicked.connect(self.open_tracking_configuration_ui)
		grid_track.addWidget(self.track_config_btn, 6) #4,2,1,1, alignment=Qt.AlignRight

		self.grid_contents.addLayout(grid_track, 4, 0, 1,4)


	def generate_segmentation_options(self):

		grid_segment = QHBoxLayout()
		grid_segment.setContentsMargins(0,0,0,0)
		grid_segment.setSpacing(0)

		self.segment_action = QCheckBox("SEGMENT")
		self.segment_action.setStyleSheet("""
			font-size: 10px;
			padding-left: 10px;
			""")
		self.segment_action.setIcon(icon(MDI6.bacteria, color='black'))
		self.segment_action.setToolTip("Segment the cells in the movie\nusing the selected pre-trained model and save\nthe labeled output in a sub-directory.")
		self.segment_action.toggled.connect(self.enable_segmentation_model_list)
		#self.to_disable.append(self.segment_action)
		grid_segment.addWidget(self.segment_action, 90)
		
		self.check_seg_btn = QPushButton()
		self.check_seg_btn.setIcon(icon(MDI6.eye_check_outline,color="black"))
		self.check_seg_btn.setIconSize(QSize(20, 20))
		self.check_seg_btn.clicked.connect(self.check_segmentation)
		self.check_seg_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.check_seg_btn.setEnabled(False)
		#self.to_disable.append(self.control_target_seg)
		grid_segment.addWidget(self.check_seg_btn, 10)
		self.grid_contents.addLayout(grid_segment, 0,0,1,4)
		
		seg_option_vbox = QVBoxLayout()
		seg_option_vbox.setContentsMargins(25,0,25,0)
		model_zoo_layout = QHBoxLayout()
		model_zoo_layout.addWidget(QLabel("Model zoo:"),90)
		self.seg_model_list = QComboBox()
		#self.to_disable.append(self.tc_seg_model_list)
		self.seg_model_list.setGeometry(50, 50, 200, 30)
		self.init_seg_model_list()


		self.upload_model_btn = QPushButton("UPLOAD")
		self.upload_model_btn.setIcon(icon(MDI6.upload,color="black"))
		self.upload_model_btn.setIconSize(QSize(20, 20))
		self.upload_model_btn.setStyleSheet(self.parent.parent.button_style_sheet_3)
		self.upload_model_btn.setToolTip("Upload a new segmentation model (Deep learning or threshold-based).")
		model_zoo_layout.addWidget(self.upload_model_btn, 5)
		self.upload_model_btn.clicked.connect(self.upload_segmentation_model)
		# self.to_disable.append(self.upload_tc_model)

		self.train_btn = QPushButton("TRAIN")
		self.train_btn.setToolTip("Train or retrain a segmentation model on newly annotated data.")
		self.train_btn.setIcon(icon(MDI6.redo_variant,color='black'))
		self.train_btn.setIconSize(QSize(20, 20))
		self.train_btn.setStyleSheet(self.parent.parent.button_style_sheet_3)
		self.train_btn.clicked.connect(self.open_segmentation_model_config_ui)
		model_zoo_layout.addWidget(self.train_btn, 5)
		# self.train_button_tc.clicked.connect(self.train_stardist_model_tc)
		# self.to_disable.append(self.train_button_tc)

		seg_option_vbox.addLayout(model_zoo_layout)
		seg_option_vbox.addWidget(self.seg_model_list)
		self.seg_model_list.setEnabled(False)
		self.grid_contents.addLayout(seg_option_vbox, 2, 0, 1, 4)

	def check_segmentation(self):

		#self.freeze()
		#QApplication.setOverrideCursor(Qt.WaitCursor)
		test = self.parent.locate_selected_position()
		if test:
			control_segmentation_napari(self.parent.pos, prefix=self.parent.movie_prefix, population=self.mode,flush_memory=True)
			gc.collect()

	def check_signals(self):

		test = self.parent.locate_selected_position()
		if test:
			self.SignalAnnotator = SignalAnnotator(self)
			self.SignalAnnotator.show()		


	def enable_segmentation_model_list(self):
		if self.segment_action.isChecked():
			self.seg_model_list.setEnabled(True)
		else:
			self.seg_model_list.setEnabled(False)

	def enable_signal_model_list(self):
		if self.signal_analysis_action.isChecked():
			self.signal_models_list.setEnabled(True)
		else:
			self.signal_models_list.setEnabled(False)			
	
	def init_seg_model_list(self):

		self.seg_model_list.clear()
		self.seg_models = get_segmentation_models_list(mode=self.mode, return_path=False)
		self.seg_models.insert(0,'Threshold')
		thresh = 40
		models_truncated = [m[:thresh - 3]+'...' if len(m)>thresh else m for m in self.seg_models]
		self.seg_model_list.addItems(models_truncated)
		for i in range(len(self.seg_models)):
			self.seg_model_list.setItemData(i, self.seg_models[i], Qt.ToolTipRole)

		
		#if ("live_nuclei_channel" in self.exp_channels)*("dead_nuclei_channel" in self.exp_channels):
		# 	print("both channels found")
		# 	index = self.tc_seg_model_list.findText("MCF7_Hoescht_PI_w_primary_NK", Qt.MatchFixedString)
		# 	if index >= 0:
		# 		self.tc_seg_model_list.setCurrentIndex(index)
		# elif ("live_nuclei_channel" in self.exp_channels)*("dead_nuclei_channel" not in self.exp_channels):
		# 	index = self.tc_seg_model_list.findText("MCF7_Hoescht_w_primary_NK", Qt.MatchFixedString)
		# 	if index >= 0:
		# 		self.tc_seg_model_list.setCurrentIndex(index)
		# elif ("live_nuclei_channel" not in self.exp_channels)*("dead_nuclei_channel" in self.exp_channels):
		# 	index = self.tc_seg_model_list.findText("MCF7_PI_w_primary_NK", Qt.MatchFixedString)
		# 	if index >= 0:
		# 		self.tc_seg_model_list.setCurrentIndex(index)
		# elif ("live_nuclei_channel" not in self.exp_channels)*("dead_nuclei_channel" not in self.exp_channels)*("adhesion_channel" in self.exp_channels):
		# 	index = self.tc_seg_model_list.findText("RICM", Qt.MatchFixedString)
		# 	if index >= 0:
		# 		self.tc_seg_model_list.setCurrentIndex(index)

	def tick_all_actions(self):
		self.switch_all_ticks_option()
		if self.all_ticked:
			self.select_all_btn.setIcon(icon(MDI6.checkbox_outline,color="black"))
			self.select_all_btn.setIconSize(QSize(20, 20))
			self.segment_action.setChecked(True)
		else:
			self.select_all_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
			self.select_all_btn.setIconSize(QSize(20, 20))
			self.segment_action.setChecked(False)

	def switch_all_ticks_option(self):
		if self.all_ticked == True:
			self.all_ticked = False
		else:
			self.all_ticked = True

	def upload_segmentation_model(self):

		self.SegModelLoader = SegmentationModelLoader(self)
		self.SegModelLoader.show()

	def open_tracking_configuration_ui(self):

		self.ConfigTracking = ConfigTracking(self)
		self.ConfigTracking.show()

	def open_signal_model_config_ui(self):

		self.ConfigSignalTrain = ConfigSignalModelTraining(self)
		self.ConfigSignalTrain.show()

	def open_segmentation_model_config_ui(self):

		self.ConfigSegmentationTrain = ConfigSegmentationModelTraining(self)
		self.ConfigSegmentationTrain.show()

	def open_measurement_configuration_ui(self):

		self.ConfigMeasurements = ConfigMeasurements(self)
		self.ConfigMeasurements.show()

	def open_classifier_ui(self):
		self.ClassifierWidget = ClassifierWidget(self)
		self.ClassifierWidget.show()

	def open_signal_annotator_configuration_ui(self):
		self.ConfigSignalAnnotator = ConfigSignalAnnotator(self)
		self.ConfigSignalAnnotator.show()

	def process_population(self):
		
		if self.parent.well_list.currentText()=="*":
			self.well_index = np.linspace(0,len(self.wells)-1,len(self.wells),dtype=int)
		else:
			self.well_index = [self.parent.well_list.currentIndex()]
			print(f"Processing well {self.parent.well_list.currentText()}...")

		# self.freeze()
		# QApplication.setOverrideCursor(Qt.WaitCursor)

		if self.mode=="targets":
			self.threshold_config = self.threshold_config_targets
		elif self.mode=="effectors":
			self.threshold_config = self.threshold_config_effectors
		
		loop_iter=0

		if self.parent.position_list.currentText()=="*":
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Question)
			msgBox.setText("If you continue, all positions will be processed.\nDo you want to proceed?")
			msgBox.setWindowTitle("Info")
			msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.No:
				return None

		for w_idx in self.well_index:

			pos = self.parent.positions[w_idx]
			if self.parent.position_list.currentText()=="*":
				pos_indices = np.linspace(0,len(pos)-1,len(pos),dtype=int)
				print("Processing all positions...")
			else:
				pos_indices = natsorted([pos.index(self.parent.position_list.currentText())])
				print(f"Processing position {self.parent.position_list.currentText()}...")

			well = self.parent.wells[w_idx]

			for pos_idx in pos_indices:
				
				self.pos = natsorted(glob(well+f"{os.path.split(well)[-1].replace('W','').replace(os.sep,'')}*/"))[pos_idx]
				print(f"Position {self.pos}...\nLoading stack movie...")
				model_name = self.seg_models[self.seg_model_list.currentIndex()]

				if not os.path.exists(self.pos + 'output/'):
					os.mkdir(self.pos + 'output/')
				if not os.path.exists(self.pos + 'output/tables/'):
					os.mkdir(self.pos + 'output/tables/')

				if self.segment_action.isChecked():
					
					if len(glob(os.sep.join([self.pos, f'labels_{self.mode}','*.tif'])))>0 and self.parent.position_list.currentText()!="*":
						msgBox = QMessageBox()
						msgBox.setIcon(QMessageBox.Question)
						msgBox.setText("Labels have already been produced for this position. Do you want to segment again?")
						msgBox.setWindowTitle("Info")
						msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
						returnValue = msgBox.exec()
						if returnValue == QMessageBox.No:
							return None

					if (self.seg_model_list.currentText()=="Threshold"):
						if self.threshold_config is None:
							msgBox = QMessageBox()
							msgBox.setIcon(QMessageBox.Warning)
							msgBox.setText("Please set a threshold configuration from the upload menu first. Abort.")
							msgBox.setWindowTitle("Warning")
							msgBox.setStandardButtons(QMessageBox.Ok)
							returnValue = msgBox.exec()
							if returnValue == QMessageBox.Ok:
								return None					
						else:
							print(f"Segmentation from threshold config: {self.threshold_config}")
							segment_from_threshold_at_position(self.pos, self.mode, self.threshold_config)
					else:
						segment_at_position(self.pos, self.mode, model_name, stack_prefix=self.parent.movie_prefix, use_gpu=True)

				if self.track_action.isChecked():
					if os.path.exists(os.sep.join([self.pos, 'output', 'tables', f'trajectories_{self.mode}.csv'])) and self.parent.position_list.currentText()!="*":
						msgBox = QMessageBox()
						msgBox.setIcon(QMessageBox.Question)
						msgBox.setText("A trajectory set already exists. Previously annotated data for\nthis position will be lost. Do you want to proceed?")
						msgBox.setWindowTitle("Info")
						msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
						returnValue = msgBox.exec()
						if returnValue == QMessageBox.No:
							return None
					track_at_position(self.pos, self.mode)

				if self.measure_action.isChecked():
					measure_at_position(self.pos, self.mode)

				table = os.sep.join([self.pos, 'output', 'tables', f'trajectories_{self.mode}.csv'])
				if self.signal_analysis_action.isChecked() and os.path.exists(table):
					print('table exists')
					table = pd.read_csv(table)
					cols = list(table.columns)
					print(table, cols)
					if 'class_color' in cols:
						print(cols, 'class_color in cols')
						colors = list(table['class_color'].to_numpy())
						if 'tab:orange' in colors or 'tab:cyan' in colors:
							if self.parent.position_list.currentText()!="*":
								msgBox = QMessageBox()
								msgBox.setIcon(QMessageBox.Question)
								msgBox.setText("The signals of the cells in the position appear to have been annotated... Do you want to proceed?")
								msgBox.setWindowTitle("Info")
								msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
								returnValue = msgBox.exec()
								if returnValue == QMessageBox.No:
									return None
					analyze_signals_at_position(self.pos, self.signal_models_list.currentText(), self.mode)


			# self.stack = None
		self.parent.update_position_options()
		if self.segment_action.isChecked():
			self.segment_action.setChecked(False)

		# QApplication.restoreOverrideCursor()
		# self.unfreeze()

	def open_napari_tracking(self):
		control_tracking_btrack(self.parent.pos, prefix=self.parent.movie_prefix, population=self.mode)

	def view_table_ui(self):
		
		print('Load table...')
		self.load_available_tables()

		if self.df is not None:
			self.tab_ui = TableUI(self.df, f"Well {self.parent.well_list.currentText()}; Position {self.parent.position_list.currentText()}")
			self.tab_ui.show()
		else:
			print('Table could not be loaded...')
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("No table could be loaded...")
			msgBox.setWindowTitle("Info")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None

	# def interpret_pos_location(self):
		
	# 	"""
	# 	Read the well/position selection from the control panel to decide which data to load
	# 	Set position_indices to None if all positions must be taken

	# 	"""
		
	# 	if self.well_option==len(self.wells):
	# 		self.well_indices = np.arange(len(self.wells))
	# 	else:
	# 		self.well_indices = np.array([self.well_option],dtype=int)

	# 	if self.position_option==0:
	# 		self.position_indices = None
	# 	else:
	# 		self.position_indices = np.array([self.position_option],dtype=int)

	def load_available_tables(self):

		"""
		Load the tables of the selected wells/positions from the control Panel for the population of interest

		"""

		self.well_option = self.parent.well_list.currentIndex()
		if self.well_option==len(self.wells):
			wo = '*'
		else:
			wo = self.well_option
		self.position_option = self.parent.position_list.currentIndex()
		if self.position_option==0:
			po = '*'
		else:
			po = self.position_option

		self.df, self.df_pos_info = load_experiment_tables(self.exp_dir, well_option=wo, position_option=po, population=self.mode, return_pos_info=True)
		if self.df is None:
			print('no table could be found...')

class NeighPanel(QFrame):
	def __init__(self, parent):

		super().__init__()		
		self.parent = parent
		self.exp_channels = self.parent.exp_channels
		self.exp_dir = self.parent.exp_dir

		self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.grid = QGridLayout(self)
		self.generate_header()
	
	def generate_header(self):

		"""
		Read the mode and prepare a collapsable block to process a specific cell population.

		"""

		panel_title = QLabel(f"NEIGHBORHOOD")
		panel_title.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")

		self.grid.addWidget(panel_title, 0, 0, 1, 4, alignment=Qt.AlignCenter)

		self.select_all_btn = QPushButton()
		self.select_all_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
		self.select_all_btn.setIconSize(QSize(20, 20))
		self.all_ticked = False
		#self.select_all_btn.clicked.connect(self.tick_all_actions)
		self.select_all_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.grid.addWidget(self.select_all_btn, 0, 0, 1, 4, alignment=Qt.AlignLeft)
		#self.to_disable.append(self.all_tc_actions)
		
		self.collapse_btn = QPushButton()
		self.collapse_btn.setIcon(icon(MDI6.chevron_down,color="black"))
		self.collapse_btn.setIconSize(QSize(25, 25))
		self.collapse_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.grid.addWidget(self.collapse_btn, 0, 0, 1, 4, alignment=Qt.AlignRight)

		self.populate_contents()
		
		self.grid.addWidget(self.ContentsFrame, 1, 0, 1, 4, alignment=Qt.AlignTop)
		self.collapse_btn.clicked.connect(lambda: self.ContentsFrame.setHidden(not self.ContentsFrame.isHidden()))
		self.collapse_btn.clicked.connect(self.collapse_advanced)
		self.ContentsFrame.hide()

	def collapse_advanced(self):
		if self.ContentsFrame.isHidden():
			self.collapse_btn.setIcon(icon(MDI6.chevron_down,color="black"))
			self.collapse_btn.setIconSize(QSize(20, 20))
			self.parent.w.adjustSize()
			self.parent.adjustSize()
		else:
			self.collapse_btn.setIcon(icon(MDI6.chevron_up,color="black"))
			self.collapse_btn.setIconSize(QSize(20, 20))
			self.parent.w.adjustSize()
			self.parent.adjustSize()

	def populate_contents(self):

		self.ContentsFrame = QFrame()
		self.grid_contents = QGridLayout(self.ContentsFrame)
		self.grid_contents.setContentsMargins(0,0,0,0)


		# DISTANCE NEIGHBORHOOD
		dist_neigh_hbox = QHBoxLayout()
		dist_neigh_hbox.setContentsMargins(0,0,0,0)
		dist_neigh_hbox.setSpacing(0)

		self.dist_neigh_action = QCheckBox("DISTANCE CUT")
		self.dist_neigh_action.setStyleSheet("""
			font-size: 10px;
			padding-left: 10px;
			""")
		self.dist_neigh_action.setIcon(icon(MDI6.circle_expand, color='black'))
		self.dist_neigh_action.setToolTip("Match cells for which the center of mass is within a threshold distance of each other.")
		#self.segment_action.toggled.connect(self.enable_segmentation_model_list)
		#self.to_disable.append(self.segment_action)
		dist_neigh_hbox.addWidget(self.dist_neigh_action, 95)
		
		self.config_distance_neigh_btn = QPushButton()
		self.config_distance_neigh_btn.setIcon(icon(MDI6.cog_outline,color="black"))
		self.config_distance_neigh_btn.setIconSize(QSize(20, 20))
		self.config_distance_neigh_btn.setToolTip("Configure distance cut neighbourhood computation.")
		self.config_distance_neigh_btn.setStyleSheet(self.parent.parent.button_select_all)
		#self.config_distance_neigh_btn.clicked.connect(self.open_signal_annotator_configuration_ui)
		dist_neigh_hbox.addWidget(self.config_distance_neigh_btn,5)

		self.grid_contents.addLayout(dist_neigh_hbox, 0,0,1,4)

		# MASK INTERSECTION NEIGHBORHOOD

		mask_neigh_hbox = QHBoxLayout()
		mask_neigh_hbox.setContentsMargins(0,0,0,0)
		mask_neigh_hbox.setSpacing(0)

		self.mask_neigh_action = QCheckBox("MASK INTERSECTION")
		self.mask_neigh_action.setStyleSheet("""
			font-size: 10px;
			padding-left: 10px;
			""")
		self.mask_neigh_action.setIcon(icon(MDI6.domino_mask, color='black'))
		self.mask_neigh_action.setToolTip("Match cells that are co-localizing.")
		#self.segment_action.toggled.connect(self.enable_segmentation_model_list)
		#self.to_disable.append(self.segment_action)
		mask_neigh_hbox.addWidget(self.mask_neigh_action, 95)
		
		self.config_mask_neigh_btn = QPushButton()
		self.config_mask_neigh_btn.setIcon(icon(MDI6.cog_outline,color="black"))
		self.config_mask_neigh_btn.setIconSize(QSize(20, 20))
		self.config_mask_neigh_btn.setToolTip("Configure mask intersection computation.")
		self.config_mask_neigh_btn.setStyleSheet(self.parent.parent.button_select_all)
		#self.config_distance_neigh_btn.clicked.connect(self.open_signal_annotator_configuration_ui)
		mask_neigh_hbox.addWidget(self.config_mask_neigh_btn,5)

		self.grid_contents.addLayout(mask_neigh_hbox, 1,0,1,4)

		self.grid_contents.addWidget(QHSeperationLine(), 2, 0, 1, 4)
		self.submit_btn = QPushButton("Submit")
		self.submit_btn.setStyleSheet(self.parent.parent.button_style_sheet_2)
		#self.submit_btn.clicked.connect(self.process_population)
		self.grid_contents.addWidget(self.submit_btn, 3, 0, 1, 4)