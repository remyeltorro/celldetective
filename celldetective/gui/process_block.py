from PyQt5.QtWidgets import QFrame, QGridLayout, QRadioButton, QButtonGroup, QGroupBox, QComboBox,QTabWidget,QSizePolicy,QListWidget, QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QCheckBox, \
	QMessageBox, QWidget, QLineEdit, QScrollArea, QSpacerItem, QLayout, QSizePolicy
from PyQt5.QtCore import Qt, QSize
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import gc
from PyQt5.QtGui import QIcon, QDoubleValidator, QIntValidator

from celldetective.gui.signal_annotator import MeasureAnnotator
from celldetective.io import get_segmentation_models_list, control_segmentation_napari, get_signal_models_list, control_tracking_btrack, load_experiment_tables
from celldetective.io import locate_segmentation_model, auto_load_number_of_frames, load_frames
from celldetective.gui import SegmentationModelLoader, ClassifierWidget, ConfigNeighborhoods, ConfigSegmentationModelTraining, ConfigTracking, SignalAnnotator, ConfigSignalModelTraining, ConfigMeasurements, ConfigSignalAnnotator, TableUI
from celldetective.gui.gui_utils import QHSeperationLine
from celldetective.segmentation import segment_at_position, segment_from_threshold_at_position
from celldetective.tracking import track_at_position
from celldetective.measure import measure_at_position
from celldetective.signals import analyze_signals_at_position
from celldetective.utils import extract_experiment_channels
import numpy as np
from glob import glob
from natsort import natsorted
from superqt import QLabeledDoubleSlider, QLabeledSlider, QLabeledRangeSlider, QLabeledSlider, QLabeledDoubleRangeSlider
import os
import pandas as pd
from tqdm import tqdm
from celldetective.gui.gui_utils import center_window
from tifffile import imwrite
import json
import psutil
from celldetective.neighborhood import compute_neighborhood_at_position
from celldetective.gui.gui_utils import FigureCanvas
import matplotlib.pyplot as plt
from celldetective.filters import std_filter, median_filter, gauss_filter
from stardist import fill_label_holes
from celldetective.preprocessing import correct_background, estimate_background_per_condition
from celldetective.utils import _estimate_scale_factor, _extract_channel_indices_from_config, _extract_channel_indices, ConfigSectionMap, _extract_nbr_channels_from_config, _get_img_num_per_channel, normalize_per_channel
from celldetective.gui.gui_utils import StackVisualizer, ThresholdedStackVisualizer, ThresholdLineEdit, QuickSliderLayout, BackgroundFitCorrectionLayout



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
		self.cellpose_calibrated = False
		self.stardist_calibrated = False



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
			self.parent.scroll.setMinimumHeight(int(500))
			#self.parent.w.adjustSize()
			self.parent.adjustSize()
			#self.parent.scroll.adjustSize()
		else:
			self.collapse_btn.setIcon(icon(MDI6.chevron_up,color="black"))
			self.collapse_btn.setIconSize(QSize(20, 20))
			#self.parent.w.adjustSize()
			#self.parent.adjustSize()
			self.parent.scroll.setMinimumHeight(min(int(880), int(0.8*self.parent.screen_height)))
			self.parent.scroll.setMinimumWidth(410)

			#self.parent.scroll.adjustSize()

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
		self.view_tab_btn.setToolTip('View table')
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
		self.measure_action.setToolTip("Measure.")
		measure_layout.addWidget(self.measure_action, 90)
		#self.to_disable.append(self.measure_action_tc)

		self.classify_btn = QPushButton()
		self.classify_btn.setIcon(icon(MDI6.scatter_plot, color="black"))
		self.classify_btn.setIconSize(QSize(20, 20))
		self.classify_btn.setToolTip("Classify data.")
		self.classify_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.classify_btn.clicked.connect(self.open_classifier_ui)
		measure_layout.addWidget(self.classify_btn, 5) #4,2,1,1, alignment=Qt.AlignRight

		self.check_measurements_btn=QPushButton()
		self.check_measurements_btn.setIcon(icon(MDI6.eye_check_outline,color="black"))
		self.check_measurements_btn.setIconSize(QSize(20, 20))
		self.check_measurements_btn.setToolTip("Explore measurements in-situ.")
		self.check_measurements_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.check_measurements_btn.clicked.connect(self.check_measurements)
		measure_layout.addWidget(self.check_measurements_btn, 5)


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
		self.signal_analysis_action.setToolTip("Detect events in single-cell signals.")
		self.signal_analysis_action.toggled.connect(self.enable_signal_model_list)
		signal_hlayout.addWidget(self.signal_analysis_action, 90)

		self.check_signals_btn = QPushButton()
		self.check_signals_btn.setIcon(icon(MDI6.eye_check_outline,color="black"))
		self.check_signals_btn.setIconSize(QSize(20, 20))
		self.check_signals_btn.clicked.connect(self.check_signals)
		self.check_signals_btn.setToolTip("Explore signals in-situ.")
		self.check_signals_btn.setStyleSheet(self.parent.parent.button_select_all)
		signal_hlayout.addWidget(self.check_signals_btn, 6)

		self.config_signal_annotator_btn = QPushButton()
		self.config_signal_annotator_btn.setIcon(icon(MDI6.cog_outline,color="black"))
		self.config_signal_annotator_btn.setIconSize(QSize(20, 20))
		self.config_signal_annotator_btn.setToolTip("Configure the dynamic visualizer.")
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
		self.check_tracking_result_btn.setToolTip("View raw bTrack output in napari.")
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
		self.segment_action.setToolTip(f"Segment the {self.mode} cells on the images.")
		self.segment_action.toggled.connect(self.enable_segmentation_model_list)
		#self.to_disable.append(self.segment_action)
		grid_segment.addWidget(self.segment_action, 90)
		
		self.check_seg_btn = QPushButton()
		self.check_seg_btn.setIcon(icon(MDI6.eye_check_outline,color="black"))
		self.check_seg_btn.setIconSize(QSize(20, 20))
		self.check_seg_btn.clicked.connect(self.check_segmentation)
		self.check_seg_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.check_seg_btn.setToolTip("View segmentation output in napari.")
		#self.check_seg_btn.setEnabled(False)
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
		self.upload_model_btn.setToolTip("Upload a new segmentation model\n(Deep learning or threshold-based).")
		model_zoo_layout.addWidget(self.upload_model_btn, 5)
		self.upload_model_btn.clicked.connect(self.upload_segmentation_model)
		# self.to_disable.append(self.upload_tc_model)

		self.train_btn = QPushButton("TRAIN")
		self.train_btn.setToolTip("Train or retrain a segmentation model\non newly annotated data.")
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

		if not os.path.exists(os.sep.join([self.parent.pos,f'labels_{self.mode}', os.sep])):
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Question)
			msgBox.setText("No labels can be found for this position. Do you want to annotate from scratch?")
			msgBox.setWindowTitle("Info")
			msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.No:
				return None
			else:
				os.mkdir(os.sep.join([self.parent.pos,f'labels_{self.mode}']))
				lbl = np.zeros((self.parent.shape_x, self.parent.shape_y), dtype=int)
				for i in range(self.parent.len_movie):
					imwrite(os.sep.join([self.parent.pos,f'labels_{self.mode}', str(i).zfill(4)+'.tif']), lbl)

		#self.freeze()
		#QApplication.setOverrideCursor(Qt.WaitCursor)
		test = self.parent.locate_selected_position()
		if test:
			print('Memory use: ', dict(psutil.virtual_memory()._asdict()))
			try:
				control_segmentation_napari(self.parent.pos, prefix=self.parent.movie_prefix, population=self.mode,flush_memory=True)
			except Exception as e:
				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Warning)
				msgBox.setText(str(e))
				msgBox.setWindowTitle("Warning")
				msgBox.setStandardButtons(QMessageBox.Ok)
				returnValue = msgBox.exec()
			gc.collect()

	def check_signals(self):

		test = self.parent.locate_selected_position()
		if test:
			self.SignalAnnotator = SignalAnnotator(self)
			self.SignalAnnotator.show()		

	def check_measurements(self):

		test = self.parent.locate_selected_position()
		if test:
			self.MeasureAnnotator = MeasureAnnotator(self)
			self.MeasureAnnotator.show()

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
		thresh = 40
		self.models_truncated = [m[:thresh - 3]+'...' if len(m)>thresh else m for m in self.seg_models]
		#self.seg_model_list.addItems(models_truncated)

		self.seg_models_generic = get_segmentation_models_list(mode="generic", return_path=False)
		self.seg_models.append('Threshold')
		self.seg_models.extend(self.seg_models_generic)

		#self.seg_models_generic.insert(0,'Threshold')
		self.seg_model_list.addItems(self.seg_models)		

		for i in range(len(self.seg_models)):
			self.seg_model_list.setItemData(i, self.seg_models[i], Qt.ToolTipRole)

		self.seg_model_list.insertSeparator(len(self.models_truncated))

		
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

		self.load_available_tables()
		if self.df is None:

			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("No table was found...")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None	
			else:
				return None
		else:	
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
		
		if self.seg_model_list.currentIndex() > len(self.models_truncated):
			self.model_name = self.seg_models[self.seg_model_list.currentIndex()-1]
		else:
			self.model_name = self.seg_models[self.seg_model_list.currentIndex()]
		print(self.model_name, self.seg_model_list.currentIndex())

		if self.model_name.startswith('CP') and self.model_name in self.seg_models_generic and not self.cellpose_calibrated:

			self.diamWidget = QWidget()
			self.diamWidget.setWindowTitle('Estimate diameter')
			
			layout = QVBoxLayout()
			self.diamWidget.setLayout(layout)
			self.diameter_le = QLineEdit('40')

			self.cellpose_channel_cb = [QComboBox() for i in range(2)]
			self.cellpose_channel_template = ['brightfield_channel', 'live_nuclei_channel']
			if self.model_name=="CP_nuclei":
				self.cellpose_channel_template = ['live_nuclei_channel', 'None']


			for k in range(2):
				hbox_channel = QHBoxLayout()
				hbox_channel.addWidget(QLabel(f'channel {k+1}: '))
				hbox_channel.addWidget(self.cellpose_channel_cb[k])
				if k==1:
					self.cellpose_channel_cb[k].addItems(list(self.exp_channels)+['None'])
				else:
					self.cellpose_channel_cb[k].addItems(list(self.exp_channels))					
				idx = self.cellpose_channel_cb[k].findText(self.cellpose_channel_template[k])
				self.cellpose_channel_cb[k].setCurrentIndex(idx)
				layout.addLayout(hbox_channel)

			hbox = QHBoxLayout()
			hbox.addWidget(QLabel('diameter [px]: '), 33)
			hbox.addWidget(self.diameter_le, 66)
			layout.addLayout(hbox)

			self.flow_slider = QLabeledDoubleSlider()
			self.flow_slider.setOrientation(1)
			self.flow_slider.setRange(-6,6)
			self.flow_slider.setValue(0.4)

			hbox = QHBoxLayout()
			hbox.addWidget(QLabel('flow threshold: '), 33)
			hbox.addWidget(self.flow_slider, 66)
			layout.addLayout(hbox)	

			self.cellprob_slider = QLabeledDoubleSlider()
			self.cellprob_slider.setOrientation(1)
			self.cellprob_slider.setRange(-6,6)
			self.cellprob_slider.setValue(0.)

			hbox = QHBoxLayout()
			hbox.addWidget(QLabel('cellprob threshold: '), 33)
			hbox.addWidget(self.cellprob_slider, 66)
			layout.addLayout(hbox)			

			self.set_cellpose_scale_btn = QPushButton('set')
			self.set_cellpose_scale_btn.clicked.connect(self.set_cellpose_scale)
			layout.addWidget(self.set_cellpose_scale_btn)

			self.diamWidget.show()
			center_window(self.diamWidget)
			return None


		if self.model_name.startswith('SD') and self.model_name in self.seg_models_generic and not self.stardist_calibrated:

			self.diamWidget = QWidget()
			self.diamWidget.setWindowTitle('Channels')
			
			layout = QVBoxLayout()
			self.diamWidget.setLayout(layout)

			self.stardist_channel_cb = [QComboBox() for i in range(1)]
			self.stardist_channel_template = ['live_nuclei_channel']
			max_i = 1
			if self.model_name=="SD_versatile_he":
				self.stardist_channel_template = ["H&E_1","H&E_2","H&E_3"]
				self.stardist_channel_cb = [QComboBox() for i in range(3)]
				max_i = 3

			for k in range(max_i):
				hbox_channel = QHBoxLayout()
				hbox_channel.addWidget(QLabel(f'channel {k+1}: '))
				hbox_channel.addWidget(self.stardist_channel_cb[k])
				if k==1:
					self.stardist_channel_cb[k].addItems(list(self.exp_channels)+['None'])
				else:
					self.stardist_channel_cb[k].addItems(list(self.exp_channels))					
				idx = self.stardist_channel_cb[k].findText(self.stardist_channel_template[k])
				self.stardist_channel_cb[k].setCurrentIndex(idx)
				layout.addLayout(hbox_channel)

			self.set_stardist_scale_btn = QPushButton('set')
			self.set_stardist_scale_btn.clicked.connect(self.set_stardist_scale)
			layout.addWidget(self.set_stardist_scale_btn)

			self.diamWidget.show()
			center_window(self.diamWidget)
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
							segment_from_threshold_at_position(self.pos, self.mode, self.threshold_config, threads=self.parent.parent.n_threads)
					else:
						segment_at_position(self.pos, self.mode, self.model_name, stack_prefix=self.parent.movie_prefix, use_gpu=self.parent.parent.use_gpu, threads=self.parent.parent.n_threads)

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
					track_at_position(self.pos, self.mode, threads=self.parent.parent.n_threads)

				if self.measure_action.isChecked():
					measure_at_position(self.pos, self.mode, threads=self.parent.parent.n_threads)

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
		control_tracking_btrack(self.parent.pos, prefix=self.parent.movie_prefix, population=self.mode, threads=self.parent.parent.n_threads)

	def view_table_ui(self):
		
		print('Load table...')
		self.load_available_tables()

		if self.df is not None:
			plot_mode = 'plot_track_signals'
			if 'TRACK_ID' not in list(self.df.columns):
				plot_mode = 'static'
			self.tab_ui = TableUI(self.df, f"Well {self.parent.well_list.currentText()}; Position {self.parent.position_list.currentText()}", population=self.mode, plot_mode=plot_mode)
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
			po = self.position_option - 1

		self.df, self.df_pos_info = load_experiment_tables(self.exp_dir, well_option=wo, position_option=po, population=self.mode, return_pos_info=True)
		if self.df is None:
			print('No table could be found...')

	def set_cellpose_scale(self):

		scale = self.parent.PxToUm * float(self.diameter_le.text()) / 30.0
		if self.model_name=="CP_nuclei":
			scale = self.parent.PxToUm * float(self.diameter_le.text()) / 17.0
		flow_thresh = self.flow_slider.value()
		cellprob_thresh = self.cellprob_slider.value()
		model_complete_path = locate_segmentation_model(self.model_name)
		input_config_path = model_complete_path+"config_input.json"
		new_channels = [self.cellpose_channel_cb[i].currentText() for i in range(2)]
		print(new_channels)
		with open(input_config_path) as config_file:
			input_config = json.load(config_file)
		
		input_config['spatial_calibration'] = scale
		input_config['channels'] = new_channels
		input_config['flow_threshold'] = flow_thresh
		input_config['cellprob_threshold'] = cellprob_thresh
		with open(input_config_path, 'w') as f:
			json.dump(input_config, f, indent=4)

		self.cellpose_calibrated = True
		print('model scale automatically computed: ', scale)
		self.diamWidget.close()
		self.process_population()

	def set_stardist_scale(self):

		# scale = self.parent.PxToUm * float(self.diameter_le.text()) / 30.0
		# if self.model_name=="CP_nuclei":
		# 	scale = self.parent.PxToUm * float(self.diameter_le.text()) / 17.0
		# flow_thresh = self.flow_slider.value()
		# cellprob_thresh = self.cellprob_slider.value()
		model_complete_path = locate_segmentation_model(self.model_name)
		input_config_path = model_complete_path+"config_input.json"
		new_channels = [self.stardist_channel_cb[i].currentText() for i in range(len(self.stardist_channel_cb))]
		with open(input_config_path) as config_file:
			input_config = json.load(config_file)
		
		# input_config['spatial_calibration'] = scale
		input_config['channels'] = new_channels
		# input_config['flow_threshold'] = flow_thresh
		# input_config['cellprob_threshold'] = cellprob_thresh
		with open(input_config_path, 'w') as f:
			json.dump(input_config, f, indent=4)

		self.stardist_calibrated = True
		self.diamWidget.close()
		self.process_population()


class NeighPanel(QFrame):
	def __init__(self, parent):

		super().__init__()		
		self.parent = parent
		self.exp_channels = self.parent.exp_channels
		self.exp_dir = self.parent.exp_dir
		self.wells = np.array(self.parent.wells,dtype=str)

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
		self.config_distance_neigh_btn.clicked.connect(self.open_config_neighborhood)
		dist_neigh_hbox.addWidget(self.config_distance_neigh_btn,5)

		self.grid_contents.addLayout(dist_neigh_hbox, 0,0,1,4)

		# MASK INTERSECTION NEIGHBORHOOD
		# mask_neigh_hbox = QHBoxLayout()
		# mask_neigh_hbox.setContentsMargins(0,0,0,0)
		# mask_neigh_hbox.setSpacing(0)

		# self.mask_neigh_action = QCheckBox("MASK INTERSECTION")
		# self.mask_neigh_action.setStyleSheet("""
		# 	font-size: 10px;
		# 	padding-left: 10px;
		# 	""")
		# self.mask_neigh_action.setIcon(icon(MDI6.domino_mask, color='black'))
		# self.mask_neigh_action.setToolTip("Match cells that are co-localizing.")
		# #self.segment_action.toggled.connect(self.enable_segmentation_model_list)
		# #self.to_disable.append(self.segment_action)
		# mask_neigh_hbox.addWidget(self.mask_neigh_action, 95)
		
		# self.config_mask_neigh_btn = QPushButton()
		# self.config_mask_neigh_btn.setIcon(icon(MDI6.cog_outline,color="black"))
		# self.config_mask_neigh_btn.setIconSize(QSize(20, 20))
		# self.config_mask_neigh_btn.setToolTip("Configure mask intersection computation.")
		# self.config_mask_neigh_btn.setStyleSheet(self.parent.parent.button_select_all)
		# #self.config_distance_neigh_btn.clicked.connect(self.open_signal_annotator_configuration_ui)
		# mask_neigh_hbox.addWidget(self.config_mask_neigh_btn,5)

		# self.grid_contents.addLayout(mask_neigh_hbox, 1,0,1,4)

		self.grid_contents.addWidget(QHSeperationLine(), 2, 0, 1, 4)
		self.submit_btn = QPushButton("Submit")
		self.submit_btn.setStyleSheet(self.parent.parent.button_style_sheet_2)
		self.submit_btn.clicked.connect(self.process_neighborhood)
		self.grid_contents.addWidget(self.submit_btn, 3, 0, 1, 4)

	def open_config_neighborhood(self):

		self.ConfigNeigh = ConfigNeighborhoods(self)
		self.ConfigNeigh.show()

	def process_neighborhood(self):
		
		if self.parent.well_list.currentText()=="*":
			self.well_index = np.linspace(0,len(self.wells)-1,len(self.wells),dtype=int)
		else:
			self.well_index = [self.parent.well_list.currentIndex()]
			print(f"Processing well {self.parent.well_list.currentText()}...")

		# self.freeze()
		# QApplication.setOverrideCursor(Qt.WaitCursor)

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
				
				self.pos = natsorted(glob(well+f"{os.path.split(well)[-1].replace('W','').replace(os.sep,'')}*{os.sep}"))[pos_idx]
				print(f"Position {self.pos}...\nLoading stack movie...")

				if not os.path.exists(self.pos + 'output' + os.sep):
					os.mkdir(self.pos + 'output' + os.sep)
				if not os.path.exists(self.pos + os.sep.join(['output','tables'])+os.sep):
					os.mkdir(self.pos + os.sep.join(['output','tables'])+os.sep)

				if self.dist_neigh_action.isChecked():
				
					config = self.exp_dir + os.sep.join(["configs","neighborhood_instructions.json"])
					
					if not os.path.exists(config):
						print('config could not be found', config)
						msgBox = QMessageBox()
						msgBox.setIcon(QMessageBox.Warning)
						msgBox.setText("Please define a neighborhood first.")
						msgBox.setWindowTitle("Info")
						msgBox.setStandardButtons(QMessageBox.Ok)
						returnValue = msgBox.exec()
						return None

					with open(config, 'r') as f:
						config = json.load(f)

					compute_neighborhood_at_position(self.pos, 
													config['distance'], 
													population=config['population'], 
													theta_dist=None, 
													img_shape=(self.parent.shape_x,self.parent.shape_y), 
													return_tables=False,
													clear_neigh=config['clear_neigh'],
													event_time_col=config['event_time_col'],
													neighborhood_kwargs=config['neighborhood_kwargs'],
													)
		print('Done.')


class PreprocessingPanel(QFrame):
	
	def __init__(self, parent):

		super().__init__()		
		self.parent = parent
		self.exp_channels = self.parent.exp_channels
		self.exp_dir = self.parent.exp_dir
		self.wells = np.array(self.parent.wells,dtype=str)
		exp_config = self.exp_dir + "config.ini"
		self.channel_names, self.channels = extract_experiment_channels(exp_config)
		self.channel_names = np.array(self.channel_names)
		self.background_correction = []
		self.onlyFloat = QDoubleValidator()
		self.onlyInt = QIntValidator()
		
		self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.grid = QGridLayout(self)

		self.generate_header()
	
	def generate_header(self):

		"""
		Read the mode and prepare a collapsable block to process a specific cell population.

		"""

		panel_title = QLabel(f"PREPROCESSING")
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
			self.parent.scroll.setMinimumHeight(int(500))
			#self.parent.w.adjustSize()
			self.parent.adjustSize()
		else:
			self.collapse_btn.setIcon(icon(MDI6.chevron_up,color="black"))
			self.collapse_btn.setIconSize(QSize(20, 20))
			#self.parent.w.adjustSize()
			#self.parent.adjustSize()
			self.parent.scroll.setMinimumHeight(min(int(880), int(0.8*self.parent.screen_height)))
			self.parent.scroll.setMinimumWidth(410)

	def populate_contents(self):

		self.ContentsFrame = QFrame()
		self.grid_contents = QGridLayout(self.ContentsFrame)

		layout = QVBoxLayout()
		self.normalisation_lbl = QLabel("BACKGROUND CORRECTION")
		self.normalisation_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		layout.addWidget(self.normalisation_lbl, alignment=Qt.AlignCenter)
		self.tabs = QTabWidget()
		self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		
		self.tab2 = QWidget()
		self.tab_condition = QWidget()

		self.normalisation_list = QListWidget()
		self.tabs.addTab(self.tab2, 'Fit method')
		self.tabs.addTab(self.tab_condition, 'Model-free method')

		self.fit_correction_layout = BackgroundFitCorrectionLayout(self, self.tab2)
		#self.tab2.setLayout(self.fit_correction_layout)

		self.populate_condition_norm_tab()
		self.tab_condition.setLayout(self.tab_condition_layout)

		layout.addWidget(self.tabs)
		self.norm_list_lbl = QLabel('Background corrections to perform:')
		hbox = QHBoxLayout()
		hbox.addWidget(self.norm_list_lbl)
		self.del_norm_btn = QPushButton("")
		self.del_norm_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.del_norm_btn.setIcon(icon(MDI6.trash_can, color="black"))
		self.del_norm_btn.setToolTip("Remove background correction")
		self.del_norm_btn.setIconSize(QSize(20, 20))
		hbox.addWidget(self.del_norm_btn, alignment=Qt.AlignRight)
		layout.addLayout(hbox)
		self.del_norm_btn.clicked.connect(self.remove_item_from_list)
		layout.addWidget(self.normalisation_list)

		self.submit_preprocessing_btn = QPushButton("Submit")
		self.submit_preprocessing_btn.setStyleSheet(self.parent.parent.button_style_sheet_2)
		self.submit_preprocessing_btn.clicked.connect(self.launch_preprocessing)
		layout.addWidget(self.submit_preprocessing_btn)

		

		self.grid_contents.addLayout(layout, 0,0,1,4)

	def launch_preprocessing(self):
		
		msgBox1 = QMessageBox()
		msgBox1.setIcon(QMessageBox.Question)
		msgBox1.setText("Do you want to apply the preprocessing\nto all wells and positions?")
		msgBox1.setWindowTitle("Selection")
		msgBox1.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
		returnValue = msgBox1.exec()
		if returnValue == QMessageBox.Cancel:
			return None
		elif returnValue == QMessageBox.Yes:
			self.parent.well_list.setCurrentIndex(self.parent.well_list.count()-1)
			self.parent.position_list.setCurrentIndex(0)
		elif returnValue == QMessageBox.No:
			msgBox2 = QMessageBox()
			msgBox2.setIcon(QMessageBox.Question)
			msgBox2.setText("Do you want to apply the preprocessing\nto the positions selected at the top only?")
			msgBox2.setWindowTitle("Selection")
			msgBox2.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
			returnValue = msgBox2.exec()
			if returnValue == QMessageBox.Cancel:
				return None
			if returnValue == QMessageBox.No:
				return None
		print('Proceed with correction...')

		for correction_protocol in self.background_correction:
			if correction_protocol['correction_type']=='model-free':

				if self.parent.well_list.currentText()=='*':
					well_option = "*"
				else:
					well_option = self.parent.well_list.currentIndex()

				
				if self.parent.position_list.currentText()=='*':
					pos_option = "*"
				else:
					pos_option = self.parent.position_list.currentIndex()-1

				if self.timeseries_rb.isChecked():
					mode = "timeseries"
				elif self.tiles_rb.isChecked():
					mode = "tiles"

				if self.regress_cb.isChecked():
					optimize_option = True
					opt_coef_range = self.coef_range_slider.value()
					opt_coef_nbr = int(self.nbr_coef_le.text())
				else:
					optimize_option = False
					opt_coef_range = None
					opt_coef_nbr = None

				if self.tab_cdt_subtract.isChecked():
					operation = "subtract"
				else:
					operation = "divide"
					clip = None

				if self.tab_cdt_clip.isChecked() and self.tab_cdt_subtract.isChecked():
					clip = True
				else:
					clip = False

				correct_background(self.exp_dir, 
								   well_option=well_option, #+1 ??
								   position_option=pos_option,
								   target_channel=self.tab_condition_channel_dropdown.currentText(),
								   mode = mode,
								   threshold_on_std = self.tab_cdt_std_le.get_threshold(),
								   frame_range = self.frame_range_slider.value(),
								   optimize_option = optimize_option,
								   opt_coef_range = opt_coef_range,
								   opt_coef_nbr = opt_coef_nbr,
								   operation = operation,
								   clip = clip,
								   export= True,
								   return_stacks=False,
								   show_progress_per_well = True,
								   show_progress_per_pos = True,
								)

				#self.parent.movie_prefix = "Corrected_"			




	def populate_condition_norm_tab(self):

		self.tab_condition_layout = QGridLayout(self.tab_condition)
		self.tab_condition_layout.setContentsMargins(15,15,15,15)
		
		channel_hbox = QHBoxLayout()
		self.tab_condition_channel_dropdown = QComboBox()
		self.tab_condition_channel_dropdown.addItems(self.channel_names)
		channel_hbox.addWidget(QLabel('Channel: '), 25)
		channel_hbox.addWidget(self.tab_condition_channel_dropdown, 75)
		self.tab_condition_layout.addLayout(channel_hbox, 0,0,1,3)

		acquisition_mode_hbox = QHBoxLayout()
		acquisition_mode_hbox.addWidget(QLabel('Stack mode: '), 25)

		self.acq_mode_group = QButtonGroup()
		self.timeseries_rb = QRadioButton('timeseries')
		self.timeseries_rb.setChecked(True)
		self.tiles_rb = QRadioButton('tiles')

		self.acq_mode_group.addButton(self.timeseries_rb, 0)
		self.acq_mode_group.addButton(self.tiles_rb, 1)

		# acq_options_hbox = QHBoxLayout()
		# acq_options_hbox.setContentsMargins(0,0,0,0)
		# acq_options_hbox.setSpacing(0)

		# self.timeseries_rb = QRadioButton('timeseries')
		# self.timeseries_rb.setChecked(True)
		# self.tiles_rb = QRadioButton('tiles')

		acquisition_mode_hbox.addWidget(self.timeseries_rb, 75//2, alignment=Qt.AlignCenter)
		acquisition_mode_hbox.addWidget(self.tiles_rb, 75//2, alignment=Qt.AlignCenter)
		# self.acq_mode_group.setLayout(acq_options_hbox)

		# acquisition_mode_hbox.addWidget(self.acq_mode_group, 75)
		self.tab_condition_layout.addLayout(acquisition_mode_hbox, 1,0,1,3)
		
		self.frame_range_slider = QLabeledRangeSlider()
		frame_selection_layout = QuickSliderLayout(label='Time range: ',
												  slider = self.frame_range_slider,
												  slider_initial_value=(0,5),
												  slider_range=(0,self.parent.len_movie),
												  slider_tooltip='frame [#]',
												  decimal_option = False,
											 	 )
		frame_selection_layout.qlabel.setToolTip('Frame range for which the background\nis most likely to be observed.')
		self.tab_condition_layout.addLayout(frame_selection_layout, 2,0,1,3) # error triggered from parenting problem?

		self.time_range_options = [self.frame_range_slider, frame_selection_layout.qlabel]
		self.timeseries_rb.toggled.connect(self.activate_time_range)
		self.tiles_rb.toggled.connect(self.activate_time_range)
		
		threshold_hbox = QHBoxLayout()
		self.thresh_lbl = QLabel("Threshold: ")
		self.thresh_lbl.setToolTip('Threshold on the STD-filtered image.\nPixel values above the threshold are\nconsidered as non-background and are\nmasked prior to background estimation.')
		threshold_hbox.addWidget(self.thresh_lbl, 25)

		self.check_threshold_cdt_btn = QPushButton()
		self.check_threshold_cdt_btn.setIcon(icon(MDI6.image_check, color="k"))
		self.check_threshold_cdt_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.check_threshold_cdt_btn.clicked.connect(self.set_std_threshold_for_model_free)

		self.check_bg_btn = QPushButton()
		self.check_bg_btn.setIcon(icon(MDI6.image_check, color="k"))
		self.check_bg_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.check_bg_btn.setToolTip('View reconstructed background.')

		self.test_correction_btn = QPushButton("")
		self.test_correction_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.test_correction_btn.setIcon(icon(MDI6.eye_outline, color="black"))
		self.test_correction_btn.setToolTip("View corrected image")
		self.test_correction_btn.setIconSize(QSize(20, 20))
		self.test_correction_btn.clicked.connect(self.preview_correction)

		self.tab_cdt_submit = QPushButton()
		self.tab_cdt_submit.setText('Add correction')
		self.tab_cdt_submit.setIcon(icon(MDI6.plus, color="#1565c0"))
		self.tab_cdt_submit.setToolTip('Add correction.')
		self.tab_cdt_submit.setIconSize(QSize(25, 25))
		#tab_cdt_submit.setStyleSheet(self.parent.parent.button_select_all)
		self.tab_cdt_submit.setStyleSheet(self.parent.parent.button_style_sheet_2)
		self.tab_cdt_submit.clicked.connect(self.add_item_to_list)

		self.tab_cdt_std_le = ThresholdLineEdit(init_value=2, connected_buttons=[self.check_threshold_cdt_btn, self.check_bg_btn, self.test_correction_btn, self.tab_cdt_submit])
		# self.tab_cdt_std_le = QLineEdit()
		# self.tab_cdt_std_le.setText('2,0')
		# self.tab_cdt_std_le.setValidator(self.onlyFloat)
		# self.tab_cdt_std_le.setPlaceholderText('px > thresh are masked')
		threshold_hbox.addWidget(self.tab_cdt_std_le, 70)

		threshold_hbox.addWidget(self.check_threshold_cdt_btn, 5)

		self.tab_condition_layout.addLayout(threshold_hbox, 3, 0, 1, 3)

		self.well_slider = QLabeledSlider()
		control_bg_layout = QuickSliderLayout(label='QC for well: ',
											  slider = self.well_slider,
											  slider_initial_value=1,
											  slider_range=(1,len(self.wells)),
											  slider_tooltip='well [#]',
											  decimal_option = False,
											  layout_ratio=(0.25,0.70)
											  )

		control_bg_layout.addWidget(self.check_bg_btn,5)

		self.check_bg_btn.clicked.connect(self.estimate_bg)
		self.tab_condition_layout.addLayout(control_bg_layout,4,0,1,3)


		self.regress_cb = QCheckBox('Optimize for each frame?')
		self.regress_cb.toggled.connect(self.activate_coef_options)
		self.regress_cb.setChecked(False)
		self.tab_condition_layout.addWidget(self.regress_cb, 5,0,1,3)


		self.coef_range_slider = QLabeledDoubleRangeSlider()
		self.coef_range_layout = QuickSliderLayout(label='Coef. range: ',
											  slider = self.coef_range_slider,
											  slider_initial_value=(0.95,1.05),
											  slider_range=(0.75,1.25),
											  slider_tooltip='Coefficient range to increase or decrease the background intensity level...',
											  )
		self.tab_condition_layout.addLayout(self.coef_range_layout, 6,0,1,3) # error triggered from parenting problem?

		coef_nbr_hbox = QHBoxLayout()
		self.nbr_coefs_lbl = QLabel("Nbr of coefs: ")
		self.nbr_coefs_lbl.setToolTip('Number of coefficients to be tested within range.\nThe more, the slower.')
		coef_nbr_hbox.addWidget(self.nbr_coefs_lbl, 25)
		self.nbr_coef_le = QLineEdit()
		self.nbr_coef_le.setText('100')
		self.nbr_coef_le.setValidator(self.onlyInt)
		self.nbr_coef_le.setPlaceholderText('nbr of coefs')
		coef_nbr_hbox.addWidget(self.nbr_coef_le, 75)
		self.tab_condition_layout.addLayout(coef_nbr_hbox, 7,0,1,3)

		self.coef_widgets = [self.coef_range_layout.qlabel, self.coef_range_slider, self.nbr_coefs_lbl, self.nbr_coef_le]
		for c in self.coef_widgets:
			c.setEnabled(False)


		operation_hbox = QHBoxLayout()
		self.tab_cdt_subtract = QRadioButton('Subtract')
		self.tab_cdt_divide = QRadioButton('Divide')
		self.tab_cdt_sd_btn_group = QButtonGroup(self)
		self.tab_cdt_sd_btn_group.addButton(self.tab_cdt_subtract)
		self.tab_cdt_sd_btn_group.addButton(self.tab_cdt_divide)
		self.tab_cdt_subtract.toggled.connect(self.activate_clipping_options)
		self.tab_cdt_divide.toggled.connect(self.activate_clipping_options)

		operation_hbox.addWidget(QLabel('Operation:'), 25)
		operation_hbox.addWidget(self.tab_cdt_subtract, 75//2, alignment=Qt.AlignCenter)
		operation_hbox.addWidget(self.tab_cdt_divide, 75//2, alignment=Qt.AlignCenter)
		self.tab_condition_layout.addLayout(operation_hbox, 8, 0, 1, 3)

		clip_hbox = QHBoxLayout()
		self.tab_cdt_clip = QRadioButton('Clip')
		self.tab_cdt_no_clip = QRadioButton("Don't clip")

		self.tab_cdt_clip_group = QButtonGroup(self)
		self.tab_cdt_clip_group.addButton(self.tab_cdt_clip)
		self.tab_cdt_clip_group.addButton(self.tab_cdt_no_clip)
		#self.tab_cdt_clip.setEnabled(True)
		#self.tab_cdt_no_clip.setEnabled(False)

		clip_hbox.addWidget(QLabel(''), 25)
		clip_hbox.addWidget(self.tab_cdt_clip, 75//4, alignment=Qt.AlignCenter)
		clip_hbox.addWidget(self.tab_cdt_no_clip, 75//4, alignment=Qt.AlignCenter)
		clip_hbox.addWidget(QLabel(''), 75//2)
		self.tab_condition_layout.addLayout(clip_hbox, 9, 0, 1, 3)
		#self.tab2_subtract.toggled.connect(self.show_clipping_options)
		#self.tab2_divide.toggled.connect(self.show_clipping_options)


		self.tab_condition_layout.addWidget(self.test_correction_btn, 9, 2)

		self.tab_condition_layout.addWidget(self.tab_cdt_submit, 10, 0, 1, 3)
		
		test_btn = QPushButton('test')
		test_btn.clicked.connect(self.open_test_widget)
		self.tab_condition_layout.addWidget(test_btn, 11, 0, 1, 3)

		self.tab_cdt_subtract.click()
		self.tab_cdt_no_clip.click()

	def open_test_widget(self):

		pass
		# self.locate_image()	
		# self.set_target_channel()
		# print(type(self.tab_cdt_std_le))
		# if self.current_stack is not None:
		# 	self.viewer = ThresholdedStackVisualizer(initial_threshold=float(self.tab_cdt_std_le.text().replace(',','.')),parent_le = self.tab_cdt_std_le,preprocessing=[['gauss',2],["std",4]], stack_path=self.current_stack, n_channels=len(self.channel_names), target_channel=self.target_channel, window_title='Test')
		# 	self.viewer.show()

	def preview_correction(self):

		if self.parent.well_list.currentText()=="*" or self.parent.position_list.currentText()=="*":
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("Please select a single position...")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None

		if self.timeseries_rb.isChecked():
			mode = "timeseries"
		elif self.tiles_rb.isChecked():
			mode = "tiles"

		if self.regress_cb.isChecked():
			optimize_option = True
			opt_coef_range = self.coef_range_slider.value()
			opt_coef_nbr = int(self.nbr_coef_le.text())
		else:
			optimize_option = False
			opt_coef_range = None
			opt_coef_nbr = None

		if self.tab_cdt_subtract.isChecked():
			operation = "subtract"
		else:
			operation = "divide"
			clip = None

		if self.tab_cdt_clip.isChecked() and self.tab_cdt_subtract.isChecked():
			clip = True
		else:
			clip = False

		corrected_stacks = correct_background(self.exp_dir, 
						   well_option=self.parent.well_list.currentIndex(), #+1 ??
						   position_option=self.parent.position_list.currentIndex()-1, #+1??
						   target_channel=self.tab_condition_channel_dropdown.currentText(),
						   mode = mode,
						   threshold_on_std = self.tab_cdt_std_le.get_threshold(),
						   frame_range = self.frame_range_slider.value(),
						   optimize_option = optimize_option,
						   opt_coef_range = opt_coef_range,
						   opt_coef_nbr = opt_coef_nbr,
						   operation = operation,
						   clip = clip,
						   export= False,
						   return_stacks=True,
						   show_progress_per_well = True,
						   show_progress_per_pos = False,
							)
		

		self.viewer = StackVisualizer(
									  stack=corrected_stacks[0],
									  window_title='Corrected channel',
									  frame_slider = True,
									  contrast_slider = True
									 )
		self.viewer.show()


	# 	self.corrected_stack = corrected_stacks[0]
	# 	self.fig_corr_stack, self.ax_corr_stack = plt.subplots(figsize=(5, 5))
	# 	self.imshow_corr_stack = FigureCanvas(self.fig_corr_stack, title="Corrected channel", interactive=True)
	# 	self.ax_corr_stack.clear()
	# 	self.im_corr_stack = self.ax_corr_stack.imshow(self.corrected_stack[0], cmap='gray',interpolation='none')

	# 	self.ax_corr_stack.set_xticks([])
	# 	self.ax_corr_stack.set_yticks([])
	# 	self.fig_corr_stack.set_facecolor('none')  # or 'None'
	# 	self.fig_corr_stack.canvas.setStyleSheet("background-color: transparent;")
	# 	self.imshow_corr_stack.canvas.draw()	

	# 	self.frame_nbr_corr_hbox = QHBoxLayout()
	# 	self.frame_nbr_corr_hbox.setContentsMargins(15,0,15,0)
	# 	self.frame_nbr_corr_hbox.addWidget(QLabel('frame: '), 10)
	# 	self.frame_corr_slider = QLabeledSlider()
	# 	self.frame_corr_slider.setSingleStep(1)
	# 	self.frame_corr_slider.setTickInterval(1)
	# 	self.frame_corr_slider.setOrientation(1)
	# 	self.frame_corr_slider.setRange(0,len(self.corrected_stack)-1)
	# 	self.frame_corr_slider.setValue(0)
	# 	self.frame_corr_slider.valueChanged.connect(self.change_frame_corr)
	# 	self.frame_nbr_corr_hbox.addWidget(self.frame_corr_slider, 90)
	# 	self.imshow_corr_stack.layout.addLayout(self.frame_nbr_corr_hbox)

	# 	self.contrast_corr_hbox = QHBoxLayout()
	# 	self.contrast_corr_hbox.addWidget(QLabel('contrast: '), 10)
	# 	self.constrast_slider_corr = QLabeledDoubleRangeSlider()
	# 	self.constrast_slider_corr.setSingleStep(0.00001)
	# 	self.constrast_slider_corr.setTickInterval(0.00001)
	# 	self.constrast_slider_corr.setOrientation(1)
	# 	self.constrast_slider_corr.setRange(np.amin(self.corrected_stack[self.corrected_stack==self.corrected_stack]),
	# 										  np.amax(self.corrected_stack[self.corrected_stack==self.corrected_stack]))
	# 	self.constrast_slider_corr.setValue([np.percentile(self.corrected_stack[self.corrected_stack==self.corrected_stack].flatten(), 1),
	# 										   np.percentile(self.corrected_stack[self.corrected_stack==self.corrected_stack].flatten(), 99.99)])
	# 	self.im_corr_stack.set_clim(vmin=np.percentile(self.corrected_stack[self.corrected_stack==self.corrected_stack].flatten(), 1),
	# 							 vmax=np.percentile(self.corrected_stack[self.corrected_stack==self.corrected_stack].flatten(), 99.99))
	# 	self.constrast_slider_corr.valueChanged.connect(self.change_corr_contrast)
	# 	self.contrast_corr_hbox.addWidget(self.constrast_slider_corr, 90)
	# 	self.imshow_corr_stack.layout.addLayout(self.contrast_corr_hbox)


	# 	self.imshow_corr_stack.show()

	# def change_corr_contrast(self, value):
		
	# 	vmin = value[0]
	# 	vmax = value[1]
	# 	self.im_corr_stack.set_clim(vmin=vmin, vmax=vmax)
	# 	self.fig_corr_stack.canvas.draw_idle()

	# def change_frame_corr(self, value):
		
	# 	self.im_corr_stack.set_data(self.corrected_stack[value])
	# 	self.change_corr_contrast(self.constrast_slider_corr.value())

	def add_item_to_list(self):

		if self.timeseries_rb.isChecked():
			mode = "timeseries"
		elif self.tiles_rb.isChecked():
			mode = "tiles"

		if self.regress_cb.isChecked():
			optimize_option = True
			opt_coef_range = self.coef_range_slider.value()
			opt_coef_nbr = int(self.nbr_coef_le.text())
		else:
			optimize_option = False
			opt_coef_range = None
			opt_coef_nbr = None

		if self.tab_cdt_subtract.isChecked():
			operation = "subtract"
		else:
			operation = "divide"
			clip = None

		if self.tab_cdt_clip.isChecked() and self.tab_cdt_subtract.isChecked():
			clip = True
		else:
			clip = False

		dictionary = {
					  "target_channel": self.tab_condition_channel_dropdown.currentText(),
					  "correction_type": "model-free",
					  "threshold_on_std": self.tab_cdt_std_le.get_threshold(),
					  "frame_range": self.frame_range_slider.value(),
					  "mode": mode,
					  "optimize_option": optimize_option,
					  "opt_coef_range": opt_coef_range,
					  "opt_coef_nbr": opt_coef_nbr,
					  "operation": operation,
					  "clip": clip
					 }

		self.background_correction.append(dictionary)
		correction_description = ""
		for index, (key, value) in enumerate(dictionary.items()):
			if index > 0:
				correction_description += ", "
			correction_description += str(key) + " : " + str(value)
		self.normalisation_list.addItem(correction_description)

	def remove_item_from_list(self):
		current_item = self.normalisation_list.currentRow()
		if current_item > -1:
			del self.background_correction[current_item]
			self.normalisation_list.takeItem(current_item)


	def activate_clipping_options(self):
		
		if self.tab_cdt_subtract.isChecked():
			self.tab_cdt_clip.setEnabled(True)
			self.tab_cdt_no_clip.setEnabled(True)

		else:
			self.tab_cdt_clip.setEnabled(False)
			self.tab_cdt_no_clip.setEnabled(False)


	def activate_coef_options(self):
		

		if self.regress_cb.isChecked():
			for c in self.coef_widgets:
				c.setEnabled(True)
		else:
			for c in self.coef_widgets:
				c.setEnabled(False)			

	def locate_image(self):

		"""
		Load the first frame of the first movie found in the experiment folder as a sample.
		"""

		movies = glob(self.parent.pos + os.sep.join(['movie', f"{self.parent.movie_prefix}*.tif"]))

		if len(movies) == 0:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("Please select a position containing a movie...")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				self.current_stack = None
				return None
		else:
			self.current_stack = movies[0]

	def set_target_channel_for_model_free(self):

		channel_indices = _extract_channel_indices_from_config(self.parent.exp_config, [self.tab_condition_channel_dropdown.currentText()])
		self.target_channel = channel_indices[0]

	def set_target_channel_for_fit(self):

		channel_indices = _extract_channel_indices_from_config(self.parent.exp_config, [self.tab2_channel_dropdown.currentText()])
		self.target_channel = channel_indices[0]

	def compute_mask(self, threshold_value):
		
		processed_frame = self.test_frame.copy().astype(float)
		processed_frame = gauss_filter(processed_frame, 2)
		std_frame = std_filter(processed_frame, 4)
		
		self.mask = std_frame > threshold_value
		self.mask = fill_label_holes(self.mask).astype(int)

	def set_std_threshold_for_model_free(self):

		self.locate_image()	
		self.set_target_channel_for_model_free()
		thresh = self.tab_cdt_std_le.get_threshold()
		if self.current_stack is not None and thresh is not None:
			self.viewer = ThresholdedStackVisualizer(initial_threshold=thresh,
													 parent_le = self.tab_cdt_std_le,
													 preprocessing=[['gauss',2],["std",4]],
													 stack_path=self.current_stack,
													 n_channels=len(self.channel_names),
													 target_channel=self.target_channel,
													 window_title='Set the exclusion threshold',
													 )
			self.viewer.show()

	def set_std_threshold_for_fit(self):

		self.locate_image()	
		self.set_target_channel_for_fit()
		thresh = self.tab2_txt_threshold.get_threshold()
		if self.current_stack is not None and thresh is not None:
			self.viewer = ThresholdedStackVisualizer(initial_threshold=thresh,
													 parent_le = self.tab2_txt_threshold,
													 preprocessing=[['gauss',2],["std",4]],
													 stack_path=self.current_stack,
													 n_channels=len(self.channel_names),
													 target_channel=self.target_channel,
													 window_title='Set the exclusion threshold',
													 )
			self.viewer.show()


	def activate_time_range(self):

		if self.timeseries_rb.isChecked():
			for wg in self.time_range_options:
				wg.setEnabled(True)
		elif self.tiles_rb.isChecked():
			for wg in self.time_range_options:
				wg.setEnabled(False)

	def estimate_bg(self):

		if self.timeseries_rb.isChecked():
			mode = "timeseries"
		elif self.tiles_rb.isChecked():
			mode = "tiles"

		bg = estimate_background_per_condition(
											  self.exp_dir, 
											  well_option = self.well_slider.value() - 1,
											  frame_range = self.frame_range_slider.value(),
											  target_channel = self.tab_condition_channel_dropdown.currentText(),
											  show_progress_per_pos = True,
											  threshold_on_std = self.tab_cdt_std_le.get_threshold(),
											  mode = mode,
										  	)
		bg = bg[0]
		bg = bg['bg']

		self.viewer = StackVisualizer(
									  stack=[bg],
									  window_title='Reconstructed background',
									  frame_slider = False,
									 )
		self.viewer.show()