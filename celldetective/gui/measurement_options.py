from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QScrollArea, QComboBox, QFrame, QCheckBox, QFileDialog, QGridLayout, QTextEdit, QLineEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize
from celldetective.gui.gui_utils import center_window, FeatureChoice, ListWidget, QHSeperationLine, FigureCanvas, GeometryChoice, OperationChoice
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider,QLabeledSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from celldetective.utils import extract_experiment_channels, get_software_location
from celldetective.io import interpret_tracking_configuration, load_frames
from celldetective.measure import compute_haralick_features
import numpy as np
import json
from shutil import copyfile
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob

class ConfigMeasurements(QMainWindow):
	
	"""
	UI to set tracking parameters for bTrack.

	"""

	def __init__(self, parent=None):
		
		super().__init__()
		self.parent = parent
		self.setWindowTitle("Configure measurements")
		self.mode = self.parent.mode
		self.exp_dir = self.parent.exp_dir
		if self.mode=="targets":
			self.config_name = "btrack_config_targets.json"
			self.measure_instructions_path = self.parent.exp_dir + "measurement_instructions_targets.json"
		elif self.mode=="effectors":
			self.config_name = "btrack_config_effectors.json"
			self.measure_instructions_path = self.parent.exp_dir + "measurement_instructions_effectors.json"
		self.soft_path = get_software_location()
		
		exp_config = self.exp_dir +"config.ini"
		self.config_path = self.exp_dir + self.config_name
		self.channel_names, self.channels = extract_experiment_channels(exp_config)
		self.channel_names = np.array(self.channel_names)
		self.channels = np.array(self.channels)

		center_window(self)
		self.setMinimumWidth(500)
		self.setMinimumHeight(800)
		self.setMaximumHeight(1160)
		self.populate_widget()
		#self.load_previous_tracking_instructions()

	def populate_widget(self):

		"""
		Create the multibox design.

		"""
		
		# Create button widget and layout
		self.scroll_area = QScrollArea(self)
		self.button_widget = QWidget()
		main_layout = QVBoxLayout()
		self.button_widget.setLayout(main_layout)
		main_layout.setContentsMargins(30,30,30,30)

		# first frame for FEATURES
		self.features_frame = QFrame()
		self.features_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_features_frame()
		main_layout.addWidget(self.features_frame)

		# second frame for ISOTROPIC MEASUREMENTS
		self.iso_frame = QFrame()
		self.iso_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_iso_frame()
		main_layout.addWidget(self.iso_frame)

		self.submit_btn = QPushButton('Save')
		self.submit_btn.setStyleSheet(self.parent.parent.parent.button_style_sheet)
		self.submit_btn.clicked.connect(self.write_instructions)
		main_layout.addWidget(self.submit_btn)

		#self.populate_left_panel()
		#grid.addLayout(self.left_side, 0, 0, 1, 1)
		self.button_widget.adjustSize()

		self.scroll_area.setAlignment(Qt.AlignCenter)
		self.scroll_area.setWidget(self.button_widget)
		self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		self.scroll_area.setWidgetResizable(True)
		self.setCentralWidget(self.scroll_area)
		self.show()

		QApplication.processEvents()
		self.adjustScrollArea()


	def populate_iso_frame(self):

		"""
		Add widgets and layout in the POST-PROCESSING frame.
		"""

		grid = QGridLayout(self.iso_frame)

		self.iso_lbl = QLabel("ISOTROPIC MEASUREMENTS")
		self.iso_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.iso_lbl, 0, 0, 1, 4, alignment=Qt.AlignCenter)
		self.generate_iso_contents()
		grid.addWidget(self.ContentsIso, 1, 0, 1, 4, alignment=Qt.AlignTop)

	def populate_features_frame(self):

		"""
		Add widgets and layout in the FEATURES frame.
		"""

		grid = QGridLayout(self.features_frame)

		self.feature_lbl = QLabel("FEATURES")
		self.feature_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.feature_lbl, 0, 0, 1, 4, alignment=Qt.AlignCenter)

		self.generate_feature_panel_contents()
		grid.addWidget(self.ContentsFeatures, 1, 0, 1, 4, alignment=Qt.AlignTop)


	def generate_iso_contents(self):

		self.ContentsIso = QFrame()
		layout = QVBoxLayout(self.ContentsIso)
		layout.setContentsMargins(0,0,0,0)


		radii_layout = QHBoxLayout()
		self.radii_lbl = QLabel('Measurement radii (from center):')
		self.radii_lbl.setToolTip('Define radii or donughts for intensity measurements.')
		radii_layout.addWidget(self.radii_lbl, 90)

		self.del_radius_btn = QPushButton("")
		self.del_radius_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.del_radius_btn.setIcon(icon(MDI6.trash_can,color="black"))
		self.del_radius_btn.setToolTip("Remove radius")
		self.del_radius_btn.setIconSize(QSize(20, 20))
		radii_layout.addWidget(self.del_radius_btn, 5)

		self.add_radius_btn = QPushButton("")
		self.add_radius_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.add_radius_btn.setIcon(icon(MDI6.plus,color="black"))
		self.add_radius_btn.setToolTip("Add radius")
		self.add_radius_btn.setIconSize(QSize(20, 20))	
		radii_layout.addWidget(self.add_radius_btn, 5)
		layout.addLayout(radii_layout)
		
		self.radii_list = ListWidget(self, GeometryChoice, initial_features=["10"], dtype=int)
		layout.addWidget(self.radii_list)

		self.del_radius_btn.clicked.connect(self.radii_list.removeSel)
		self.add_radius_btn.clicked.connect(self.radii_list.addItem)

		# Operation
		operation_layout = QHBoxLayout()
		self.op_lbl = QLabel('Operation to perform:')
		self.op_lbl.setToolTip('Set the operations to perform inside the ROI.')
		operation_layout.addWidget(self.op_lbl, 90)

		self.del_op_btn = QPushButton("")
		self.del_op_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.del_op_btn.setIcon(icon(MDI6.trash_can,color="black"))
		self.del_op_btn.setToolTip("Remove operation")
		self.del_op_btn.setIconSize(QSize(20, 20))
		operation_layout.addWidget(self.del_op_btn, 5)

		self.add_op_btn = QPushButton("")
		self.add_op_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.add_op_btn.setIcon(icon(MDI6.plus,color="black"))
		self.add_op_btn.setToolTip("Add operation")
		self.add_op_btn.setIconSize(QSize(20, 20))	
		operation_layout.addWidget(self.add_op_btn, 5)
		layout.addLayout(operation_layout)
		
		self.operations_list = ListWidget(self, OperationChoice, initial_features=["mean"])
		layout.addWidget(self.operations_list)

		self.del_op_btn.clicked.connect(self.operations_list.removeSel)
		self.add_op_btn.clicked.connect(self.operations_list.addItem)


	def generate_feature_panel_contents(self):
		
		self.ContentsFeatures = QFrame()
		layout = QVBoxLayout(self.ContentsFeatures)
		layout.setContentsMargins(0,0,0,0)

		feature_layout = QHBoxLayout()
		feature_layout.setContentsMargins(0,0,0,0)

		self.feature_lbl = QLabel("Add features:")
		self.del_feature_btn = QPushButton("")
		self.del_feature_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.del_feature_btn.setIcon(icon(MDI6.trash_can,color="black"))
		self.del_feature_btn.setToolTip("Remove feature")
		self.del_feature_btn.setIconSize(QSize(20, 20))

		self.add_feature_btn = QPushButton("")
		self.add_feature_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.add_feature_btn.setIcon(icon(MDI6.filter_plus,color="black"))
		self.add_feature_btn.setToolTip("Add feature")
		self.add_feature_btn.setIconSize(QSize(20, 20))		

		self.features_list = ListWidget(self, FeatureChoice, initial_features=['area','intensity_mean',])

		self.del_feature_btn.clicked.connect(self.features_list.removeSel)
		self.add_feature_btn.clicked.connect(self.features_list.addItem)

		feature_layout.addWidget(self.feature_lbl, 90)
		feature_layout.addWidget(self.del_feature_btn, 5)
		feature_layout.addWidget(self.add_feature_btn, 5)
		layout.addLayout(feature_layout)
		layout.addWidget(self.features_list)

		self.feat_sep2 = QHSeperationLine()
		layout.addWidget(self.feat_sep2)

		contour_layout = QHBoxLayout()
		self.border_dist_lbl = QLabel('Contour measurements (from edge of mask):')
		self.border_dist_lbl.setToolTip('Apply the intensity measurements defined above\nto a slice of each cell mask, defined as distance\nfrom the edge.')
		contour_layout.addWidget(self.border_dist_lbl, 90)

		self.del_contour_btn = QPushButton("")
		self.del_contour_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.del_contour_btn.setIcon(icon(MDI6.trash_can,color="black"))
		self.del_contour_btn.setToolTip("Remove distance")
		self.del_contour_btn.setIconSize(QSize(20, 20))
		contour_layout.addWidget(self.del_contour_btn, 5)

		self.add_contour_btn = QPushButton("")
		self.add_contour_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.add_contour_btn.setIcon(icon(MDI6.plus,color="black"))
		self.add_contour_btn.setToolTip("Add distance")
		self.add_contour_btn.setIconSize(QSize(20, 20))	
		contour_layout.addWidget(self.add_contour_btn, 5)
		layout.addLayout(contour_layout)
		
		self.contours_list = ListWidget(self, GeometryChoice, initial_features=[], dtype=int)
		layout.addWidget(self.contours_list)

		self.del_contour_btn.clicked.connect(self.contours_list.removeSel)
		self.add_contour_btn.clicked.connect(self.contours_list.addItem)

		self.feat_sep3 = QHSeperationLine()
		layout.addWidget(self.feat_sep3)

		# Haralick features parameters
		self.activate_haralick_btn = QCheckBox('activate Haralick texture features')
		self.activate_haralick_btn.toggled.connect(self.show_haralick_options)

		self.haralick_channel_choice = QComboBox()
		self.haralick_channel_choice.addItems(self.channel_names)
		self.haralick_channel_lbl = QLabel('Target channel: ')

		self.haralick_distance_le = QLineEdit("1")
		self.haralick_distance_lbl = QLabel('Distance: ')

		self.haralick_n_gray_levels_le = QLineEdit("256")
		self.haralick_n_gray_levels_lbl = QLabel('# gray levels: ')

		# Slider to set vmin & vmax
		self.haralick_scale_slider = QLabeledDoubleSlider()
		self.haralick_scale_slider.setSingleStep(0.05)
		self.haralick_scale_slider.setTickInterval(0.05)
		self.haralick_scale_slider.setSingleStep(1)
		self.haralick_scale_slider.setOrientation(1)
		self.haralick_scale_slider.setRange(0,1)
		self.haralick_scale_slider.setValue(0.5)
		self.haralick_scale_lbl = QLabel('Scale: ')

		self.haralick_percentile_min_le = QLineEdit('0.01')
		self.haralick_percentile_max_le = QLineEdit('99.9')
		self.haralick_normalization_mode_btn = QPushButton()
		self.haralick_normalization_mode_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.haralick_normalization_mode_btn.setIcon(icon(MDI6.percent_circle,color="black"))
		self.haralick_normalization_mode_btn.setIconSize(QSize(20, 20))		
		self.haralick_normalization_mode_btn.setToolTip("Switch to absolute normalization values.")
		self.percentile_mode = True

		self.haralick_percentile_min_lbl = QLabel('Min percentile: ')
		self.haralick_percentile_max_lbl = QLabel('Max percentile: ')

		self.haralick_hist_btn = QPushButton()
		self.haralick_hist_btn.clicked.connect(self.control_haralick_intensity_histogram)
		self.haralick_hist_btn.setIcon(icon(MDI6.poll,color="k"))
		self.haralick_hist_btn.setStyleSheet(self.parent.parent.parent.button_select_all)

		self.haralick_digit_btn = QPushButton()
		self.haralick_digit_btn.clicked.connect(self.control_haralick_digitalization)
		self.haralick_digit_btn.setIcon(icon(MDI6.image_check,color="k"))
		self.haralick_digit_btn.setStyleSheet(self.parent.parent.parent.button_select_all)

		self.haralick_layout = QVBoxLayout()
		self.haralick_layout.setContentsMargins(20,20,20,20)

		activate_layout = QHBoxLayout()
		activate_layout.addWidget(self.activate_haralick_btn, 80)
		activate_layout.addWidget(self.haralick_hist_btn, 10)
		activate_layout.addWidget(self.haralick_digit_btn, 10)
		self.haralick_layout.addLayout(activate_layout)

		channel_layout = QHBoxLayout()
		channel_layout.addWidget(self.haralick_channel_lbl, 40)
		channel_layout.addWidget(self.haralick_channel_choice, 60)
		self.haralick_layout.addLayout(channel_layout)

		distance_layout = QHBoxLayout()
		distance_layout.addWidget(self.haralick_distance_lbl,40)
		distance_layout.addWidget(self.haralick_distance_le, 60)
		self.haralick_layout.addLayout(distance_layout)

		gl_layout = QHBoxLayout()
		gl_layout.addWidget(self.haralick_n_gray_levels_lbl,40)
		gl_layout.addWidget(self.haralick_n_gray_levels_le,60)
		self.haralick_layout.addLayout(gl_layout)

		slider_layout = QHBoxLayout()
		slider_layout.addWidget(self.haralick_scale_lbl,40)
		slider_layout.addWidget(self.haralick_scale_slider,60)
		self.haralick_layout.addLayout(slider_layout)

		slider_min_percentile_layout = QHBoxLayout()
		slider_min_percentile_layout.addWidget(self.haralick_percentile_min_lbl,40)
		slider_min_percentile_layout.addWidget(self.haralick_percentile_min_le,55)
		slider_min_percentile_layout.addWidget(self.haralick_normalization_mode_btn, 5)
		self.haralick_layout.addLayout(slider_min_percentile_layout)

		slider_max_percentile_layout = QHBoxLayout()
		slider_max_percentile_layout.addWidget(self.haralick_percentile_max_lbl,40)
		slider_max_percentile_layout.addWidget(self.haralick_percentile_max_le,60)
		self.haralick_layout.addLayout(slider_max_percentile_layout)

		self.haralick_to_hide = [self.haralick_hist_btn, self.haralick_digit_btn, self.haralick_channel_lbl, self.haralick_channel_choice,
								self.haralick_distance_le, self.haralick_distance_lbl, self.haralick_n_gray_levels_le, self.haralick_n_gray_levels_lbl,
								self.haralick_scale_lbl, self.haralick_scale_slider, self.haralick_percentile_min_lbl, self.haralick_percentile_min_le,
								self.haralick_percentile_max_lbl, self.haralick_percentile_max_le, self.haralick_normalization_mode_btn]

		self.features_to_disable = [self.feature_lbl, self.del_feature_btn, self.add_feature_btn, self.features_list, 
									self.activate_haralick_btn]

		self.activate_haralick_btn.setChecked(True)
		self.haralick_normalization_mode_btn.clicked.connect(self.switch_to_absolute_normalization_mode)
		layout.addLayout(self.haralick_layout)

	def switch_to_absolute_normalization_mode(self):
		if self.percentile_mode:
			self.percentile_mode = False
			self.haralick_normalization_mode_btn.setIcon(icon(MDI6.percent_circle_outline,color="black"))
			self.haralick_normalization_mode_btn.setIconSize(QSize(20, 20))		
			self.haralick_normalization_mode_btn.setToolTip("Switch to percentile normalization values.")			
			self.haralick_percentile_min_lbl.setText('Min value: ')
			self.haralick_percentile_max_lbl.setText('Max value: ')
			self.haralick_percentile_min_le.setText('0')
			self.haralick_percentile_max_le.setText('10000')

		else:
			self.percentile_mode = True
			self.haralick_normalization_mode_btn.setIcon(icon(MDI6.percent_circle,color="black"))
			self.haralick_normalization_mode_btn.setIconSize(QSize(20, 20))
			self.haralick_normalization_mode_btn.setToolTip("Switch to absolute normalization values.")			
			self.haralick_percentile_min_lbl.setText('Min percentile: ')
			self.haralick_percentile_max_lbl.setText('Max percentile: ')			
			self.haralick_percentile_min_le.setText('0.01')
			self.haralick_percentile_max_le.setText('99.99')


	# def generate_config_panel_contents(self):
		
	# 	self.ContentsConfig = QFrame()
	# 	layout = QVBoxLayout(self.ContentsConfig)
	# 	layout.setContentsMargins(0,0,0,0)

	# 	btrack_config_layout = QHBoxLayout()
	# 	self.config_lbl = QLabel("bTrack configuration: ")
	# 	btrack_config_layout.addWidget(self.config_lbl, 90)

	# 	self.upload_btrack_config_btn = QPushButton()
	# 	self.upload_btrack_config_btn.setIcon(icon(MDI6.plus,color="black"))
	# 	self.upload_btrack_config_btn.setIconSize(QSize(20, 20))
	# 	self.upload_btrack_config_btn.setToolTip("Upload a new bTrack configuration.")
	# 	self.upload_btrack_config_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
	# 	self.upload_btrack_config_btn.clicked.connect(self.upload_btrack_config)
	# 	btrack_config_layout.addWidget(self.upload_btrack_config_btn, 5)  #4,3,1,1, alignment=Qt.AlignLeft

	# 	self.reset_config_btn = QPushButton()
	# 	self.reset_config_btn.setIcon(icon(MDI6.arrow_u_right_top,color="black"))
	# 	self.reset_config_btn.setIconSize(QSize(20, 20))
	# 	self.reset_config_btn.setToolTip("Reset the configuration to the default bTrack config.")
	# 	self.reset_config_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
	# 	self.reset_config_btn.clicked.connect(self.reset_btrack_config)
	# 	btrack_config_layout.addWidget(self.reset_config_btn, 5)  #4,3,1,1, alignment=Qt.AlignLeft

	# 	layout.addLayout(btrack_config_layout)

	# 	self.config_le = QTextEdit()
	# 	self.config_le.setMinimumHeight(150)
	# 	#self.config_le.setStyleSheet("""
	# 	#							background: #EEEDEB;
	# 	#							border: 2px solid black;
	# 	#							""")
	# 	layout.addWidget(self.config_le)
	# 	self.load_cell_config()


	def show_haralick_options(self):

		"""
		Show the Haralick texture options.
		"""

		if self.activate_haralick_btn.isChecked():
			for element in self.haralick_to_hide:
				element.setEnabled(True)
		else:
			for element in self.haralick_to_hide:
				element.setEnabled(False)   		

	def upload_btrack_config(self):

		"""
		Upload a specific bTrack config to the experiment folder for the cell population.
		"""

		self.file_dialog = QFileDialog()
		try:
			modelpath = self.soft_path+"/celldetective/models/tracking_configs/"
			self.filename = self.file_dialog.getOpenFileName(None, "Load config", modelpath, "json files (*.json)")[0]
			if self.filename!=self.config_path:
				copyfile(self.filename, self.config_path)
			self.load_cell_config()
		except Exception as e:
			print(e, modelpath)
			return None

	def reset_btrack_config(self):

		"""
		Set the bTrack config to the default bTrack config.
		"""

		msgBox = QMessageBox()
		msgBox.setIcon(QMessageBox.Question)
		msgBox.setText("You are about to revert to the default bTrack configuration? Do you want to proceed?")
		msgBox.setWindowTitle("Confirm")
		msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
		returnValue = msgBox.exec()
		if returnValue == QMessageBox.Yes:
			config = interpret_tracking_configuration(None)
			if config!=self.config_path:
				copyfile(config, self.config_path)
			self.load_cell_config()
		else:
			return None

	def activate_feature_options(self):

		"""
		Tick the features option.
		"""

		self.switch_feature_option()
		if self.features_ticked:
			for element in self.features_to_disable:
				element.setEnabled(True)
			self.select_features_btn.setIcon(icon(MDI6.checkbox_outline,color="black"))
			self.select_features_btn.setIconSize(QSize(20, 20))
		else:
			for element in self.features_to_disable:
				element.setEnabled(False)
			self.select_features_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
			self.select_features_btn.setIconSize(QSize(20, 20))
			self.features_list.list_widget.clearSelection()
			self.activate_haralick_btn.setChecked(False)


	def activate_post_proc_options(self):

		"""
		Tick the features option.
		"""

		self.switch_post_proc_option()
		if self.post_proc_ticked:
			for element in self.post_proc_options_to_disable:
				element.setEnabled(True)
			self.select_post_proc_btn.setIcon(icon(MDI6.checkbox_outline,color="black"))
			self.select_post_proc_btn.setIconSize(QSize(20, 20))
		else:
			for element in self.post_proc_options_to_disable:
				element.setEnabled(False)
			self.select_post_proc_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
			self.select_post_proc_btn.setIconSize(QSize(20, 20))

	def switch_feature_option(self):

		"""
		Switch the feature option.
		"""

		if self.features_ticked == True:
			self.features_ticked = False
		else:
			self.features_ticked = True

	def switch_post_proc_option(self):

		"""
		Switch the feature option.
		"""

		if self.post_proc_ticked == True:
			self.post_proc_ticked = False
		else:
			self.post_proc_ticked = True	

	def adjustScrollArea(self):
		
		"""
		Auto-adjust scroll area to fill space 
		(from https://stackoverflow.com/questions/66417576/make-qscrollarea-use-all-available-space-of-qmainwindow-height-axis)
		"""

		step = 5
		while self.scroll_area.verticalScrollBar().isVisible() and self.height() < self.maximumHeight():
			self.resize(self.width(), self.height() + step)

	def load_cell_config(self):

		"""
		Load the cell configuration and write in the QLineEdit.
		"""

		file_name = interpret_tracking_configuration(self.config_path)
		with open(file_name, 'r') as f:
			json_data = json.load(f)
			self.config_le.setText(json.dumps(json_data, indent=4))

	def write_instructions(self):

		"""
		Write the selected options in a json file for later reading by the software.
		"""

		print('Writing instructions...')
		measurement_options = {}
		features = self.features_list.getItems()
		if not features:
			features = None
		measurement_options.update({'features': features})

		border_distances = self.contours_list.getItems()
		if not border_distances:
			border_distances = None
		measurement_options.update({'border_distances': border_distances})
		
		self.extract_haralick_options()
		measurement_options.update({'haralick_options': self.haralick_options})

		intensity_measurement_radii = self.radii_list.getItems()
		if not intensity_measurement_radii:
			intensity_measurement_radii = None

		isotropic_operations = self.operations_list.getItems()
		if not isotropic_operations:
			isotropic_operations = None
			intensity_measurement_radii = None
		measurement_options.update({'intensity_measurement_radii': intensity_measurement_radii,
									'isotropic_operations': isotropic_operations})

		print('Measurement instructions: ', measurement_options)
		file_name = self.measure_instructions_path
		with open(file_name, 'w') as f:
			json.dump(measurement_options, f, indent=4)

		print('Done.')
		self.close()

	def uncheck_post_proc(self):
		self.select_post_proc_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
		self.select_post_proc_btn.setIconSize(QSize(20, 20))
		self.post_proc_ticked = False
		for element in self.post_proc_options_to_disable:
			element.setEnabled(False)

	def check_post_proc(self):
		self.select_post_proc_btn.setIcon(icon(MDI6.checkbox_outline,color="black"))
		self.select_post_proc_btn.setIconSize(QSize(20, 20))
		self.post_proc_ticked = True		
		for element in self.post_proc_options_to_disable:
			element.setEnabled(True)

	def uncheck_features(self):
		self.select_features_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
		self.select_features_btn.setIconSize(QSize(20, 20))
		self.features_ticked = False
		for element in self.features_to_disable:
			element.setEnabled(False)

	def check_features(self):
		self.select_features_btn.setIcon(icon(MDI6.checkbox_outline,color="black"))
		self.select_features_btn.setIconSize(QSize(20, 20))
		self.features_ticked = True		
		for element in self.features_to_disable:
			element.setEnabled(True)

	def extract_haralick_options(self):

		if self.activate_haralick_btn.isChecked():
			self.haralick_options = {"target_channel": self.haralick_channel_choice.currentIndex(),
								"scale_factor": float(self.haralick_scale_slider.value()),
								"n_intensity_bins": int(self.haralick_n_gray_levels_le.text()),
								"distance" : int(self.haralick_distance_le.text()),
								}
			if self.percentile_mode:
				self.haralick_options.update({"percentiles": (float(self.haralick_percentile_min_le.text()), float(self.haralick_percentile_max_le.text())), "clip_values": None})
			else:
				self.haralick_options.update({"percentiles": None, "clip_values": (float(self.haralick_percentile_min_le.text()), float(self.haralick_percentile_max_le.text()))})

		else:
			self.haralick_options = None		

	def load_previous_tracking_instructions(self):

		"""
		Read the tracking options from a previously written json file.
		"""

		print('Reading instructions..')
		if os.path.exists(self.track_instructions_write_path):
			with open(self.track_instructions_write_path, 'r') as f:
				tracking_instructions = json.load(f)
				print(tracking_instructions)
				
				# Features
				features = tracking_instructions['features']
				if (features is not None) and len(features)>0:
					self.check_features()
					self.ContentsFeatures.show()
					self.features_list.list_widget.clear()
					self.features_list.list_widget.addItems(features)
				else:
					self.ContentsFeatures.hide()
					self.uncheck_features()

				# Uncheck channels that are masked
				mask_channels = tracking_instructions['mask_channels']
				if (mask_channels is not None) and len(mask_channels)>0:
					for ch in mask_channels:
						for cb in self.mask_channels_cb:
							if cb.text()==ch:
								cb.setChecked(False)

				haralick_options = tracking_instructions['haralick_options']
				if haralick_options is None:
					self.activate_haralick_btn.setChecked(False)
					self.show_haralick_options()
				else:
					self.activate_haralick_btn.setChecked(True)
					self.show_haralick_options()
					if 'target_channel' in haralick_options:
						idx = haralick_options['target_channel']
						#idx = self.haralick_channel_choice.findText(text_to_find)
						self.haralick_channel_choice.setCurrentIndex(idx)
					if 'scale_factor' in haralick_options:
						self.haralick_scale_slider.setValue(float(haralick_options['scale_factor']))
					if ('percentiles' in haralick_options) and (haralick_options['percentiles'] is not None):
						perc = list(haralick_options['percentiles'])
						self.haralick_percentile_min_le.setText(str(perc[0]))
						self.haralick_percentile_max_le.setText(str(perc[1]))
					if ('clip_values' in haralick_options) and (haralick_options['clip_values'] is not None):
						values = list(haralick_options['clip_values'])
						self.haralick_percentile_min_le.setText(str(values[0]))
						self.haralick_percentile_max_le.setText(str(values[1]))
						self.percentile_mode=True
						self.switch_to_absolute_normalization_mode()
					if 'n_intensity_bins' in haralick_options:
						self.haralick_n_gray_levels_le.setText(str(haralick_options['n_intensity_bins']))
					if 'distance' in haralick_options:
						self.haralick_distance_le.setText(str(haralick_options['distance']))
		


				# Post processing options
				post_processing_options = tracking_instructions['post_processing_options']
				if post_processing_options is None:
					self.uncheck_post_proc()
					self.ContentsPostProc.hide()
					for element in [self.remove_not_in_last_checkbox, self.remove_not_in_first_checkbox, self.interpolate_gaps_checkbox, 
									self.extrapolate_post_checkbox, self.extrapolate_pre_checkbox, self.interpolate_na_features_checkbox]:
						element.setChecked(False)
					self.min_tracklength_slider.setValue(0)

				else:
					self.check_post_proc()
					self.ContentsPostProc.show()
					if "minimum_tracklength" in post_processing_options:
						self.min_tracklength_slider.setValue(int(post_processing_options["minimum_tracklength"]))
					if "remove_not_in_first" in post_processing_options:
						self.remove_not_in_first_checkbox.setChecked(post_processing_options["remove_not_in_first"])
					if "remove_not_in_last" in post_processing_options:
						self.remove_not_in_last_checkbox.setChecked(post_processing_options["remove_not_in_last"])
					if "interpolate_position_gaps" in post_processing_options:
						self.interpolate_gaps_checkbox.setChecked(post_processing_options["interpolate_position_gaps"])
					if "extrapolate_tracks_pre" in post_processing_options:
						self.extrapolate_pre_checkbox.setChecked(post_processing_options["extrapolate_tracks_pre"])
					if "extrapolate_tracks_post" in post_processing_options:
						self.extrapolate_post_checkbox.setChecked(post_processing_options["extrapolate_tracks_post"])
					if "interpolate_na" in post_processing_options:
						self.interpolate_na_features_checkbox.setChecked(post_processing_options["interpolate_na"])

	def locate_image(self):
		
		"""
		Load the first frame of the first movie found in the experiment folder as a sample.
		"""

		movies = glob(self.parent.parent.exp_dir + f"*/*/movie/{self.parent.parent.movie_prefix}*.tif")
		if len(movies)==0:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("No movies are detected in the experiment folder. Cannot load an image to test Haralick.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Yes:
				self.test_frame = None
				return None
		else:
			stack0 = movies[0]
			n_channels = len(self.channels)
			self.test_frame = load_frames(np.arange(n_channels), stack0, scale=None, normalize_input=False)


	def control_haralick_digitalization(self):

		"""
		Load an image for the first experiment movie found. 
		Apply the Haralick parameters and check the result of the digitization (normalization + binning of intensities).

		"""

		self.locate_image()
		self.extract_haralick_options()
		if self.test_frame is not None:
			digitized_img = compute_haralick_features(self.test_frame, np.zeros(self.test_frame.shape[:2]), 
													 channels=self.channel_names, return_digit_image_only=True,
													 **self.haralick_options
													 )

			self.fig, self.ax = plt.subplots()
			divider = make_axes_locatable(self.ax)
			cax = divider.append_axes('right', size='5%', pad=0.05)

			self.imshow_digit_window = FigureCanvas(self.fig, title="Haralick: control digitization")
			self.ax.clear()
			im = self.ax.imshow(digitized_img, cmap='gray')
			self.fig.colorbar(im, cax=cax, orientation='vertical')
			self.ax.set_xticks([])
			self.ax.set_yticks([])
			self.fig.set_facecolor('none')  # or 'None'
			self.fig.canvas.setStyleSheet("background-color: transparent;")
			self.imshow_digit_window.canvas.draw()
			self.imshow_digit_window.show()

	def control_haralick_intensity_histogram(self):

		"""
		Load an image for the first experiment movie found. 
		Apply the Haralick normalization parameters and check the normalized intensity histogram.
		
		"""

		self.locate_image()
		self.extract_haralick_options()
		if self.test_frame is not None:
			norm_img = compute_haralick_features(self.test_frame, np.zeros(self.test_frame.shape[:2]), 
													 channels=self.channel_names, return_norm_image_only=True,
													 **self.haralick_options
													 )
			self.fig, self.ax = plt.subplots(1,1,figsize=(4,3))
			self.hist_window = FigureCanvas(self.fig, title="Haralick: control digitized histogram")
			self.ax.clear()
			self.ax.hist(norm_img.flatten(), bins=self.haralick_options['n_intensity_bins'])
			self.ax.set_xlabel('gray level value')
			self.ax.set_ylabel('#')
			plt.tight_layout()
			self.fig.set_facecolor('none')  # or 'None'
			self.fig.canvas.setStyleSheet("background-color: transparent;")
			self.hist_window.canvas.draw()
			self.hist_window.show()





