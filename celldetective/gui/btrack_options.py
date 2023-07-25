from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QScrollArea, QComboBox, QFrame, QCheckBox, QFileDialog, QGridLayout, QTextEdit, QLineEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize
from celldetective.gui.gui_utils import center_window, FeatureChoice, ListWidget, QHSeperationLine
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider,QLabeledSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from celldetective.utils import extract_experiment_channels, get_software_location
from celldetective.io import interpret_tracking_configuration
import numpy as np
import json
from shutil import copyfile
import os


class ConfigTracking(QMainWindow):

	def __init__(self, parent=None):
		
		super().__init__()
		self.parent = parent
		self.setWindowTitle("Configure tracking")
		self.mode = self.parent.mode
		self.exp_dir = self.parent.exp_dir
		if self.mode=="targets":
			self.config_name = "btrack_config_targets.json"
		elif self.mode=="effectors":
			self.config_name = "btrack_config_effectors.json"
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

	def populate_widget(self):
		
		# Create button widget and layout
		self.scroll_area = QScrollArea(self)
		self.button_widget = QWidget()
		main_layout = QVBoxLayout()
		self.button_widget.setLayout(main_layout)
		main_layout.setContentsMargins(30,30,30,30)

		# First collapsable Frame CONFIG
		self.config_frame = QFrame()
		self.config_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_config_frame()
		main_layout.addWidget(self.config_frame)

		# Second collapsable frame FEATURES
		self.features_frame = QFrame()
		self.features_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_features_frame()
		main_layout.addWidget(self.features_frame)

		# Third collapsable frame POST-PROCESSING
		self.post_proc_frame = QFrame()
		self.post_proc_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_post_proc_frame()
		main_layout.addWidget(self.post_proc_frame)

		self.submit_btn = QPushButton('Submit')
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

	def populate_post_proc_frame(self):

		grid = QGridLayout(self.post_proc_frame)

		self.select_post_proc_btn = QPushButton()
		self.select_post_proc_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
		self.select_post_proc_btn.setIconSize(QSize(20, 20))
		self.post_proc_ticked = False
		self.select_post_proc_btn.clicked.connect(self.activate_post_proc_options)
		self.select_post_proc_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		grid.addWidget(self.select_post_proc_btn, 0,0,1,4,alignment=Qt.AlignLeft)

		self.post_proc_lbl = QLabel("POST-PROCESSING")
		self.post_proc_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.post_proc_lbl, 0, 0, 1, 4, alignment=Qt.AlignCenter)

		self.collapse_post_proc_btn = QPushButton()
		self.collapse_post_proc_btn.setIcon(icon(MDI6.chevron_down,color="black"))
		self.collapse_post_proc_btn.setIconSize(QSize(20, 20))
		self.collapse_post_proc_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		grid.addWidget(self.collapse_post_proc_btn, 0, 0, 1, 4, alignment=Qt.AlignRight)

		self.generate_post_proc_panel_contents()
		grid.addWidget(self.ContentsPostProc, 1, 0, 1, 4, alignment=Qt.AlignTop)
		self.collapse_post_proc_btn.clicked.connect(lambda: self.ContentsPostProc.setHidden(not self.ContentsPostProc.isHidden()))
		# self.collapse_post_proc_btn.clicked.connect(self.collapse_features_advanced)
		self.ContentsPostProc.hide()


	def populate_features_frame(self):

		grid = QGridLayout(self.features_frame)

		self.select_features_btn = QPushButton()
		self.select_features_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
		self.select_features_btn.setIconSize(QSize(20, 20))
		self.features_ticked = True
		self.select_features_btn.clicked.connect(self.activate_feature_options)
		self.select_features_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		grid.addWidget(self.select_features_btn, 0,0,1,4,alignment=Qt.AlignLeft)


		self.feature_lbl = QLabel("FEATURES")
		self.feature_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.feature_lbl, 0, 0, 1, 4, alignment=Qt.AlignCenter)

		self.collapse_features_btn = QPushButton()
		self.collapse_features_btn.setIcon(icon(MDI6.chevron_down,color="black"))
		self.collapse_features_btn.setIconSize(QSize(20, 20))
		self.collapse_features_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		grid.addWidget(self.collapse_features_btn, 0, 0, 1, 4, alignment=Qt.AlignRight)

		self.generate_feature_panel_contents()
		grid.addWidget(self.ContentsFeatures, 1, 0, 1, 4, alignment=Qt.AlignTop)
		self.collapse_features_btn.clicked.connect(lambda: self.ContentsFeatures.setHidden(not self.ContentsFeatures.isHidden()))
		self.collapse_features_btn.clicked.connect(self.collapse_features_advanced)
		#self.ContentsFeatures.hide()

	def collapse_features_advanced(self):

		if self.ContentsFeatures.isHidden():
			self.collapse_features_btn.setIcon(icon(MDI6.chevron_down,color="black"))
			self.collapse_features_btn.setIconSize(QSize(20, 20))
			self.button_widget.adjustSize()
			self.adjustSize()
		else:
			self.collapse_features_btn.setIcon(icon(MDI6.chevron_up,color="black"))
			self.collapse_features_btn.setIconSize(QSize(20, 20))
			self.button_widget.adjustSize()
			self.adjustSize()	

	def generate_post_proc_panel_contents(self):

		self.ContentsPostProc = QFrame()
		layout = QVBoxLayout(self.ContentsPostProc)
		layout.setContentsMargins(0,0,0,0)

		post_proc_layout  = QHBoxLayout()
		self.post_proc_lbl = QLabel("Post processing on the tracks:")
		post_proc_layout.addWidget(self.post_proc_lbl, 90)
		layout.addLayout(post_proc_layout)

		clean_traj_sublayout = QVBoxLayout()
		clean_traj_sublayout.setContentsMargins(15,15,15,15)

		tracklength_layout = QHBoxLayout()
		self.min_tracklength_slider = QLabeledSlider()
		self.min_tracklength_slider.setSingleStep(1)
		self.min_tracklength_slider.setTickInterval(1)
		self.min_tracklength_slider.setSingleStep(1)
		self.min_tracklength_slider.setOrientation(1)
		self.min_tracklength_slider.setRange(0,self.parent.parent.len_movie)
		self.min_tracklength_slider.setValue(0)
		tracklength_layout.addWidget(QLabel('Min. tracklength: '),40)
		tracklength_layout.addWidget(self.min_tracklength_slider, 60)
		clean_traj_sublayout.addLayout(tracklength_layout)

		self.remove_not_in_first_checkbox = QCheckBox('Remove tracks that do not start at the beginning')
		self.remove_not_in_first_checkbox.setIcon(icon(MDI6.arrow_expand_right,color="k"))
		clean_traj_sublayout.addWidget(self.remove_not_in_first_checkbox)

		self.remove_not_in_last_checkbox = QCheckBox('Remove tracks that do not end at the end')
		self.remove_not_in_last_checkbox.setIcon(icon(MDI6.arrow_expand_left,color="k"))
		clean_traj_sublayout.addWidget(self.remove_not_in_last_checkbox)

		self.interpolate_gaps_checkbox = QCheckBox('Interpolate missed detections within tracks')
		self.interpolate_gaps_checkbox.setIcon(icon(MDI6.chart_timeline_variant_shimmer,color="k"))
		clean_traj_sublayout.addWidget(self.interpolate_gaps_checkbox)

		self.extrapolate_post_checkbox = QCheckBox('Sustain last position until the end of the movie')
		self.extrapolate_post_checkbox.setIcon(icon(MDI6.repeat,color="k"))

		self.extrapolate_pre_checkbox = QCheckBox('Sustain first position from the beginning of the movie')
		self.extrapolate_pre_checkbox.setIcon(icon(MDI6.repeat,color="k"))

		clean_traj_sublayout.addWidget(self.extrapolate_post_checkbox)
		clean_traj_sublayout.addWidget(self.extrapolate_pre_checkbox)

		self.interpolate_na_features_checkbox = QCheckBox('Interpolate features of missed detections')
		self.interpolate_na_features_checkbox.setIcon(icon(MDI6.format_color_fill,color="k"))

		clean_traj_sublayout.addWidget(self.interpolate_na_features_checkbox)
		clean_traj_sublayout.addStretch()

		self.post_proc_options_to_disable = [self.post_proc_lbl, self.min_tracklength_slider, self.remove_not_in_first_checkbox,
											self.remove_not_in_last_checkbox, self.interpolate_gaps_checkbox, self.extrapolate_post_checkbox,
											self.extrapolate_pre_checkbox, self.interpolate_na_features_checkbox]

		layout.addLayout(clean_traj_sublayout)


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

		self.use_channel_lbl = QLabel('Use channel:')
		layout.addWidget(self.use_channel_lbl)
		self.mask_channels_cb = [QCheckBox() for i in range(len(self.channels))]
		for cb,chn in zip(self.mask_channels_cb,self.channel_names):
			cb.setText(chn)
			cb.setChecked(True)
			layout.addWidget(cb)

		self.feat_sep1 = QHSeperationLine()
		layout.addWidget(self.feat_sep1)

		self.activate_haralick_btn = QCheckBox('activate Haralick texture features')
		self.activate_haralick_btn.toggled.connect(self.show_haralick_options)
		# Haralick features parameters

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
		self.haralick_percentile_min_lbl = QLabel('Min percentile: ')
		self.haralick_percentile_max_lbl = QLabel('Max percentile: ')

		self.haralick_hist_btn = QPushButton()
		#self.haralick_hist_btn.clicked.connect(self.control_haralick_histogram)
		self.haralick_hist_btn.setIcon(icon(MDI6.poll,color="k"))
		self.haralick_hist_btn.setStyleSheet(self.parent.parent.parent.button_select_all)

		self.haralick_digit_btn = QPushButton()
		#self.haralick_digit_btn.clicked.connect(self.control_haralick_digitalization)
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
		slider_min_percentile_layout.addWidget(self.haralick_percentile_min_le,60)
		self.haralick_layout.addLayout(slider_min_percentile_layout)

		slider_max_percentile_layout = QHBoxLayout()
		slider_max_percentile_layout.addWidget(self.haralick_percentile_max_lbl,40)
		slider_max_percentile_layout.addWidget(self.haralick_percentile_max_le,60)
		self.haralick_layout.addLayout(slider_max_percentile_layout)

		self.haralick_to_hide = [self.haralick_hist_btn, self.haralick_digit_btn, self.haralick_channel_lbl, self.haralick_channel_choice,
								self.haralick_distance_le, self.haralick_distance_lbl, self.haralick_n_gray_levels_le, self.haralick_n_gray_levels_lbl,
								self.haralick_scale_lbl, self.haralick_scale_slider, self.haralick_percentile_min_lbl, self.haralick_percentile_min_le,
								self.haralick_percentile_max_lbl, self.haralick_percentile_max_le]
		for element in self.haralick_to_hide:
			element.hide()

		self.features_to_disable = [self.feature_lbl, self.del_feature_btn, self.add_feature_btn, self.features_list, 
									self.use_channel_lbl, *self.mask_channels_cb, self.activate_haralick_btn]

		layout.addLayout(self.haralick_layout)

	def populate_config_frame(self):

		grid = QGridLayout(self.config_frame)
		panel_title = QLabel(f"CONFIGURATION")
		panel_title.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")

		grid.addWidget(panel_title, 0, 0, 1, 4, alignment=Qt.AlignCenter)

		self.collapse_config_btn = QPushButton()
		self.collapse_config_btn.setIcon(icon(MDI6.chevron_down,color="black"))
		self.collapse_config_btn.setIconSize(QSize(20, 20))
		self.collapse_config_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		grid.addWidget(self.collapse_config_btn, 0, 0, 1, 4, alignment=Qt.AlignRight)

		self.generate_config_panel_contents()
		grid.addWidget(self.ContentsConfig, 1, 0, 1, 4, alignment=Qt.AlignTop)
		self.collapse_config_btn.clicked.connect(lambda: self.ContentsConfig.setHidden(not self.ContentsConfig.isHidden()))
		self.collapse_config_btn.clicked.connect(self.collapse_config_advanced)
		#self.ContentsConfig.hide()

	def collapse_config_advanced(self):

		if self.ContentsConfig.isHidden():
			self.collapse_config_btn.setIcon(icon(MDI6.chevron_down,color="black"))
			self.collapse_config_btn.setIconSize(QSize(20, 20))
			self.button_widget.adjustSize()
			self.adjustSize()
		else:
			self.collapse_config_btn.setIcon(icon(MDI6.chevron_up,color="black"))
			self.collapse_config_btn.setIconSize(QSize(20, 20))
			self.button_widget.adjustSize()
			self.adjustSize()	

	def generate_config_panel_contents(self):
		
		self.ContentsConfig = QFrame()
		layout = QVBoxLayout(self.ContentsConfig)
		layout.setContentsMargins(0,0,0,0)

		btrack_config_layout = QHBoxLayout()
		self.config_lbl = QLabel("bTrack configuration: ")
		btrack_config_layout.addWidget(self.config_lbl, 90)

		self.upload_btrack_config_btn = QPushButton()
		self.upload_btrack_config_btn.setIcon(icon(MDI6.plus,color="black"))
		self.upload_btrack_config_btn.setIconSize(QSize(20, 20))
		self.upload_btrack_config_btn.setToolTip("Upload a new bTrack configuration.")
		self.upload_btrack_config_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.upload_btrack_config_btn.clicked.connect(self.upload_btrack_config)
		btrack_config_layout.addWidget(self.upload_btrack_config_btn, 5)  #4,3,1,1, alignment=Qt.AlignLeft

		self.reset_config_btn = QPushButton()
		self.reset_config_btn.setIcon(icon(MDI6.arrow_u_right_top,color="black"))
		self.reset_config_btn.setIconSize(QSize(20, 20))
		self.reset_config_btn.setToolTip("Reset the configuration to the default bTrack config.")
		self.reset_config_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.reset_config_btn.clicked.connect(self.reset_btrack_config)
		btrack_config_layout.addWidget(self.reset_config_btn, 5)  #4,3,1,1, alignment=Qt.AlignLeft

		layout.addLayout(btrack_config_layout)

		self.config_le = QTextEdit()
		self.config_le.setMinimumHeight(150)
		#self.config_le.setStyleSheet("""
		#							background: #EEEDEB;
		#							border: 2px solid black;
		#							""")
		layout.addWidget(self.config_le)
		self.load_cell_config()


	def show_haralick_options(self):

		"""
		Show the Haralick texture options.
		"""

		if self.activate_haralick_btn.isChecked():
			for element in self.haralick_to_hide:
				element.show()
		else:
			for element in self.haralick_to_hide:
				element.hide()   		

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

		print('Writing instructions...')
		tracking_options = {'btrack_config_path': self.config_path}
		if not self.features_ticked:
			features = None
			masked_channels = None
		else:
			features = self.features_list.getItems()
			masked_channels = self.channel_names[np.array([not cb.isChecked() for cb in self.mask_channels_cb])]
			if len(masked_channels)==0:
				masked_channels = None
			else:
				masked_channels = list(masked_channels)

		tracking_options.update({'features': features, 'mask_channels': masked_channels})
		
		if self.activate_haralick_btn.isChecked():
			haralick_options = {"target_channel": self.haralick_channel_choice.currentText(),
								"scale_factor": float(self.haralick_scale_slider.value()),
								"percentiles": (float(self.haralick_percentile_min_le.text()), float(self.haralick_percentile_max_le.text())),
								"n_intensity_bins": int(self.haralick_n_gray_levels_le.text()),
								}
		else:
			haralick_options = None
		tracking_options.update({'haralick_options': haralick_options})

		if self.post_proc_ticked:
			post_processing_options = {"minimum_tracklength": int(self.min_tracklength_slider.value()),
									   "remove_not_in_first": self.remove_not_in_first_checkbox.isChecked(), 
									   "remove_not_in_last": self.remove_not_in_last_checkbox.isChecked(),
									   "interpolate_position_gap": self.interpolate_gaps_checkbox.isChecked(), 
									   "extrapolate_pre": self.extrapolate_pre_checkbox.isChecked(),
									   "extrapolate_post": self.extrapolate_post_checkbox.isChecked(),
									   'interpolate_na': self.interpolate_na_features_checkbox.isChecked()
									   }
		else:

			post_processing_options = None

		tracking_options.update({'post_processing_options': post_processing_options})

		print(tracking_options)

			



