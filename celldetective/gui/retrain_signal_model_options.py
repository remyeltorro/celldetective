from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QScrollArea, QComboBox, QFrame, QCheckBox, QFileDialog, QGridLayout, QTextEdit, QLineEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize
from celldetective.gui.gui_utils import center_window, FeatureChoice, ListWidget, QHSeperationLine, FigureCanvas, GeometryChoice, OperationChoice
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider,QLabeledSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from celldetective.utils import extract_experiment_channels, get_software_location
from celldetective.io import interpret_tracking_configuration, load_frames
from celldetective.measure import compute_haralick_features, contour_of_instance_segmentation
import numpy as np
import json
from shutil import copyfile
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
from natsort import natsorted
from tifffile import imread
from pathlib import Path, PurePath
from datetime import datetime

class ConfigSignalModelTraining(QMainWindow):
	
	"""
	UI to set measurement instructions.

	"""

	def __init__(self, parent=None):
		
		super().__init__()
		self.parent = parent
		self.setWindowTitle("Train signal model")
		self.mode = self.parent.mode
		self.exp_dir = self.parent.exp_dir
		self.soft_path = get_software_location()

		self.screen_height = self.parent.parent.parent.screen_height
		center_window(self)

		self.setMinimumWidth(500)
		self.setMinimumHeight(int(0.3*self.screen_height))
		self.setMaximumHeight(int(0.8*self.screen_height))
		self.populate_widget()
		#self.load_previous_measurement_instructions()

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
		self.model_frame = QFrame()
		self.model_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_model_frame()
		main_layout.addWidget(self.model_frame)

		self.data_frame = QFrame()
		self.data_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_data_frame()
		main_layout.addWidget(self.data_frame)

		self.hyper_frame = QFrame()
		self.hyper_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_hyper_frame()
		main_layout.addWidget(self.hyper_frame)

		self.submit_btn = QPushButton('Train')
		self.submit_btn.setStyleSheet(self.parent.parent.parent.button_style_sheet)
		#self.submit_btn.clicked.connect(self.write_instructions)
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

	def populate_hyper_frame(self):

		"""
		Add widgets and layout in the POST-PROCESSING frame.
		"""

		grid = QGridLayout(self.hyper_frame)
		grid.setContentsMargins(30,30,30,30)
		grid.setSpacing(30)

		self.hyper_lbl = QLabel("HYPERPARAMETERS")
		self.hyper_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.hyper_lbl, 0, 0, 1, 4, alignment=Qt.AlignCenter)
		self.generate_hyper_contents()
		grid.addWidget(self.ContentsHyper, 1, 0, 1, 4, alignment=Qt.AlignTop)

	def generate_hyper_contents(self):

		self.ContentsHyper = QFrame()
		layout = QVBoxLayout(self.ContentsHyper)
		layout.setContentsMargins(0,0,0,0)

		lr_layout = QHBoxLayout()
		lr_layout.addWidget(QLabel('learning rate: '),30)
		self.lr_le = QLineEdit('0.001')
		lr_layout.addWidget(self.lr_le, 70)
		layout.addLayout(lr_layout)

		bs_layout = QHBoxLayout()
		bs_layout.addWidget(QLabel('batch size: '),30)
		self.bs_le = QLineEdit('4')
		bs_layout.addWidget(self.bs_le, 70)
		layout.addLayout(bs_layout)

		epochs_layout = QHBoxLayout()
		epochs_layout.addWidget(QLabel('# epochs: '), 30)
		self.epochs_slider = QLabeledSlider()
		self.epochs_slider.setRange(1,3000)
		self.epochs_slider.setSingleStep(1)
		self.epochs_slider.setTickInterval(1)		
		self.epochs_slider.setOrientation(1)
		self.epochs_slider.setValue(300)
		epochs_layout.addWidget(self.epochs_slider, 70)
		layout.addLayout(epochs_layout)



	def populate_data_frame(self):

		"""
		Add widgets and layout in the POST-PROCESSING frame.
		"""

		grid = QGridLayout(self.data_frame)
		grid.setContentsMargins(30,30,30,30)
		grid.setSpacing(30)

		self.data_lbl = QLabel("DATA")
		self.data_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.data_lbl, 0, 0, 1, 4, alignment=Qt.AlignCenter)
		self.generate_data_contents()
		grid.addWidget(self.ContentsData, 1, 0, 1, 4, alignment=Qt.AlignTop)

	def populate_model_frame(self):

		"""
		Add widgets and layout in the FEATURES frame.
		"""

		grid = QGridLayout(self.model_frame)
		grid.setContentsMargins(30,30,30,30)
		grid.setSpacing(30)

		self.model_lbl = QLabel("MODEL")
		self.model_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.model_lbl, 0, 0, 1, 4, alignment=Qt.AlignCenter)

		self.generate_model_panel_contents()
		grid.addWidget(self.ContentsModel, 1, 0, 1, 4, alignment=Qt.AlignTop)


	def generate_data_contents(self):

		self.ContentsData = QFrame()
		layout = QVBoxLayout(self.ContentsData)
		layout.setContentsMargins(0,0,0,0)

		train_data_layout = QHBoxLayout()
		train_data_layout.addWidget(QLabel('Training data: '), 30)
		self.select_data_folder_btn = QPushButton('Choose folder')
		self.select_data_folder_btn.clicked.connect(self.showDialog_dataset)
		self.data_folder_label = QLabel('No folder chosen')
		train_data_layout.addWidget(self.select_data_folder_btn, 35)
		train_data_layout.addWidget(self.data_folder_label, 30)

		self.cancel_dataset = QPushButton()
		self.cancel_dataset.setIcon(icon(MDI6.close,color="black"))
		self.cancel_dataset.clicked.connect(self.clear_dataset)
		self.cancel_dataset.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.cancel_dataset.setIconSize(QSize(20, 20))
		self.cancel_dataset.setVisible(False)
		train_data_layout.addWidget(self.cancel_dataset, 5)


		layout.addLayout(train_data_layout)

		include_dataset_layout = QHBoxLayout()
		include_dataset_layout.addWidget(QLabel('include dataset: '),30)
		self.dataset_cb = QComboBox()
		available_datasets = glob(self.soft_path+'/celldetective/datasets/signals/*/')
		signal_datasets = ['--'] + [d.split('/')[-2] for d in available_datasets]
		self.dataset_cb.addItems(signal_datasets)
		include_dataset_layout.addWidget(self.dataset_cb, 70)
		layout.addLayout(include_dataset_layout)

		augmentation_hbox = QHBoxLayout()
		augmentation_hbox.addWidget(QLabel('augmentation\nfactor: '), 30)
		self.augmentation_slider = QLabeledDoubleSlider()
		self.augmentation_slider.setSingleStep(0.01)
		self.augmentation_slider.setTickInterval(0.01)		
		self.augmentation_slider.setOrientation(1)
		self.augmentation_slider.setRange(1, 5)
		self.augmentation_slider.setValue(2)

		augmentation_hbox.addWidget(self.augmentation_slider, 70)
		layout.addLayout(augmentation_hbox)

		validation_split_layout = QHBoxLayout()
		validation_split_layout.addWidget(QLabel('validation split: '),30)
		self.validation_slider = QLabeledDoubleSlider()
		self.validation_slider.setSingleStep(0.01)
		self.validation_slider.setTickInterval(0.01)		
		self.validation_slider.setOrientation(1)
		self.validation_slider.setRange(0,0.9)
		self.validation_slider.setValue(0.25)		
		validation_split_layout.addWidget(self.validation_slider, 70)
		layout.addLayout(validation_split_layout)


	def generate_model_panel_contents(self):
		
		self.ContentsModel = QFrame()
		layout = QVBoxLayout(self.ContentsModel)
		layout.setContentsMargins(0,0,0,0)

		modelname_layout = QHBoxLayout()
		modelname_layout.addWidget(QLabel('Model name: '), 30)
		self.modelname_le = QLineEdit()
		self.modelname_le.setText(f"Untitled_model_{datetime.today().strftime('%Y-%m-%d')}")
		modelname_layout.addWidget(self.modelname_le, 70)
		layout.addLayout(modelname_layout)

		pretrained_layout = QHBoxLayout()
		pretrained_layout.setContentsMargins(0,0,0,0)
		pretrained_layout.addWidget(QLabel('Pretrained model: '), 30)

		self.browse_pretrained_btn = QPushButton('Choose folder')
		self.browse_pretrained_btn.clicked.connect(self.showDialog_pretrained)
		pretrained_layout.addWidget(self.browse_pretrained_btn, 35)

		self.pretrained_lbl = QLabel('No folder chosen')
		pretrained_layout.addWidget(self.pretrained_lbl, 30)

		self.cancel_pretrained = QPushButton()
		self.cancel_pretrained.setIcon(icon(MDI6.close,color="black"))
		self.cancel_pretrained.clicked.connect(self.clear_pretrained)
		self.cancel_pretrained.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.cancel_pretrained.setIconSize(QSize(20, 20))
		self.cancel_pretrained.setVisible(False)
		pretrained_layout.addWidget(self.cancel_pretrained, 5)

		layout.addLayout(pretrained_layout)

		recompile_layout = QHBoxLayout()
		recompile_layout.addWidget(QLabel('Recompile: '), 30)
		self.recompile_option = QCheckBox()
		self.recompile_option.setEnabled(False)
		recompile_layout.addWidget(self.recompile_option, 70)
		layout.addLayout(recompile_layout)

		self.channel_cbs = [QComboBox() for i in range(4)]

		self.channel_items = ['--', 'brightfield_channel', 'live_nuclei_channel', 'dead_nuclei_channel', 
							 'effector_fluo_channel', 'adhesion_channel', 'fluo_channel_1', 'fluo_channel_2',
							 "area", "area_bbox","area_convex","area_filled","major_axis_length", 
							 "minor_axis_length", 
							"eccentricity",
							"equivalent_diameter_area",
							"euler_number",
							"extent",
							"feret_diameter_max",
							"orientation", 
							"perimeter",
							"perimeter_crofton",
							"solidity",
							"angular_second_moment",
							"contrast",
							"correlation",
							"sum_of_square_variance",
							"inverse_difference_moment",
							"sum_average",
							"sum_variance",
							"sum_entropy",
							"entropy",
							"difference_variance",
							"difference_entropy",
							"information_measure_of_correlation_1",
							"information_measure_of_correlation_2",
							"maximal_correlation_coefficient"
							]

		for i in range(len(self.channel_cbs)):
			ch_layout = QHBoxLayout()
			ch_layout.addWidget(QLabel(f'channel {i}: '), 30)
			self.channel_cbs[i].addItems(self.channel_items)
			ch_layout.addWidget(self.channel_cbs[i], 70)
			layout.addLayout(ch_layout)

		model_length_layout = QHBoxLayout()
		model_length_layout.addWidget(QLabel('Max signal length: '), 30)
		self.model_length_slider = QLabeledSlider()
		self.model_length_slider.setSingleStep(1)
		self.model_length_slider.setTickInterval(1)
		self.model_length_slider.setSingleStep(1)
		self.model_length_slider.setOrientation(1)
		self.model_length_slider.setRange(0,1024)
		self.model_length_slider.setValue(128)		
		model_length_layout.addWidget(self.model_length_slider, 70)
		layout.addLayout(model_length_layout)

	def showDialog_pretrained(self):

		self.pretrained_model = QFileDialog.getExistingDirectory(
						self, "Open Directory",
						self.soft_path+'/celldetective/models/signal_detection/',
						QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
						)

		if self.pretrained_model is not None:
		# 	self.foldername = self.file_dialog_pretrained.selectedFiles()[0]
			subfiles = glob(self.pretrained_model+"/*")
			if self.pretrained_model+"/config_input.json" in subfiles:
				self.load_pretrained_config()
				self.pretrained_lbl.setText(self.pretrained_model.split("/")[-1])
				self.cancel_pretrained.setVisible(True)
				self.recompile_option.setEnabled(True)
				self.modelname_le.setText(f"{self.pretrained_model.split('/')[-1]}_{datetime.today().strftime('%Y-%m-%d')}")
			else:
				self.pretrained_model = None
				self.pretrained_lbl.setText('No folder chosen')	
				self.recompile_option.setEnabled(False)	
				self.cancel_pretrained.setVisible(False)
		print(self.pretrained_model)

	def showDialog_dataset(self):

		self.dataset_folder = QFileDialog.getExistingDirectory(
						self, "Open Directory",
						self.exp_dir,
						QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
						)
		if self.dataset_folder is not None:

			subfiles = glob(self.dataset_folder+"/*.npy")
			if len(subfiles)>0:
				print(f'found {len(subfiles)} files in folder')
				self.data_folder_label.setText(self.dataset_folder[:16]+'...')
				self.data_folder_label.setToolTip(self.dataset_folder)
				self.cancel_dataset.setVisible(True)
			else:
				self.data_folder_label.setText('No folder chosen')
				self.data_folder_label.setToolTip('')
				self.dataset_folder = None
				self.cancel_dataset.setVisible(False)

	def clear_pretrained(self):
		
		self.pretrained_model = None
		self.pretrained_lbl.setText('No folder chosen')
		for cb in self.channel_cbs:
			cb.setEnabled(True)
		self.recompile_option.setEnabled(False)
		self.cancel_pretrained.setVisible(False)
		self.model_length_slider.setEnabled(True)
		self.modelname_le.setText(f"Untitled_model_{datetime.today().strftime('%Y-%m-%d')}")

	def clear_dataset(self):

		self.dataset_folder = None
		self.data_folder_label.setText('No folder chosen')
		self.data_folder_label.setToolTip('')
		self.cancel_dataset.setVisible(False)


	def load_pretrained_config(self):

		f = open(self.pretrained_model+"/config_input.json")
		data = json.load(f)
		channels = data["channels"]
		signal_length = data["model_signal_length"]
		self.model_length_slider.setValue(int(signal_length))
		self.model_length_slider.setEnabled(False)

		for c,cb in zip(channels, self.channel_cbs):
			index = cb.findText(c)
			cb.setCurrentIndex(index)

		if len(channels)<len(self.channel_cbs):
			for k in range(len(self.channel_cbs)-len(channels)):
				self.channel_cbs[len(channels)+k].setCurrentIndex(0)
				self.channel_cbs[len(channels)+k].setEnabled(False)


	def adjustScrollArea(self):
		
		"""
		Auto-adjust scroll area to fill space 
		(from https://stackoverflow.com/questions/66417576/make-qscrollarea-use-all-available-space-of-qmainwindow-height-axis)
		"""

		step = 5
		while self.scroll_area.verticalScrollBar().isVisible() and self.height() < self.maximumHeight():
			self.resize(self.width(), self.height() + step)

