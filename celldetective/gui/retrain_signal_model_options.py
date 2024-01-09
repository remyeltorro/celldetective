from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QScrollArea, QComboBox, QFrame, QCheckBox, QFileDialog, QGridLayout, QTextEdit, QLineEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QIcon
from celldetective.gui.gui_utils import center_window, FeatureChoice, ListWidget, QHSeperationLine, FigureCanvas, GeometryChoice, OperationChoice
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider,QLabeledSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from celldetective.utils import extract_experiment_channels, get_software_location
from celldetective.io import interpret_tracking_configuration, load_frames
from celldetective.measure import compute_haralick_features, contour_of_instance_segmentation
from celldetective.signals import train_signal_model
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
import pandas as pd
from functools import partial

class ConfigSignalModelTraining(QMainWindow):
	
	"""
	UI to set measurement instructions.

	"""

	def __init__(self, parent=None):
		
		super().__init__()
		self.parent = parent
		self.setWindowTitle("Train signal model")
		self.setWindowIcon(QIcon(os.sep.join(['celldetective','icons','mexican-hat.png'])))
		self.mode = self.parent.mode
		self.exp_dir = self.parent.exp_dir
		self.soft_path = get_software_location()
		self.pretrained_model = None 
		self.dataset_folder = None
		self.signal_models_dir = self.soft_path+'/celldetective/models/signal_detection/'

		self.onlyFloat = QDoubleValidator()
		self.onlyInt = QIntValidator()

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
		self.submit_btn.clicked.connect(self.prep_model)
		main_layout.addWidget(self.submit_btn)
		self.submit_btn.setEnabled(False)

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
		self.lr_le = QLineEdit('0,01')
		self.lr_le.setValidator(self.onlyFloat)
		lr_layout.addWidget(self.lr_le, 70)
		layout.addLayout(lr_layout)

		bs_layout = QHBoxLayout()
		bs_layout.addWidget(QLabel('batch size: '),30)
		self.bs_le = QLineEdit('64')
		self.bs_le.setValidator(self.onlyInt)
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

		classname_layout = QHBoxLayout()
		classname_layout.addWidget(QLabel('class name: '), 30)
		self.class_name_le = QLineEdit()
		self.class_name_le.setText("")
		classname_layout.addWidget(self.class_name_le, 70)
		layout.addLayout(classname_layout)

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

		#self.channel_cbs = [QComboBox() for i in range(4)]

		self.max_nbr_channels = 5
		self.channel_cbs = [QComboBox() for i in range(self.max_nbr_channels)]
		self.normalization_mode_btns = [QPushButton('') for i in range(self.max_nbr_channels)]
		self.normalization_mode = [True for i in range(self.max_nbr_channels)]

		self.normalization_clip_btns = [QPushButton('') for i in range(self.max_nbr_channels)]
		self.clip_option = [False for i in range(self.max_nbr_channels)]

		for i in range(self.max_nbr_channels):

			self.normalization_mode_btns[i].setIcon(icon(MDI6.percent_circle,color="#1565c0"))
			self.normalization_mode_btns[i].setIconSize(QSize(20, 20))	
			self.normalization_mode_btns[i].setStyleSheet(self.parent.parent.parent.button_select_all)	
			self.normalization_mode_btns[i].setToolTip("Switch to absolute normalization values.")
			self.normalization_mode_btns[i].clicked.connect(partial(self.switch_normalization_mode, i))

			self.normalization_clip_btns[i].setIcon(icon(MDI6.content_cut,color="black"))
			self.normalization_clip_btns[i].setIconSize(QSize(20, 20))	
			self.normalization_clip_btns[i].setStyleSheet(self.parent.parent.parent.button_select_all)	
			self.normalization_clip_btns[i].clicked.connect(partial(self.switch_clipping_mode, i))
			self.normalization_clip_btns[i].setToolTip('clip')

		self.normalization_min_value_lbl = [QLabel('Min %: ') for i in range(self.max_nbr_channels)]
		self.normalization_min_value_le = [QLineEdit('0.1') for i in range(self.max_nbr_channels)]

		self.normalization_max_value_lbl = [QLabel('Max %: ') for i in range(self.max_nbr_channels)]		
		self.normalization_max_value_le = [QLineEdit('99.99') for i in range(self.max_nbr_channels)]

		tables = glob(self.exp_dir+os.sep.join(['W*','*','output','tables',f'trajectories_{self.mode}.csv']))
		print(tables)
		all_measurements = []
		for tab in tables:
			cols = pd.read_csv(tab, nrows=1).columns.tolist()
			all_measurements.extend(cols)
		all_measurements = np.unique(all_measurements)
		generic_measurements = ['brightfield_channel', 'live_nuclei_channel', 'dead_nuclei_channel', 
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
							"maximal_correlation_coefficient",
							"POSITION_X",
							"POSITION_Y",
							]

		self.channel_items = np.unique(generic_measurements + list(all_measurements))
		self.channel_items = np.insert(self.channel_items, 0, '--')

		self.channel_option_layouts = []
		for i in range(len(self.channel_cbs)):
			ch_layout = QHBoxLayout()
			ch_layout.addWidget(QLabel(f'channel {i}: '), 30)
			self.channel_cbs[i].addItems(self.channel_items)
			self.channel_cbs[i].currentIndexChanged.connect(self.check_valid_channels)
			ch_layout.addWidget(self.channel_cbs[i], 70)
			layout.addLayout(ch_layout)

			channel_norm_options_layout = QHBoxLayout()
			channel_norm_options_layout.setContentsMargins(130,0,0,0)
			channel_norm_options_layout.addWidget(self.normalization_min_value_lbl[i])			
			channel_norm_options_layout.addWidget(self.normalization_min_value_le[i])
			channel_norm_options_layout.addWidget(self.normalization_max_value_lbl[i])
			channel_norm_options_layout.addWidget(self.normalization_max_value_le[i])
			channel_norm_options_layout.addWidget(self.normalization_clip_btns[i])
			channel_norm_options_layout.addWidget(self.normalization_mode_btns[i])
			layout.addLayout(channel_norm_options_layout)

		# for i in range(len(self.channel_cbs)):
		# 	ch_layout = QHBoxLayout()
		# 	ch_layout.addWidget(QLabel(f'channel {i}: '), 30)
		# 	self.channel_cbs[i].addItems(self.channel_items)
		# 	self.channel_cbs[i].currentIndexChanged.connect(self.check_valid_channels)
		# 	ch_layout.addWidget(self.channel_cbs[i], 70)
		# 	layout.addLayout(ch_layout)

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
						os.sep.join([self.soft_path,'celldetective','models','signal_detection','']),
						QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
						)

		if self.pretrained_model is not None:
		# 	self.foldername = self.file_dialog_pretrained.selectedFiles()[0]
			subfiles = glob(os.sep.join([self.pretrained_model,"*"]))
			if os.sep.join([self.pretrained_model,"config_input.json"]) in subfiles:
				self.load_pretrained_config()
				self.pretrained_lbl.setText(self.pretrained_model.split(os.sep)[-1])
				self.cancel_pretrained.setVisible(True)
				self.recompile_option.setEnabled(True)
				self.modelname_le.setText(f"{self.pretrained_model.split(os.sep)[-1]}_{datetime.today().strftime('%Y-%m-%d')}")
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

			subfiles = glob(os.sep.join([self.dataset_folder,"*.npy"]))
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
		self.class_name_le.setText('')
		self.modelname_le.setText(f"Untitled_model_{datetime.today().strftime('%Y-%m-%d')}")

	def clear_dataset(self):

		self.dataset_folder = None
		self.data_folder_label.setText('No folder chosen')
		self.data_folder_label.setToolTip('')
		self.cancel_dataset.setVisible(False)


	def load_pretrained_config(self):

		f = open(os.sep.join([self.pretrained_model,"config_input.json"]))
		data = json.load(f)
		channels = data["channels"]
		signal_length = data["model_signal_length"]
		try:
			label = data['label']
			self.class_name_le.setText(label)
		except:
			pass
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

	def prep_model(self):

		model_name = self.modelname_le.text()
		pretrained_model = self.pretrained_model
		signal_length = self.model_length_slider.value()
		recompile_op = self.recompile_option.isChecked()

		channels = []
		for i in range(len(self.channel_cbs)):
			channels.append(self.channel_cbs[i].currentText())

		slots_to_keep = np.where(np.array(channels)!='--')[0]
		while '--' in channels:
			channels.remove('--')

		norm_values = np.array([[float(a.replace(',','.')),float(b.replace(',','.'))] for a,b in zip([l.text() for l in self.normalization_min_value_le],
											[l.text() for l in self.normalization_max_value_le])])
		norm_values = norm_values[slots_to_keep]
		norm_values = [list(v) for v in norm_values]

		clip_values = np.array(self.clip_option)
		clip_values = list(clip_values[slots_to_keep])
		clip_values = [bool(c) for c in clip_values]

		normalization_mode = np.array(self.normalization_mode)
		normalization_mode = list(normalization_mode[slots_to_keep])
		normalization_mode = [bool(m) for m in normalization_mode]

		data_folders = []
		if self.dataset_folder is not None:
			data_folders.append(self.dataset_folder)
		if self.dataset_cb.currentText()!='--':
			previous_dataset = glob(self.soft_path+os.sep.join(['celldetective','datasets','signals','*'+os.sep]))[self.dataset_cb.currentIndex()-1]
			data_folders.append(previous_dataset)

		aug_factor = self.augmentation_slider.value()
		val_split = self.validation_slider.value()

		try:
			lr = float(self.lr_le.text().replace(',','.'))
		except:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("Invalid value encountered for the learning rate.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None			
		
		bs = int(self.bs_le.text())
		epochs = self.epochs_slider.value()

		training_instructions = {'model_name': model_name,'pretrained': pretrained_model, 'channel_option': channels, 'normalization_percentile': normalization_mode,
		'normalization_clip': clip_values,'normalization_values': norm_values, 'model_signal_length': signal_length,
		'recompile_pretrained': recompile_op, 'ds': data_folders, 'augmentation_factor': aug_factor, 'validation_split': val_split,
		'learning_rate': lr, 'batch_size': bs, 'epochs': epochs, 'label': self.class_name_le.text()}

		model_folder = self.signal_models_dir + model_name + os.sep
		if not os.path.exists(model_folder):
			os.mkdir(model_folder)

		training_instructions.update({'target_directory': self.signal_models_dir})

		print(f"Set of instructions: {training_instructions}")
		with open(model_folder+"training_instructions.json", 'w') as f:
			json.dump(training_instructions, f, indent=4)
		
		train_signal_model(model_folder+"training_instructions.json")

		self.parent.refresh_signal_models()


	def check_valid_channels(self):

		if np.all([cb.currentText()=='--' for cb in self.channel_cbs]):
			self.submit_btn.setEnabled(False)
		else:
			self.submit_btn.setEnabled(True)


	def switch_normalization_mode(self, index):

		"""
		Use absolute or percentile values for the normalization of each individual channel.
		
		"""

		currentNormMode = self.normalization_mode[index]
		self.normalization_mode[index] = not currentNormMode

		if self.normalization_mode[index]:
			self.normalization_mode_btns[index].setIcon(icon(MDI6.percent_circle,color="#1565c0"))
			self.normalization_mode_btns[index].setIconSize(QSize(20, 20))	
			self.normalization_mode_btns[index].setStyleSheet(self.parent.parent.parent.button_select_all)	
			self.normalization_mode_btns[index].setToolTip("Switch to absolute normalization values.")
			self.normalization_min_value_lbl[index].setText('Min %: ')
			self.normalization_max_value_lbl[index].setText('Max %: ')
			self.normalization_min_value_le[index].setText('0.1')
			self.normalization_max_value_le[index].setText('99.99')

		else:
			self.normalization_mode_btns[index].setIcon(icon(MDI6.percent_circle_outline,color="black"))
			self.normalization_mode_btns[index].setIconSize(QSize(20, 20))	
			self.normalization_mode_btns[index].setStyleSheet(self.parent.parent.parent.button_select_all)	
			self.normalization_mode_btns[index].setToolTip("Switch to percentile normalization values.")
			self.normalization_min_value_lbl[index].setText('Min: ')
			self.normalization_min_value_le[index].setText('0')
			self.normalization_max_value_lbl[index].setText('Max: ')
			self.normalization_max_value_le[index].setText('1000')

	def switch_clipping_mode(self, index):

		currentClipMode = self.clip_option[index]
		self.clip_option[index] = not currentClipMode

		if self.clip_option[index]:
			self.normalization_clip_btns[index].setIcon(icon(MDI6.content_cut,color="#1565c0"))
			self.normalization_clip_btns[index].setIconSize(QSize(20, 20))
			self.normalization_clip_btns[index].setStyleSheet(self.parent.parent.parent.button_select_all)	

		else:
			self.normalization_clip_btns[index].setIcon(icon(MDI6.content_cut,color="black"))
			self.normalization_clip_btns[index].setIconSize(QSize(20, 20))		
			self.normalization_clip_btns[index].setStyleSheet(self.parent.parent.parent.button_select_all)	

