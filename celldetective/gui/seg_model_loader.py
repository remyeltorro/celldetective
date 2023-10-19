from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QCheckBox, QLineEdit, QHBoxLayout, QRadioButton, QComboBox, QFileDialog, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, QSize
from celldetective.gui.gui_utils import center_window
from celldetective.gui import ThresholdConfigWizard
from PyQt5.QtGui import QDoubleValidator
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import numpy as np
from glob import glob
import os
import json
import shutil

class SegmentationModelLoader(QWidget):
	
	"""
	Upload a segmentation model or define a Threshold pipeline.
	"""

	def __init__(self, parent):
		
		super().__init__()
		self.parent = parent
		self.mode = self.parent.mode
		if self.mode=="targets":
			self.target_folder = "segmentation_targets"
		elif self.mode=="effectors":
			self.target_folder = "segmentation_effectors"
		self.setWindowTitle('Upload model')
		self.generate_content()
		center_window(self)

	def generate_content(self):

		self.layout = QGridLayout(self)
		self.layout.addWidget(QLabel('Select:'), 0, 0, 1, 1)
		self.layout.setSpacing(5)

		option_layout = QHBoxLayout()
		option_layout.setContentsMargins(10,10,10,10)

		# Create radio buttons
		self.stardist_button = QRadioButton('StarDist')
		#self.stardist_button.setIcon(QIcon(abs_path+f"/icons/star.png"))
		self.stardist_button.setChecked(True)
		option_layout.addWidget(self.stardist_button)

		self.cellpose_button = QRadioButton('Cellpose')
		#self.cellpose_button.setIcon(QIcon(abs_path+f"/icons/cellpose.png"))
		option_layout.addWidget(self.cellpose_button)

		self.threshold_button = QRadioButton('Threshold')
		option_layout.addWidget(self.threshold_button)

		self.layout.addLayout(option_layout, 1,0,1,2, alignment=Qt.AlignCenter)
		self.generate_base_block()
		self.layout.addLayout(self.base_block, 2,0,1,2)
		self.generate_stardist_specific_block()
		self.layout.addLayout(self.stardist_block, 3,0,1,2)
		self.combos = [self.combo_ch1, self.combo_ch2, self.combo_ch3, self.combo_ch4]

		self.normalize_checkbox = QCheckBox()
		self.normalize_checkbox.setChecked(True)
		self.normalize_lbl = QLabel('normalize: ')
		self.layout.addWidget(self.normalize_lbl, 8,0,1,1)
		self.layout.addWidget(self.normalize_checkbox,8, 1,1,2, Qt.AlignLeft)

		self.generate_cellpose_options()
		self.layout.addLayout(self.cellpose_block, 3,0,1,2)
		self.generate_threshold_options()

		# Create file dialog
		self.file_dialog = QFileDialog()

		# Create button to open file dialog
		self.open_dialog_button = QPushButton('Choose File')
		self.open_dialog_button.clicked.connect(self.showDialog)
		self.file_label = QLabel('No file chosen', self)
		self.layout.addWidget(self.open_dialog_button, 9, 0, 1, 1)
		self.layout.addWidget(self.file_label, 9, 1, 1, 1)

		self.upload_button = QPushButton("Upload")
		self.upload_button.clicked.connect(self.upload_model)
		self.upload_button.setIcon(icon(MDI6.upload,color="white"))
		self.upload_button.setIconSize(QSize(25, 25))
		self.upload_button.setStyleSheet(self.parent.parent.parent.button_style_sheet)
		self.upload_button.setEnabled(False)
		self.layout.addWidget(self.upload_button, 10, 0, 1, 1)
		
		self.base_block_options = [self.calibration_label, self.spatial_calib_le, self.ch_1_label, self.combo_ch1, self.ch_2_label, self.combo_ch2, 
								   self.normalize_checkbox, self.normalize_lbl,
								   ]

		self.stardist_button.toggled.connect(self.show_seg_options)
		self.cellpose_button.toggled.connect(self.show_seg_options)
		self.threshold_button.toggled.connect(self.show_seg_options)

		for cb in self.combos:
			cb.activated.connect(self.unlock_upload)

		self.setLayout(self.layout)
		self.show()

	def unlock_upload(self):
		if self.stardist_button.isChecked():
			if np.any([c.currentText()!='--' for c in self.combos]):
				self.upload_button.setEnabled(True)
			else:
				self.upload_button.setEnabled(False)
		elif self.cellpose_button.isChecked():
			if np.any([c.currentText()!='--' for c in self.combos[:2]]):
				self.upload_button.setEnabled(True)
			else:
				self.upload_button.setEnabled(False)

	def generate_base_block(self):

		"""
		Create widgets common to StarDist and Cellpose.
		"""
		
		self.base_block = QGridLayout()
		
		self.calibration_label = QLabel("pixel calibration: ")
		self.base_block.addWidget(self.calibration_label,0,0,1,1, alignment=Qt.AlignLeft)
		self.spatial_calib_le = QLineEdit("0,1")
		self.qdv = QDoubleValidator(0.0, np.amax([self.parent.parent.shape_x, self.parent.parent.shape_y]), 8, notation=QDoubleValidator.StandardNotation)
		self.spatial_calib_le.setValidator(self.qdv)
		self.base_block.addWidget(self.spatial_calib_le, 0,1,1,2,alignment=Qt.AlignRight)

		self.channel_options = ["--","live_nuclei_channel", "dead_nuclei_channel", "effector_fluo_channel", "brightfield_channel", "adhesion_channel", "fluo_channel_1", "fluo_channel_2"]
		self.ch_1_label = QLabel("channel 1: ")
		self.base_block.addWidget(self.ch_1_label, 1, 0, 1, 1, alignment=Qt.AlignLeft)
		self.combo_ch1 = QComboBox()
		self.combo_ch1.addItems(self.channel_options)
		self.base_block.addWidget(self.combo_ch1, 1, 1, 1, 2, alignment=Qt.AlignRight)

		self.ch_2_label = QLabel("channel 2: ")
		self.base_block.addWidget(self.ch_2_label, 2, 0, 1, 1, alignment=Qt.AlignLeft)
		self.combo_ch2 = QComboBox()
		self.combo_ch2.addItems(self.channel_options)
		self.base_block.addWidget(self.combo_ch2, 2, 1, 1, 2, alignment=Qt.AlignRight)

	def generate_stardist_specific_block(self):

		"""
		Create StarDist specific fields to use the model properly when calling it from the app.
		"""

		self.stardist_block = QGridLayout()
		self.ch_3_label = QLabel("channel 3: ")
		self.stardist_block.addWidget(self.ch_3_label, 0, 0, 1, 1, alignment=Qt.AlignLeft)
		self.combo_ch3 = QComboBox()
		self.combo_ch3.addItems(self.channel_options)
		self.stardist_block.addWidget(self.combo_ch3, 0, 1, 1, 2, alignment=Qt.AlignRight)

		self.ch_4_label = QLabel("channel 4: ")
		self.stardist_block.addWidget(self.ch_4_label, 1, 0, 1, 1, alignment=Qt.AlignLeft)
		self.combo_ch4 = QComboBox()
		self.combo_ch4.addItems(self.channel_options)
		self.stardist_block.addWidget(self.combo_ch4, 1, 1, 1, 2, alignment=Qt.AlignRight)

		self.stardist_options = [self.ch_3_label, self.ch_4_label, self.combo_ch3, self.combo_ch4]

	def generate_cellpose_options(self):

		"""
		Create Cellpose specific fields to use the model properly when calling it from the app.
		"""

		self.cellpose_block = QGridLayout()
		self.cp_diameter_label = QLabel('diameter:   ')
		self.cp_diameter_le = QLineEdit("30,0")
		self.cp_diameter_le.setValidator(self.qdv)
		self.cellpose_block.addWidget(self.cp_diameter_label, 0, 0, 1, 2)
		self.cellpose_block.addWidget(self.cp_diameter_le, 0, 1, 1, 2)

		qdv_prob = QDoubleValidator(-6, 6, 8, notation=QDoubleValidator.StandardNotation)
		self.cp_cellprob_label = QLabel('cellprob\nthreshold:   ')
		self.cp_cellprob_le = QLineEdit('0,0')
		self.cp_cellprob_le.setValidator(qdv_prob)
		self.cellpose_block.addWidget(self.cp_cellprob_label, 1, 0, 1, 2)
		self.cellpose_block.addWidget(self.cp_cellprob_le, 1, 1, 1, 2)

		self.cp_flow_label = QLabel('flow threshold:   ')
		self.cp_flow_le = QLineEdit('0,4')
		self.cp_flow_le.setValidator(qdv_prob)
		self.cellpose_block.addWidget(self.cp_flow_label, 2, 0, 1, 2)
		self.cellpose_block.addWidget(self.cp_flow_le, 2, 1, 1, 2)

		self.cellpose_options = [self.cp_diameter_label,self.cp_diameter_le,self.cp_cellprob_label,
			self.cp_cellprob_le, self.cp_flow_label, self.cp_flow_le]
		for c in self.cellpose_options:
			c.hide()

	def generate_threshold_options(self):

		"""
		Show the threshold config pipeline button.
		"""

		self.threshold_config_button = QPushButton("Threshold Config Wizard")
		self.threshold_config_button.setIcon(icon(MDI6.auto_fix,color="#1565c0"))
		self.threshold_config_button.setIconSize(QSize(20, 20))
		self.threshold_config_button.setVisible(False)
		self.threshold_config_button.setStyleSheet(self.parent.parent.parent.button_style_sheet_2)
		self.threshold_config_button.clicked.connect(self.open_threshold_config_wizard)
		self.layout.addWidget(self.threshold_config_button,3,0,1,2)
		self.threshold_config_button.hide()

	def showDialog(self):

		"""
		Import the model in the proper folder.
		"""

		# Set the file dialog according to the option chosen
		if self.stardist_button.isChecked():
			self.seg_mode = "stardist"
			self.file_dialog.setFileMode(QFileDialog.Directory)
		elif self.cellpose_button.isChecked():
			self.seg_mode = "cellpose"
			self.file_dialog.setFileMode(QFileDialog.ExistingFile)
		elif self.threshold_button.isChecked():
			self.seg_mode = "threshold"
			self.file_dialog.setFileMode(QFileDialog.ExistingFile)

		# If accepted check validity of data
		if self.file_dialog.exec_() == QFileDialog.Accepted:
			self.filename = self.file_dialog.selectedFiles()[0]
			if self.seg_mode=="stardist":
				subfiles = glob(self.filename+"/*")
				subfiles = [s.replace('\\','/') for s in subfiles]
				if self.filename+"/thresholds.json" in subfiles:
					self.file_label.setText(self.filename.split("/")[-1])
					self.modelname = self.filename.split("/")[-1]
					self.destination = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]+f"/models/{self.target_folder}/"+self.modelname
					self.folder_dest = self.destination
				else:
					msgBox = QMessageBox()
					msgBox.setIcon(QMessageBox.Warning)
					msgBox.setText("StarDist model not recognized... Please ensure that it contains a thresholds.json file or that it is a valid StarDist model...")
					msgBox.setWindowTitle("Warning")
					msgBox.setStandardButtons(QMessageBox.Ok)
					returnValue = msgBox.exec()
					if returnValue == QMessageBox.Ok:
						return None					
			
			if self.seg_mode=="cellpose":
				self.file_label.setText(self.filename.split("/")[-1])
				self.modelname = self.filename.split("/")[-1]
				print(f"Transferring Cellpose model {self.filename}...")
				self.folder_dest = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]+f"/models/{self.target_folder}/"+self.modelname
				self.destination = self.folder_dest+f"/{self.modelname}"

			if self.seg_mode=="threshold":
				self.file_label.setText(self.filename.split("/")[-1])

	def show_seg_options(self):

		"""
		Show the relevant widgets, mask the others.
		"""

		if self.cellpose_button.isChecked():
			self.spatial_calib_le.setToolTip('Cellpose rescales the images such that the cells are 30.0 pixels. You can compute the scale from the training data as\n(pixel calibration [µm] in training images)*(cell diameter [px] in training images)/(30 [px]).\nIf you pass images with a different calibration to the model, they will be rescaled automatically.\nThe rescaling is ignored if you pass a diameter different from 30 px below.')
			self.ch_1_label.setText('cyto: ')
			self.ch_2_label.setText('nuclei: ')
			for c in self.stardist_options+[self.threshold_config_button]:
				c.hide()
			for c in self.cellpose_options+self.base_block_options:
				c.show()
			self.unlock_upload()
		elif self.stardist_button.isChecked():
			self.spatial_calib_le.setToolTip('')
			self.ch_1_label.setText('channel 1: ')
			self.ch_2_label.setText('channel 2: ')
			for c in self.stardist_options+self.base_block_options:
				c.show()
			for c in self.cellpose_options+[self.threshold_config_button]:
				c.hide()
			self.unlock_upload()
		else:
			for c in self.stardist_options+self.cellpose_options+self.base_block_options:
				c.hide()
			self.threshold_config_button.show()
			self.upload_button.setEnabled(True)
		self.adjustSize()

	def upload_model(self):

		"""
		Upload the model.
		"""

		if self.file_label.text()=='No file chosen':
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("Please select a model first.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None

		if not self.threshold_button.isChecked():
			
			if self.stardist_button.isChecked():
				
				try:
					shutil.copytree(self.filename, self.destination)
				except FileExistsError:
					msgBox = QMessageBox()
					msgBox.setIcon(QMessageBox.Warning)
					msgBox.setText("A model with the same name already exists in the models folder. Please rename it.")
					msgBox.setWindowTitle("Warning")
					msgBox.setStandardButtons(QMessageBox.Ok)
					returnValue = msgBox.exec()
					if returnValue == QMessageBox.Ok:
						return None

			elif self.cellpose_button.isChecked():
				try:
					if not os.path.exists(self.folder_dest):
						os.mkdir(self.folder_dest)
					shutil.copy(self.filename, self.destination)
				except FileExistsError:
					msgBox = QMessageBox()
					msgBox.setIcon(QMessageBox.Warning)
					msgBox.setText("A model with the same name already exists in the models folder. Please rename it.")
					msgBox.setWindowTitle("Warning")
					msgBox.setStandardButtons(QMessageBox.Ok)
					returnValue = msgBox.exec()
					if returnValue == QMessageBox.Ok:
						return None

			self.generate_input_config()
			self.parent.init_seg_model_list()
			self.close()
		else:
			if self.mode=="targets":	
				self.parent.threshold_config_targets = self.filename
				print('Path to threshold configuration successfully set in the software')
				self.close()
			elif self.mode=="effectors":
				self.parent.threshold_config_effectors = self.filename
				print('Path to threshold configuration successfully set in the software')
				self.close()	

	def generate_input_config(self):

		"""
		Check the ticked options and input parameters to create
		a configuration to use the uploaded model properly.
		"""
		
		dico = {}

		# Check model option
		if self.stardist_button.isChecked():
			
			# Get channels
			channels = []
			for c in self.combos:
				if c.currentText()!="--":
					channels.append(c.currentText())
			model_type = "stardist"
			spatial_calib = float(self.spatial_calib_le.text().replace(',','.'))
			normalize = self.normalize_checkbox.isChecked()
			dico.update({"channels": channels, 
						 "spatial_calibration": spatial_calib, 
						 "normalize": normalize,
						})

		elif self.cellpose_button.isChecked():
			
			# Get channels (cyto and nucleus)
			channels = []
			for c in self.combos[:2]:
				if c.currentText()!="--":
					channels.append(c.currentText())

			model_type = "cellpose"
			diameter = float(self.cp_diameter_le.text().replace(',','.'))
			cellprob_threshold = float(self.cp_cellprob_le.text().replace(',','.'))
			flow_threshold = float(self.cp_flow_le.text().replace(',','.'))
			normalize = self.normalize_checkbox.isChecked()
			spatial_calib = float(self.spatial_calib_le.text().replace(',','.'))
			if normalize:
				norm_percentile = [True]*len(channels)
				norm_clip = [True]*len(channels)
			else:
				norm_percentile = [False]*len(channels)
				norm_clip = [False]*len(channels)
			normalization_values = [[1.0,99.0]]*len(channels)

			dico.update({"channels": channels,
						 "diameter": diameter,
						 "cellprob_threshold": cellprob_threshold,
						 "flow_threshold": flow_threshold,
						 "normalize": normalize,
						 "spatial_calibration": spatial_calib,
						 'normalization_percentile': norm_percentile,
						 'normalization_clip': norm_clip,
						 'normalization_values': normalization_values,
						 })

		elif self.threshold_button.isChecked():
			model_type = "threshold"
			return None

		dico.update({"model_type": model_type})
		json_object = json.dumps(dico, indent=4)

		# Writing to sample.json
		# if os.path.exists(self.folder_dest):

		# 	msgBox = QMessageBox()
		# 	msgBox.setIcon(QMessageBox.Warning)
		# 	msgBox.setText("A model with the same name already exists. Do you want to replace it?")
		# 	msgBox.setWindowTitle("Confirm")
		# 	msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
		# 	returnValue = msgBox.exec()
		# 	if returnValue == QMessageBox.No:
		# 		return None
		# 	elif returnValue == QMessageBox.Yes:
		# 		shutil.rmtree(self.folder_dest)
		#os.mkdir(self.folder_dest)

		print("Configuration successfully written in ",self.folder_dest+"/config_input.json")
		with open(self.folder_dest+"/config_input.json", "w") as outfile:
			outfile.write(json_object)

	def open_threshold_config_wizard(self):

		if isinstance(self.parent.parent.pos, str):
			self.ThreshWizard = ThresholdConfigWizard(self)
			self.ThreshWizard.show()
		else:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("Please select a unique position before launching the wizard...")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None			