from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QCheckBox, QLineEdit, QHBoxLayout, QRadioButton, QComboBox, QFileDialog, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, QSize
from celldetective.gui.gui_utils import center_window
from PyQt5.QtGui import QDoubleValidator
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import numpy as np
from glob import glob
import os

class SegmentationModelLoader(QWidget):

	def __init__(self, parent):
		
		super().__init__()
		self.parent = parent
		self.mode = self.parent.mode
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
		self.cellpose_button.toggled.connect(self.show_cellpose_options)
		#self.cellpose_button.setIcon(QIcon(abs_path+f"/icons/cellpose.png"))
		option_layout.addWidget(self.cellpose_button)

		self.threshold_button = QRadioButton('Threshold')
		#self.threshold_button.toggled.connect(self.enable_config_wizard)
		option_layout.addWidget(self.threshold_button)

		self.layout.addLayout(option_layout, 1,0,1,2, alignment=Qt.AlignCenter)

		self.calibration_label = QLabel("Training calibration:\n[µm]")
		self.layout.addWidget(self.calibration_label,2,0,1,1)
		self.spatial_calib_le = QLineEdit("0,1")
		self.qdv = QDoubleValidator(0.0, np.amax([self.parent.parent.shape_x, self.parent.parent.shape_y]), 8, notation=QDoubleValidator.StandardNotation)
		self.spatial_calib_le.setValidator(self.qdv)

		self.layout.addWidget(self.spatial_calib_le, 2,1,1,2)
		#self.stardist_options = [self.spatial_calib_le, self.calibration_label]

		self.channel_options = ["--","live_nuclei_channel", "dead_nuclei_channel", "effector_fluo_channel", "brightfield_channel", "adhesion_channel", "fluo_channel_1", "fluo_channel_2"]
		self.ch_1_label = QLabel("channel 1:")
		self.layout.addWidget(self.ch_1_label, 3, 0, 1, 1)
		self.combo_ch1 = QComboBox()
		self.combo_ch1.addItems(self.channel_options)
		self.layout.addWidget(self.combo_ch1, 3, 1, 1, 2)

		self.ch_2_label = QLabel("channel 2:")
		self.layout.addWidget(self.ch_2_label, 4, 0, 1, 1)
		self.combo_ch2 = QComboBox()
		self.combo_ch2.addItems(self.channel_options)
		self.layout.addWidget(self.combo_ch2, 4, 1, 1, 2)
		
		self.ch_3_label = QLabel("channel 3:")
		self.layout.addWidget(self.ch_3_label, 5, 0, 1, 1)
		self.combo_ch3 = QComboBox()
		self.combo_ch3.addItems(self.channel_options)
		self.layout.addWidget(self.combo_ch3, 5, 1, 1, 2)

		self.ch_4_label = QLabel("channel 4:")
		self.layout.addWidget(self.ch_4_label, 6, 0, 1, 1)
		self.combo_ch4 = QComboBox()
		self.combo_ch4.addItems(self.channel_options)
		self.layout.addWidget(self.combo_ch4, 6, 1, 1, 2)

		self.stardist_options = [self.ch_3_label, self.ch_4_label, self.combo_ch3, self.combo_ch4]
		self.combos = [self.combo_ch1, self.combo_ch2, self.combo_ch3, self.combo_ch4]

		self.normalize_checkbox = QCheckBox()
		self.normalize_checkbox.setChecked(True)
		self.layout.addWidget(QLabel('normalize: '), 8,0,1,1)
		self.layout.addWidget(self.normalize_checkbox,8, 1,1,2, Qt.AlignLeft)

		self.generate_cellpose_options()

		# Create file dialog
		self.file_dialog = QFileDialog()

		# Create button to open file dialog
		self.open_dialog_button = QPushButton('Choose File')
		self.open_dialog_button.clicked.connect(self.showDialog)
		self.file_label = QLabel('No file chosen', self)
		self.layout.addWidget(self.open_dialog_button, 9, 0, 1, 1)
		self.layout.addWidget(self.file_label, 9, 1, 1, 1)

		self.upload_button = QPushButton("Upload")
		#self.upload_button.clicked.connect(self.upload_model)
		self.upload_button.setIcon(icon(MDI6.upload,color="white"))
		self.upload_button.setIconSize(QSize(25, 25))
		self.upload_button.setStyleSheet(self.parent.parent.parent.button_style_sheet)
		self.layout.addWidget(self.upload_button, 10, 0, 1, 1)

		self.setLayout(self.layout)
		self.show()

	def generate_cellpose_options(self):

		self.cp_diameter_label = QLabel('diameter: ')
		self.cp_diameter_le = QLineEdit("30,0")
		self.cp_diameter_le.setValidator(self.qdv)
		self.layout.addWidget(self.cp_diameter_label, 5, 0, 1, 2)
		self.layout.addWidget(self.cp_diameter_le, 5, 1, 1, 2)


		qdv_prob = QDoubleValidator(-6, 6, 8, notation=QDoubleValidator.StandardNotation)
		self.cp_cellprob_label = QLabel('cellprob\nthreshold: ')
		self.cp_cellprob_le = QLineEdit('0,0')
		self.cp_cellprob_le.setValidator(qdv_prob)
		self.layout.addWidget(self.cp_cellprob_label, 6, 0, 1, 2)
		self.layout.addWidget(self.cp_cellprob_le, 6, 1, 1, 2)


		self.cp_flow_label = QLabel('flow threshold: ')
		self.cp_flow_le = QLineEdit('0,4')
		self.cp_flow_le.setValidator(qdv_prob)
		self.layout.addWidget(self.cp_flow_label, 7, 0, 1, 2)
		self.layout.addWidget(self.cp_flow_le, 7, 1, 1, 2)

		self.cellpose_options = [self.cp_diameter_label,self.cp_diameter_le,self.cp_cellprob_label,
			self.cp_cellprob_le, self.cp_flow_label, self.cp_flow_le]
		for c in self.cellpose_options:
			c.hide()

	def showDialog(self):

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
				if not os.path.exists(self.folder_dest):
					os.mkdir(self.folder_dest)
				self.destination = self.folder_dest+f"/{self.modelname}"

			if self.seg_mode=="threshold":
				self.file_label.setText(self.filename.split("/")[-1])

	def show_cellpose_options(self):

		if self.cellpose_button.isChecked():
			self.spatial_calib_le.setToolTip('Cellpose rescales the images such that the cells are 30.0 pixels. You can compute the scale from the training data as\n(pixel calibration [µm] in training images)*(cell diameter [px] in training images)/(30 [px]).\nIf you pass images with a different calibration to the model, they will be rescaled automatically.\nThe rescaling is ignored if you pass a diameter different from 30 px below.')
			self.ch_1_label.setText('cyto:')
			self.ch_2_label.setText('nuclei:')
			for c in self.stardist_options:
				c.hide()
			for c in self.cellpose_options:
				c.show()

		else:
			self.spatial_calib_le.setToolTip('')
			self.ch_1_label.setText('channel 1:')
			self.ch_2_label.setText('channel 2:')
			for c in self.stardist_options:
				c.show()
			for c in self.cellpose_options:
				c.hide()
		self.layout.setSpacing(5)
		self.adjustSize()
