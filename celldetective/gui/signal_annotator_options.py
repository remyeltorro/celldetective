from PyQt5.QtWidgets import QMainWindow, QComboBox, QLabel, QRadioButton, QLineEdit, QApplication, QPushButton, QScrollArea, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QSize
from celldetective.gui.gui_utils import center_window, QHSeperationLine
from superqt import QLabeledDoubleSlider
from celldetective.utils import extract_experiment_channels, get_software_location
import json
import numpy as np
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import os

class ConfigSignalAnnotator(QMainWindow):
	
	"""
	UI to set tracking parameters for bTrack.

	"""

	def __init__(self, parent=None):
		
		super().__init__()
		self.parent = parent
		self.setWindowTitle("Configure signal annotator")
		self.mode = self.parent.mode
		self.exp_dir = self.parent.exp_dir
		self.soft_path = get_software_location()
		
		if self.mode=="targets":
			self.instructions_path = self.parent.exp_dir + "configs/signal_annotator_config_targets.json"
		elif self.mode=="effectors":
			self.instructions_path = self.parent.exp_dir + "configs/signal_annotator_config_effectors.json"
				
		exp_config = self.exp_dir +"config.ini"
		#self.config_path = self.exp_dir + self.config_name
		self.channel_names, self.channels = extract_experiment_channels(exp_config)
		self.channel_names = np.array(self.channel_names)
		self.channels = np.array(self.channels)

		self.screen_height = self.parent.parent.parent.screen_height
		center_window(self)

		self.setMinimumHeight(int(0.4*self.screen_height))
		self.setMaximumHeight(int(0.8*self.screen_height))
		self.populate_widget()
		#self.load_previous_measurement_instructions()

	def populate_widget(self):

		self.scroll_area = QScrollArea(self)
		self.button_widget = QWidget()

		main_layout = QVBoxLayout()
		main_layout.setContentsMargins(30,30,30,30)

		sub_layout = QVBoxLayout()
		sub_layout.setContentsMargins(10,10,10,20)

		self.button_widget.setLayout(main_layout)
		sub_layout.setContentsMargins(30,30,30,30)

		sub_layout.addWidget(QLabel('Modality: '))
		
		# Create radio buttons
		option_layout = QHBoxLayout()
		self.gs_btn = QRadioButton('grayscale')
		self.gs_btn.setChecked(True)
		option_layout.addWidget(self.gs_btn, alignment=Qt.AlignCenter)

		self.rgb_btn = QRadioButton('RGB')
		option_layout.addWidget(self.rgb_btn, alignment=Qt.AlignCenter)
		sub_layout.addLayout(option_layout)

		self.percentile_btn = QPushButton()
		self.percentile_btn.setIcon(icon(MDI6.percent_circle_outline,color="black"))
		self.percentile_btn.setIconSize(QSize(20, 20))	
		self.percentile_btn.setStyleSheet(self.parent.parent.parent.button_select_all)	
		self.percentile_btn.setToolTip("Switch to percentile normalization values.")
		self.percentile_btn.clicked.connect(self.switch_to_absolute_normalization_mode)
		sub_layout.addWidget(self.percentile_btn, alignment=Qt.AlignRight)
	
		self.channel_cbs = [QComboBox() for i in range(3)]
		self.channel_cbs_lbls = [QLabel() for i in range(3)]

		self.min_val_les = [QLineEdit('0') for i in range(3)]
		self.min_val_lbls = [QLabel('Min value: ') for i in range(3)]
		self.max_val_les = [QLineEdit('10000') for i in range(3)]
		self.max_val_lbls = [QLabel('Max value: ') for i in range(3)]
		self.percentile_mode = False

		self.rgb_text = ['R: ', 'G: ', 'B: ']

		for i in range(3):

			hlayout = QHBoxLayout()
			self.channel_cbs[i].addItems(self.channel_names)
			self.channel_cbs[i].setCurrentIndex(i)
			self.channel_cbs_lbls[i].setText(self.rgb_text[i])
			hlayout.addWidget(self.channel_cbs_lbls[i], 20)
			hlayout.addWidget(self.channel_cbs[i], 80)
			sub_layout.addLayout(hlayout)

			hlayout2 = QHBoxLayout()
			hlayout2.addWidget(self.min_val_lbls[i], 20)
			hlayout2.addWidget(self.min_val_les[i], 80)			
			sub_layout.addLayout(hlayout2)

			hlayout3 = QHBoxLayout()
			hlayout3.addWidget(self.max_val_lbls[i], 20)
			hlayout3.addWidget(self.max_val_les[i], 80)			
			sub_layout.addLayout(hlayout3)

		self.enable_channels()
		
		self.gs_btn.toggled.connect(self.enable_channels)
		self.rgb_btn.toggled.connect(self.enable_channels)

		self.hsep = QHSeperationLine()
		sub_layout.addWidget(self.hsep)

		hbox_frac = QHBoxLayout()
		hbox_frac.addWidget(QLabel('Fraction: '), 20)

		self.fraction_slider = QLabeledDoubleSlider()
		self.fraction_slider.setSingleStep(0.05)
		self.fraction_slider.setTickInterval(0.05)
		self.fraction_slider.setSingleStep(1)
		self.fraction_slider.setOrientation(1)
		self.fraction_slider.setRange(0.1,1)
		self.fraction_slider.setValue(0.25)

		hbox_frac.addWidget(self.fraction_slider, 80)
		sub_layout.addLayout(hbox_frac)

		main_layout.addLayout(sub_layout)

		self.submit_btn = QPushButton('Save')
		self.submit_btn.setStyleSheet(self.parent.parent.parent.button_style_sheet)
		self.submit_btn.clicked.connect(self.write_instructions)
		main_layout.addWidget(self.submit_btn)


		self.button_widget.adjustSize()
		self.scroll_area.setAlignment(Qt.AlignCenter)
		self.scroll_area.setWidget(self.button_widget)
		self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		self.scroll_area.setWidgetResizable(True)
		self.setCentralWidget(self.scroll_area)
		self.show()

		self.read_instructions()

		QApplication.processEvents()

	def enable_channels(self):
		if self.gs_btn.isChecked():
			for k in range(1,3):
				self.channel_cbs[k].setEnabled(False)
				self.channel_cbs_lbls[k].setEnabled(False)

				self.min_val_les[k].setEnabled(False)
				self.min_val_lbls[k].setEnabled(False)
				self.max_val_les[k].setEnabled(False)
				self.max_val_lbls[k].setEnabled(False)

		elif self.rgb_btn.isChecked():
			for k in range(3):

				self.channel_cbs[k].setEnabled(True)
				self.channel_cbs_lbls[k].setEnabled(True)

				self.min_val_les[k].setEnabled(True)
				self.min_val_lbls[k].setEnabled(True)
				self.max_val_les[k].setEnabled(True)
				self.max_val_lbls[k].setEnabled(True)

	def switch_to_absolute_normalization_mode(self):

		if self.percentile_mode:
			self.percentile_mode = False
			self.percentile_btn.setIcon(icon(MDI6.percent_circle_outline,color="black"))
			self.percentile_btn.setIconSize(QSize(20, 20))		
			self.percentile_btn.setToolTip("Switch to percentile normalization values.")	
			for k in range(3):
				self.min_val_lbls[k].setText('Min value: ')	
				self.min_val_les[k].setText('0')	
				self.max_val_lbls[k].setText('Max value: ')	
				self.max_val_les[k].setText('10000')
		else:
			self.percentile_mode = True
			self.percentile_btn.setIcon(icon(MDI6.percent_circle,color="black"))
			self.percentile_btn.setIconSize(QSize(20, 20))
			self.percentile_btn.setToolTip("Switch to absolute normalization values.")	
			for k in range(3):
				self.min_val_lbls[k].setText('Min percentile: ')
				self.min_val_les[k].setText('0.01')	
				self.max_val_lbls[k].setText('Max percentile: ')	
				self.max_val_les[k].setText('99.99')

	def write_instructions(self):
		instructions = {'rgb_mode': self.rgb_btn.isChecked(), 'percentile_mode': self.percentile_mode, 'fraction': float(self.fraction_slider.value())}
		max_i = 3 if self.rgb_btn.isChecked() else 1
		channels = []
		for i in range(max_i):
			channels.append([self.channel_cbs[i].currentText(), float(self.min_val_les[i].text()), float(self.max_val_les[i].text())])
		instructions.update({'channels': channels})
		
		print('Instructions: ', instructions)
		file_name = self.instructions_path
		with open(file_name, 'w') as f:
			json.dump(instructions, f, indent=4)
		print('Done.')
		self.close()

	def read_instructions(self):

		print('Reading instructions..')
		if os.path.exists(self.instructions_path):
			with open(self.instructions_path, 'r') as f:
				instructions = json.load(f)
				print(instructions)
				
				if 'rgb_mode' in instructions:
					rgb_mode = instructions['rgb_mode']
					if rgb_mode:
						self.rgb_btn.setChecked(True)
						self.gs_btn.setChecked(False)

				if 'percentile_mode' in instructions:
					percentile_mode = instructions['percentile_mode']
					if percentile_mode:
						self.percentile_btn.click()

				if 'channels' in instructions:
					channels = instructions['channels']
					
					if len(channels)==1:
						max_iter = 1
					else:
						max_iter = 3

					for i in range(max_iter):
						idx = self.channel_cbs[i].findText(channels[i][0])
						self.channel_cbs[i].setCurrentIndex(idx)					
						self.min_val_les[i].setText(str(channels[0][1]))
						self.max_val_les[i].setText(str(channels[0][2]))

				if 'fraction' in instructions:
					fraction = instructions['fraction']
					self.fraction_slider.setValue(fraction)