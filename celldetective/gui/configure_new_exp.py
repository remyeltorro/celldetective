from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QDialog, QHBoxLayout, QFileDialog, QVBoxLayout, QScrollArea, QCheckBox, QSlider, QGridLayout, QLabel, QLineEdit, QPushButton, QWidget
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from celldetective.gui.gui_utils import center_window
from superqt import QLabeledSlider
from PyQt5.QtCore import Qt
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from configparser import ConfigParser
import os
from shutil import copyfile
import time

class ConfigNewExperiment(QMainWindow):

	def __init__(self, parent=None):
		
		super().__init__()
		self.parent = parent
		self.setWindowTitle("Experiment config")
		center_window(self)
		self.setFixedWidth(500)
		self.setMaximumHeight(1160)

		self.newExpFolder = str(QFileDialog.getExistingDirectory(self, 'Select directory'))
		self.populate_widget()

	def populate_widget(self):
		
		# Create button widget and layout
		self.scroll_area = QScrollArea(self)
		button_widget = QWidget()
		grid = QGridLayout()
		button_widget.setLayout(grid)

		grid.setContentsMargins(30,30,30,30)
		grid.addWidget(QLabel("Folder:"), 0, 0, 1, 3)
		self.supFolder = QLineEdit()
		self.supFolder.setAlignment(Qt.AlignLeft)	
		self.supFolder.setEnabled(True)
		self.supFolder.setText(self.newExpFolder)
		grid.addWidget(self.supFolder, 1, 0, 1, 1)

		self.browse_button = QPushButton("Browse...")
		self.browse_button.clicked.connect(self.browse_experiment_folder)
		self.browse_button.setIcon(icon(MDI6.folder, color="white"))
		self.browse_button.setStyleSheet(self.parent.button_style_sheet)
		#self.browse_button.setIcon(QIcon_from_svg(abs_path+f"/icons/browse.svg", color='white'))
		grid.addWidget(self.browse_button, 1, 1, 1, 1)

		grid.addWidget(QLabel("Experiment name:"), 2, 0, 1, 3)
		
		self.expName = QLineEdit()
		self.expName.setAlignment(Qt.AlignLeft)	
		self.expName.setEnabled(True)
		self.expName.setFixedWidth(400)
		self.expName.setText("Untitled_Experiment")
		grid.addWidget(self.expName, 3, 0, 1, 3)

		self.generate_movie_settings()
		grid.addLayout(self.ms_grid,29,0,1,3)

		self.generate_channel_params_box()
		grid.addLayout(self.channel_grid,30,0,1,3)

		self.validate_button = QPushButton("Submit")
		self.validate_button.clicked.connect(self.create_config)
		self.validate_button.setStyleSheet(self.parent.button_style_sheet)
		#self.validate_button.setIcon(QIcon_from_svg(abs_path+f"/icons/process.svg", color='white'))

		grid.addWidget(self.validate_button, 31, 0, 1, 3, alignment = Qt.AlignBottom)		
		button_widget.adjustSize()

		self.scroll_area.setAlignment(Qt.AlignCenter)
		self.scroll_area.setWidget(button_widget)
		self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		self.scroll_area.setWidgetResizable(True)
		self.setCentralWidget(self.scroll_area)
		self.show()

		QApplication.processEvents()
		self.adjustScrollArea()

	def adjustScrollArea(self):
		
		"""
		Auto-adjust scroll area to fill space 
		(from https://stackoverflow.com/questions/66417576/make-qscrollarea-use-all-available-space-of-qmainwindow-height-axis)
		"""

		step = 5
		while self.scroll_area.verticalScrollBar().isVisible() and self.height() < self.maximumHeight():
			self.resize(self.width(), self.height() + step)

	def generate_movie_settings(self):
		
		"""
		Parameters related to the movie parameters
		"""

		onlyInt = QIntValidator()
		onlyInt.setRange(0, 100000)

		self.ms_grid = QGridLayout()
		self.ms_grid.setContentsMargins(21,30,20,30)

		ms_lbl = QLabel("MOVIE SETTINGS")
		ms_lbl.setStyleSheet("""
			font-weight: bold;
			""")
		self.ms_grid.addWidget(ms_lbl, 0,0,1,3, alignment=Qt.AlignCenter)

		self.number_of_wells = QLabel("Number of wells:")
		self.ms_grid.addWidget(self.number_of_wells, 1, 0, 1, 3)

		self.SliderWells = QLabeledSlider(Qt.Horizontal, self)
		self.SliderWells.setMinimum(1)
		self.SliderWells.setMaximum(9)
		self.ms_grid.addWidget(self.SliderWells, 2, 0, 1, 3, alignment=Qt.AlignTop)

		self.number_of_positions = QLabel("Number of positions per well:")
		self.ms_grid.addWidget(self.number_of_positions, 3, 0, 1, 3)

		self.SliderPos = QLabeledSlider(Qt.Horizontal, self)
		self.SliderPos.setMinimum(1)
		self.SliderPos.setMaximum(48)

		self.ms_grid.addWidget(self.SliderPos, 4, 0, 1, 3)

		self.ms_grid.addWidget(QLabel("Calibration from pixel to µm:"), 5, 0, 1, 3)
		self.PxToUm_field = QLineEdit()
		self.PxToUm_field.setAlignment(Qt.AlignLeft)	
		self.PxToUm_field.setEnabled(True)
		self.PxToUm_field.setFixedWidth(400)
		self.PxToUm_field.setText("0.3112")
		self.ms_grid.addWidget(self.PxToUm_field, 6, 0, 1, 3)

		self.ms_grid.addWidget(QLabel("Calibration from frame to minutes:"), 7, 0, 1, 3)
		self.FrameToMin_field = QLineEdit()
		self.FrameToMin_field.setAlignment(Qt.AlignLeft)	
		self.FrameToMin_field.setEnabled(True)
		self.FrameToMin_field.setFixedWidth(400)
		self.FrameToMin_field.setText("1.0")
		self.ms_grid.addWidget(self.FrameToMin_field, 8, 0, 1, 3)

		self.movie_length = QLabel("Number of frames:")
		self.ms_grid.addWidget(self.movie_length,9, 0, 1, 3)
		self.MovieLengthSlider = QLabeledSlider(Qt.Horizontal, self)
		self.MovieLengthSlider.setMinimum(2)
		self.MovieLengthSlider.setMaximum(128)
		self.ms_grid.addWidget(self.MovieLengthSlider, 10, 0, 1, 3)

		self.ms_grid.addWidget(QLabel("Prefix for the movies:"), 11, 0, 1, 3)
		self.movie_prefix_field = QLineEdit()
		self.movie_prefix_field.setAlignment(Qt.AlignLeft)	
		self.movie_prefix_field.setEnabled(True)
		self.movie_prefix_field.setFixedWidth(400)
		self.movie_prefix_field.setText("Aligned")
		self.ms_grid.addWidget(self.movie_prefix_field, 12, 0, 1, 3)

		self.ms_grid.addWidget(QLabel("X shape in pixels:"), 13, 0, 1, 3)
		self.shape_x_field = QLineEdit()
		self.shape_x_field.setValidator(onlyInt)
		self.shape_x_field.setAlignment(Qt.AlignLeft)	
		self.shape_x_field.setEnabled(True)
		self.shape_x_field.setFixedWidth(400)
		self.shape_x_field.setText("2048")
		self.ms_grid.addWidget(self.shape_x_field, 14, 0, 1, 3)

		self.ms_grid.addWidget(QLabel("Y shape in pixels:"), 15, 0, 1, 3)
		self.shape_y_field = QLineEdit()
		self.shape_y_field.setValidator(onlyInt)
		self.shape_y_field.setAlignment(Qt.AlignLeft)	
		self.shape_y_field.setEnabled(True)
		self.shape_y_field.setFixedWidth(400)
		self.shape_y_field.setText("2048")
		self.ms_grid.addWidget(self.shape_y_field, 16, 0, 1, 3)


	def generate_channel_params_box(self):

		"""
		Parameters related to the movie channels
		"""

		self.channel_grid = QGridLayout()
		self.channel_grid.setContentsMargins(21,30,20,30)

		channel_lbl = QLabel("CHANNELS")
		channel_lbl.setStyleSheet("""
			font-weight: bold;
			""")
		self.channel_grid.addWidget(channel_lbl, 0,0,1,3, alignment=Qt.AlignCenter)

		self.bf_op = QCheckBox("brightfield")
		self.bf_op.toggled.connect(self.show_bf_slider)
		self.channel_grid.addWidget(self.bf_op, 1,0,1,1)
		self.SliderBF = QLabeledSlider(Qt.Orientation.Horizontal)
		self.SliderBF.setMinimum(0)
		self.SliderBF.setMaximum(6)
		self.channel_grid.addWidget(self.SliderBF, 1, 1, 1, 2, alignment = Qt.AlignRight)
		self.SliderBF.setEnabled(False)

		self.live_nuc_op = QCheckBox("live nuclei channel\n(Hoechst, NucSpot®)")
		self.live_nuc_op.toggled.connect(self.show_live_slider)
		self.channel_grid.addWidget(self.live_nuc_op, 3,0,1,1)
		self.SliderLive = QLabeledSlider(Qt.Orientation.Horizontal)
		self.SliderLive.setMinimum(0)
		self.SliderLive.setMaximum(6)
		self.channel_grid.addWidget(self.SliderLive, 3, 1, 1, 2, alignment = Qt.AlignRight)
		self.SliderLive.setEnabled(False)

		self.dead_nuc_op = QCheckBox("dead nuclei channel\n(PI)")
		self.dead_nuc_op.toggled.connect(self.show_dead_slider)
		self.channel_grid.addWidget(self.dead_nuc_op, 4,0,1,1)
		self.SliderDead = QLabeledSlider(Qt.Orientation.Horizontal)
		self.SliderDead.setMinimum(0)
		self.SliderDead.setMaximum(6)
		self.channel_grid.addWidget(self.SliderDead, 4, 1, 1, 2, alignment = Qt.AlignRight)
		self.SliderDead.setEnabled(False)

		self.effector_fluo_op = QCheckBox("effector fluorescence\n(CFSE)")
		self.effector_fluo_op.toggled.connect(self.show_effector_slider)
		self.channel_grid.addWidget(self.effector_fluo_op, 5,0,1,1)
		self.SliderEffectorFluo = QLabeledSlider(Qt.Orientation.Horizontal)
		self.SliderEffectorFluo.setMinimum(0)
		self.SliderEffectorFluo.setMaximum(6)
		self.channel_grid.addWidget(self.SliderEffectorFluo, 5, 1, 1, 2, alignment = Qt.AlignRight)
		self.SliderEffectorFluo.setEnabled(False)

		self.adhesion_op = QCheckBox("adhesion\n(RICM, IRM)")
		self.adhesion_op.toggled.connect(self.show_adhesion_slider)
		self.channel_grid.addWidget(self.adhesion_op, 6,0,1,1)
		self.SliderAdhesion = QLabeledSlider(Qt.Orientation.Horizontal)
		self.SliderAdhesion.setMinimum(0)
		self.SliderAdhesion.setMaximum(6)
		self.channel_grid.addWidget(self.SliderAdhesion, 6, 1, 1, 2, alignment = Qt.AlignRight)
		self.SliderAdhesion.setEnabled(False)

		self.fluo_1_op = QCheckBox("fluorescence miscellaneous 1\n(LAMP-1, Actin...)")
		self.fluo_1_op.toggled.connect(self.show_fluo_1_slider)
		self.channel_grid.addWidget(self.fluo_1_op, 7,0,1,1)
		self.SliderFluo1 = QLabeledSlider(Qt.Orientation.Horizontal)
		self.SliderFluo1.setMinimum(0)
		self.SliderFluo1.setMaximum(6)
		self.channel_grid.addWidget(self.SliderFluo1, 7, 1, 1, 2, alignment = Qt.AlignRight)
		self.SliderFluo1.setEnabled(False)

		self.fluo_2_op = QCheckBox("fluorescence miscellaneous 2\n(LAMP-1, Actin...)")
		self.fluo_2_op.toggled.connect(self.show_fluo_2_slider)
		self.channel_grid.addWidget(self.fluo_2_op, 8,0,1,1)
		self.SliderFluo2 = QLabeledSlider(Qt.Orientation.Horizontal)
		self.SliderFluo2.setMinimum(0)
		self.SliderFluo2.setMaximum(6)
		self.channel_grid.addWidget(self.SliderFluo2, 8, 1, 1, 2, alignment = Qt.AlignRight)
		self.SliderFluo2.setEnabled(False)

	def show_fluo_2_slider(self):
		if self.fluo_2_op.isChecked():
			self.SliderFluo2.setEnabled(True)
		else:
			self.fluo_2_op.setText("fluorescence miscellaneous 2\n(LAMP-1, Actin...)")
			self.SliderFluo2.setEnabled(False)

	def show_fluo_1_slider(self):
		if self.fluo_1_op.isChecked():
			self.SliderFluo1.setEnabled(True)
		else:
			self.fluo_1_op.setText("fluorescence miscellaneous 1\n(LAMP-1, Actin...)")
			self.SliderFluo1.setEnabled(False)

	def show_adhesion_slider(self):
		if self.adhesion_op.isChecked():
			self.SliderAdhesion.setEnabled(True)
		else:
			self.adhesion_op.setText("adhesion\n(RICM, IRM)")
			self.SliderAdhesion.setEnabled(False)

	def show_effector_slider(self):
		if self.effector_fluo_op.isChecked():
			self.SliderEffectorFluo.setEnabled(True)
		else:
			self.effector_fluo_op.setText("effector fluorescence\n(CFSE)")
			self.SliderEffectorFluo.setEnabled(False)

	def show_bf_slider(self):
		if self.bf_op.isChecked():
			self.SliderBF.setEnabled(True)
		else:
			self.bf_op.setText("brightfield")
			self.SliderBF.setEnabled(False)
	
	def show_dead_slider(self):
		if self.dead_nuc_op.isChecked():
			self.SliderDead.setEnabled(True)
		else:
			self.dead_nuc_op.setText("dead nuclei channel\n(PI)")
			self.SliderDead.setEnabled(False)

	def show_live_slider(self):
		if self.live_nuc_op.isChecked():
			self.SliderLive.setEnabled(True)
		else:
			self.live_nuc_op.setText("live nuclei channel\n(Hoechst, NucSpot®)")
			self.SliderLive.setEnabled(False)

	def show_green_slider(self):
		if self.green_check.isChecked():
			self.SliderGreen.show()
		else:
			self.green_check.setText("CFSE channel")
			self.SliderGreen.hide()

	def browse_experiment_folder(self):

		"""
		Set a new base directory.
		"""

		self.newExpFolder = str(QFileDialog.getExistingDirectory(self, 'Select directory'))
		self.supFolder.setText(self.newExpFolder)

	def create_config(self):

		"""
		Create the folder tree, the config, issue a warning if the experiment folder already exists.
		"""
		
		try:
			self.directory = self.supFolder.text()+"/"+self.expName.text()
			os.mkdir(self.directory)
			os.chdir(self.directory)
			self.create_subfolders()
			self.annotate_wells()
		except FileExistsError:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("This experiment already exists... Please select another name.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None

	def create_subfolders(self):

		"""
		Create the well folders and position folders within the wells.
		"""
		
		self.nbr_wells = self.SliderWells.value()
		self.nbr_positions = self.SliderPos.value()
		for k in range(self.nbr_wells):
			well_name = f"W{k+1}/"
			os.mkdir(well_name)
			for p in range(self.nbr_positions):
				position_name = well_name+f"{k+1}0{p}/"
				os.mkdir(position_name)
				os.mkdir(position_name+"/movie/")

	def annotate_wells(self):
		self.w = SetupConditionLabels(self, self.nbr_wells)
		self.w.show()


	def create_config_file(self):

		"""

		Write all user input parameters to a configuration file associated to an experiment.
		"""

		config = ConfigParser()

		# add a new section and some values
		config.add_section('MovieSettings')
		config.set('MovieSettings', 'PxToUm', self.PxToUm_field.text())
		config.set('MovieSettings', 'FrameToMin', self.FrameToMin_field.text())
		config.set('MovieSettings', 'len_movie', str(self.MovieLengthSlider.value()))
		config.set('MovieSettings', 'shape_x', self.shape_x_field.text())
		config.set('MovieSettings', 'shape_y', self.shape_y_field.text())
		
		# Brightfield
		if self.bf_op.isChecked():
			config.set('MovieSettings', 'brightfield_channel', str(self.SliderBF.value()))
		else:
			config.set('MovieSettings', 'brightfield_channel', "nan")

		# Live nuclei
		if self.live_nuc_op.isChecked():
			config.set('MovieSettings', 'live_nuclei_channel', str(self.SliderLive.value()))
		else:
			config.set('MovieSettings', 'live_nuclei_channel', "nan")

		# Dead nuclei
		if self.dead_nuc_op.isChecked():
			config.set('MovieSettings', 'dead_nuclei_channel', str(self.SliderDead.value()))
		else:
			config.set('MovieSettings', 'dead_nuclei_channel', "nan")

		# Effector
		if self.effector_fluo_op.isChecked():
			config.set('MovieSettings', 'effector_fluo_channel', str(self.SliderEffectorFluo.value()))
		else:
			config.set('MovieSettings', 'effector_fluo_channel', "nan")

		# Adhesion
		if self.adhesion_op.isChecked():
			config.set('MovieSettings', 'adhesion_channel', str(self.SliderAdhesion.value()))
		else:
			config.set('MovieSettings', 'adhesion_channel', "nan")

		# Fluo 1
		if self.fluo_1_op.isChecked():
			config.set('MovieSettings', 'fluo_channel_1', str(self.SliderFluo1.value()))
		else:
			config.set('MovieSettings', 'fluo_channel_1', "nan")

		# Fluo 2
		if self.fluo_2_op.isChecked():
			config.set('MovieSettings', 'fluo_channel_2', str(self.SliderFluo2.value()))
		else:
			config.set('MovieSettings', 'fluo_channel_2', "nan")
		
		config.set('MovieSettings', 'movie_prefix', self.movie_prefix_field.text())

		config.add_section('SearchRadii')
		config.set('SearchRadii', 'search_radius_targets', "100")
		config.set('SearchRadii', 'search_radius_effectors', "62")

		config.add_section('Labels')
		config.set('Labels', 'cell_types', self.cell_types)
		config.set('Labels', 'antibodies', self.antibodies)
		config.set('Labels', 'concentrations', self.concentrations)
		config.set('Labels', 'pharmaceutical_agents', self.pharmaceutical_agents)

		# save to a file
		with open('config.ini', 'w') as configfile:
			config.write(configfile)

		self.parent.set_experiment_path(self.directory)
		print(f'New experiment successfully configured in folder {self.directory}...')
		self.close()

class SetupConditionLabels(QWidget):
	def __init__(self, parent, n_wells):
		super().__init__()
		self.parent = parent
		self.n_wells = n_wells
		self.setWindowTitle("Label the wells")
		self.layout = QVBoxLayout()
		self.layout.setContentsMargins(30,30,30,30)
		self.setLayout(self.layout)
		self.populate()
		center_window(self)

	def populate(self):

		self.cell_type_cbs = [QLineEdit() for i in range(self.n_wells)]
		self.antibodies_cbs = [QLineEdit() for i in range(self.n_wells)]
		self.concentrations_cbs = [QLineEdit() for i in range(self.n_wells)]
		self.pharmaceutical_agents_cbs = [QLineEdit() for i in range(self.n_wells)]

		for i in range(self.n_wells):
			hbox = QHBoxLayout()
			hbox.setContentsMargins(15,5,15,5)
			hbox.addWidget(QLabel(f'well {i+1}'), 5, alignment=Qt.AlignLeft)
			hbox.addWidget(QLabel('cell type: '), 5)
			hbox.addWidget(self.cell_type_cbs[i], 10)
			self.cell_type_cbs[i].setPlaceholderText('e.g. T-cell, NK')

			hbox.addWidget(QLabel('antibody: '), 5)
			hbox.addWidget(self.antibodies_cbs[i], 10)
			self.antibodies_cbs[i].setPlaceholderText('e.g. anti-CD4')
			
			hbox.addWidget(QLabel('concentration: '), 5)
			hbox.addWidget(self.concentrations_cbs[i], 10)
			self.concentrations_cbs[i].setPlaceholderText('e.g. 100 (pM)')
			
			hbox.addWidget(QLabel('pharmaceutical agents: '), 5)
			hbox.addWidget(self.pharmaceutical_agents_cbs[i], 10)
			self.pharmaceutical_agents_cbs[i].setPlaceholderText('e.g. dextran')

			self.layout.addLayout(hbox)

		btn_hbox = QHBoxLayout()
		btn_hbox.setContentsMargins(0,20,0,0)
		self.skip_btn = QPushButton('Skip')
		self.skip_btn.setStyleSheet(self.parent.parent.button_style_sheet_2)
		self.skip_btn.clicked.connect(self.set_default_values)
		btn_hbox.addWidget(self.skip_btn)

		self.submit_btn = QPushButton('Submit')
		self.submit_btn.setStyleSheet(self.parent.parent.button_style_sheet)
		self.submit_btn.clicked.connect(self.set_user_values)
		btn_hbox.addWidget(self.submit_btn)
		self.layout.addLayout(btn_hbox)

	def set_default_values(self):
		
		for i in range(self.n_wells):
			self.cell_type_cbs[i].setText(str(i))
			self.antibodies_cbs[i].setText(str(i))
			self.concentrations_cbs[i].setText(str(i))
			self.pharmaceutical_agents_cbs[i].setText(str(i))
		self.set_attributes()
		self.parent.create_config_file()
		self.close()

	def set_user_values(self):
		for i in range(self.n_wells):
			if self.cell_type_cbs[i].text()=='':
				self.cell_type_cbs[i].setText(str(i))
			if self.antibodies_cbs[i].text()=='':
				self.antibodies_cbs[i].setText(str(i))
			if self.concentrations_cbs[i].text()=='':
				self.concentrations_cbs[i].setText(str(i))
			if self.pharmaceutical_agents_cbs[i].text()=='':
				self.pharmaceutical_agents_cbs[i].setText(str(i))
		self.set_attributes()
		self.parent.create_config_file()
		self.close()

	def set_attributes(self):

		cell_type_text = [c.text() for c in self.cell_type_cbs]
		self.parent.cell_types = ','.join(cell_type_text)

		antibodies_text = [c.text() for c in self.antibodies_cbs]
		self.parent.antibodies = ','.join(antibodies_text)

		concentrations_text = [c.text() for c in self.concentrations_cbs]
		self.parent.concentrations = ','.join(concentrations_text)

		pharamaceutical_text = [c.text() for c in self.pharmaceutical_agents_cbs]
		self.parent.pharmaceutical_agents = ','.join(pharamaceutical_text)





