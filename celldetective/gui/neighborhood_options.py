from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QScrollArea, QComboBox, QFrame, QCheckBox, QFileDialog, QGridLayout, QTextEdit, QLineEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from celldetective.gui.gui_utils import center_window, FeatureChoice, ListWidget, QHSeperationLine, FigureCanvas, DistanceChoice, OperationChoice
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider,QLabeledSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from celldetective.utils import extract_experiment_channels, get_software_location
from celldetective.io import interpret_tracking_configuration, load_frames, auto_load_number_of_frames
from celldetective.measure import compute_haralick_features, contour_of_instance_segmentation
import numpy as np
from tifffile import imread
import json
from shutil import copyfile
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
from natsort import natsorted
from tifffile import imread
from pathlib import Path, PurePath
import gc
import pandas as pd

class ConfigNeighborhoods(QMainWindow):
	
	"""
	UI to set measurement instructions.

	"""

	def __init__(self, parent=None):
		
		super().__init__()
		self.parent = parent
		self.setWindowTitle("Configure neighborhoods")
		#self.setWindowIcon(QIcon(os.sep.join(['celldetective','icons','mexican-hat.png'])))

		self.exp_dir = self.parent.exp_dir
		self.neigh_instructions = self.parent.exp_dir + os.sep.join(["configs","neighborhood_instructions.json"])
		self.clear_previous = False
		self.not_status_reference = False
		self.not_status_neighbor = False	

		exp_config = self.exp_dir +"config.ini"
		self.channel_names, self.channels = extract_experiment_channels(exp_config)
		self.channel_names = np.array(self.channel_names)
		self.channels = np.array(self.channels)

		self.screen_height = self.parent.parent.parent.screen_height
		center_window(self)

		self.setMinimumWidth(750)
		self.setMinimumHeight(int(0.5*self.screen_height))
		self.setMaximumHeight(int(0.95*self.screen_height))
		
		self.populate_widget()
		self.load_previous_neighborhood_instructions()

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

		# second frame for ISOTROPIC MEASUREMENTS
		pop_hbox = QHBoxLayout()

		self.reference_population_frame = QFrame()
		self.reference_population_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_reference_frame()
		pop_hbox.addWidget(self.reference_population_frame, 50)

		self.neigh_population_frame = QFrame()
		self.neigh_population_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_neigh_frame()
		pop_hbox.addWidget(self.neigh_population_frame, 50)
		main_layout.addLayout(pop_hbox)

		self.radii_frame = QFrame()
		self.radii_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_radii_frame()
		main_layout.addWidget(self.radii_frame)

		self.clear_previous_btn = QCheckBox('clear previous neighborhoods')
		main_layout.addWidget(self.clear_previous_btn, alignment=Qt.AlignRight)

		main_layout.addWidget(QLabel(''))
		self.submit_btn = QPushButton('Set')
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
		#self.adjustScrollArea()


	def populate_reference_frame(self):

		"""
		Add widgets and layout in the reference population frame.
		"""

		grid = QVBoxLayout(self.reference_population_frame)
		grid.setSpacing(15)
		self.ref_lbl = QLabel("REFERENCE")
		self.ref_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.ref_lbl, 30, alignment=Qt.AlignCenter)
		self.generate_reference_contents()
		grid.addWidget(self.ContentsReference, 70)

	def populate_neigh_frame(self):

		"""
		Add widgets and layout in the neighbor population frame.
		"""

		grid = QVBoxLayout(self.neigh_population_frame)
		grid.setSpacing(15)

		self.neigh_lbl = QLabel("NEIGHBORS")
		self.neigh_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.neigh_lbl, 30, alignment=Qt.AlignCenter)
		self.generate_neighbors_contents()
		grid.addWidget(self.ContentsNeigh, 70)

	def populate_radii_frame(self):

		"""
		Add widgets and layout in the radii frame.
		"""

		grid = QVBoxLayout(self.radii_frame)

		self.dist_lbl = QLabel("NEIGHBORHOOD CUT-DISTANCES")
		self.dist_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.dist_lbl, alignment=Qt.AlignCenter)
		self.generate_radii_contents()
		grid.addWidget(self.ContentsIso) #1, 0, 1, 4, alignment=Qt.AlignTop


	def generate_radii_contents(self):

		self.ContentsIso = QFrame()
		layout = QVBoxLayout(self.ContentsIso)
		layout.setContentsMargins(0,0,0,0)

		radii_layout = QHBoxLayout()
		self.radii_lbl = QLabel('Cut-distance radii:')
		self.radii_lbl.setToolTip('From reference cells, in pixel units. Define radii for neighborhood computations.')
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
		
		self.radii_list = ListWidget(self, DistanceChoice, initial_features=["60"], dtype=int)
		layout.addWidget(self.radii_list)

		self.del_radius_btn.clicked.connect(self.radii_list.removeSel)
		self.add_radius_btn.clicked.connect(self.radii_list.addItem)

	def generate_reference_contents(self):

		self.ContentsReference = QFrame()
		layout = QVBoxLayout(self.ContentsReference)
		layout.setContentsMargins(15,15,15,15)

		pop_hbox = QHBoxLayout()
		pop_hbox.addWidget(QLabel('population: '),30)
		self.ref_pop_cb = QComboBox()
		self.ref_pop_cb.addItems(['targets','effectors'])
		pop_hbox.addWidget(self.ref_pop_cb,70)
		layout.addLayout(pop_hbox)

		status_hbox = QHBoxLayout()
		status_hbox.addWidget(QLabel('status: '), 30)
		self.ref_pop_status_cb = QComboBox()
		#self.ref_pop_status_cb.addItems(['--'])

		status_cb_hbox = QHBoxLayout()
		status_cb_hbox.setContentsMargins(0,0,0,0)
		status_cb_hbox.addWidget(self.ref_pop_status_cb,90)
		# replace with not gate

		self.ref_not_gate_btn = QPushButton("")
		self.ref_not_gate_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.ref_not_gate_btn.setIcon(icon(MDI6.gate_not,color="black"))
		self.ref_not_gate_btn.setToolTip("NOT (flip zeros and ones)")
		self.ref_not_gate_btn.setIconSize(QSize(20, 20))
		self.ref_not_gate_btn.clicked.connect(self.switch_not_reference)
		status_cb_hbox.addWidget(self.ref_not_gate_btn, 5)
		status_hbox.addLayout(status_cb_hbox,70)
		layout.addLayout(status_hbox)

		event_hbox = QHBoxLayout()
		event_hbox.addWidget(QLabel('event time: '),30)
		self.event_time_cb = QComboBox()
		#self.event_time_cb.addItems(['--'])
		event_hbox.addWidget(self.event_time_cb,70)
		layout.addLayout(event_hbox)
		
		self.set_combo_boxes_reference()
		self.ref_pop_cb.currentIndexChanged.connect(self.set_combo_boxes_reference)

	def switch_not_reference(self):
		self.not_status_reference = not self.not_status_reference
		if self.not_status_reference:
			self.ref_not_gate_btn.setIcon(icon(MDI6.gate_not,color="#1565c0"))
			self.ref_not_gate_btn.setIconSize(QSize(20, 20))
		else:
			self.ref_not_gate_btn.setIcon(icon(MDI6.gate_not,color="black"))
			self.ref_not_gate_btn.setIconSize(QSize(20, 20))

	def generate_neighbors_contents(self):

		self.ContentsNeigh = QFrame()
		layout = QVBoxLayout(self.ContentsNeigh)
		layout.setContentsMargins(15,15,15,15)

		pop_hbox = QHBoxLayout()
		pop_hbox.addWidget(QLabel('population: '),30)
		self.neigh_pop_cb = QComboBox()
		self.neigh_pop_cb.addItems(['targets','effectors'])
		self.neigh_pop_cb.currentIndexChanged.connect(self.set_combo_boxes_neigh)

		pop_hbox.addWidget(self.neigh_pop_cb,70)
		layout.addLayout(pop_hbox)

		status_hbox = QHBoxLayout()
		status_hbox.addWidget(QLabel('status: '),30)
		self.neigh_pop_status_cb = QComboBox()
		self.set_combo_boxes_neigh()

		status_cb_hbox = QHBoxLayout()
		status_cb_hbox.setContentsMargins(0,0,0,0)
		status_cb_hbox.addWidget(self.neigh_pop_status_cb,90)
		self.neigh_not_gate_btn = QPushButton("")
		self.neigh_not_gate_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.neigh_not_gate_btn.setIcon(icon(MDI6.gate_not,color="black"))
		self.neigh_not_gate_btn.setToolTip("NOT (flip zeros and ones)")
		self.neigh_not_gate_btn.setIconSize(QSize(20, 20))
		self.neigh_not_gate_btn.clicked.connect(self.switch_not_neigh)
		status_cb_hbox.addWidget(self.neigh_not_gate_btn, 5)
		status_hbox.addLayout(status_cb_hbox, 70)
		layout.addLayout(status_hbox)

		self.cum_presence_btn = QCheckBox('cumulated presence')
		layout.addWidget(self.cum_presence_btn)

		self.symmetrize_btn = QCheckBox('symmetrize')
		layout.addWidget(self.symmetrize_btn)

	def switch_not_neigh(self):
		self.not_status_neighbor = not self.not_status_neighbor
		if self.not_status_neighbor:
			self.neigh_not_gate_btn.setIcon(icon(MDI6.gate_not,color="#1565c0"))
			self.neigh_not_gate_btn.setIconSize(QSize(20, 20))
		else:
			self.neigh_not_gate_btn.setIcon(icon(MDI6.gate_not,color="black"))
			self.neigh_not_gate_btn.setIconSize(QSize(20, 20))

	def set_combo_boxes_neigh(self):
		pop = self.neigh_pop_cb.currentText()
		class_cols, status_cols, _ = self.locate_population_columns(pop)
		self.neigh_pop_status_cb.clear()
		self.neigh_pop_status_cb.addItems(['--','class','status']+class_cols+status_cols)

	def set_combo_boxes_reference(self):
		pop = self.ref_pop_cb.currentText()
		class_cols, status_cols, time_cols = self.locate_population_columns(pop)
		self.ref_pop_status_cb.clear()
		self.ref_pop_status_cb.addItems(['--','class', 'status']+class_cols+status_cols)
		self.event_time_cb.addItems(['--', 't0']+time_cols)

	def locate_population_columns(self, population):

		# Look for all classes and times
		tables = glob(self.exp_dir+os.sep.join(['W*','*','output','tables',f'trajectories_{population}.csv']))
		self.all_columns = []
		for tab in tables:
			cols = pd.read_csv(tab, nrows=1).columns.tolist()
			self.all_columns.extend(cols)
		self.all_columns = np.unique(self.all_columns)

		class_idx = np.array([s.startswith('class_') for s in self.all_columns])
		status_idx = np.array([s.startswith('status_') for s in self.all_columns])
		time_idx = np.array([s.startswith('t_') for s in self.all_columns])

		if len(class_idx)>0:
			class_columns = list(self.all_columns[class_idx])
			for c in ['class_id', 'class_color']:
				if c in class_columns:
					class_columns.remove(c)
		else:
			class_columns = []

		if len(status_idx)>0:
			status_columns = list(self.all_columns[status_idx])
		else:
			status_columns = []

		if len(time_idx)>0:
			time_columns = list(self.all_columns[time_idx])
		else:
			time_columns = []

		return class_columns, status_columns, time_columns

	def write_instructions(self):

		"""
		Write the selected options in a json file for later reading by the software.
		"""

		print('Writing instructions...')
		
		neighborhood_options = {}
		pop = [self.ref_pop_cb.currentText(), self.neigh_pop_cb.currentText()]
		neighborhood_options.update({'population': pop})
		
		status_options = [self.ref_pop_status_cb.currentText(), self.neigh_pop_status_cb.currentText()]
		for k in range(2):
			if status_options[k]=='--':
				status_options[k] = None
		if pop[0]!=pop[1]:
			mode = 'two-pop'
		else:
			mode = 'self'

		distances = self.radii_list.getItems()
		neighborhood_options.update({'distance': distances})	
		neighborhood_options.update({'clear_neigh': self.clear_previous_btn.isChecked()})
		event_time_col = self.event_time_cb.currentText()
		if event_time_col=='--':
			event_time_col = None
		neighborhood_options.update({'event_time_col': event_time_col})		


		neighborhood_kwargs = {'mode': mode, 'status': status_options, 'not_status_option': [self.not_status_reference, self.not_status_neighbor], 
							  'compute_cum_sum': self.cum_presence_btn.isChecked(), 'attention_weight': True, 'symmetrize': self.symmetrize_btn.isChecked(),
							  'include_dead_weight': True}

		neighborhood_options.update({'neighborhood_kwargs': neighborhood_kwargs})

		print('Neighborhood instructions: ', neighborhood_options)
		file_name = self.neigh_instructions
		with open(file_name, 'w') as f:
			json.dump(neighborhood_options, f, indent=4)
		print('Done.')
		self.close()
		

	def load_previous_neighborhood_instructions(self):

		"""
		Read the measurmeent options from a previously written json file and format properly for the UI.
		"""

		print('Reading instructions..')
		if os.path.exists(self.neigh_instructions):
			with open(self.neigh_instructions, 'r') as f:
				neigh_instructions = json.load(f)
				print(neigh_instructions)

				if 'distance' in neigh_instructions:
					distances = neigh_instructions['distance']
					distances = [str(d) for d in distances]
					self.radii_list.list_widget.clear()
					self.radii_list.list_widget.addItems(distances)

				if 'population' in neigh_instructions:

					pop = neigh_instructions['population']
					idx0 = self.ref_pop_cb.findText(pop[0])
					self.ref_pop_cb.setCurrentIndex(idx0)
					idx1 = self.neigh_pop_cb.findText(pop[1])
					self.neigh_pop_cb.setCurrentIndex(idx1)

				if 'clear_neigh' in neigh_instructions:
					clear_neigh = neigh_instructions['clear_neigh']
					self.clear_previous_btn.setChecked(clear_neigh)

				if 'event_time_col' in neigh_instructions:
					event_time_col = neigh_instructions['event_time_col']
					if event_time_col is None:
						event_time_col = '--'
					idx = self.event_time_cb.findText(event_time_col)
					self.event_time_cb.setCurrentIndex(idx)

				if 'neighborhood_kwargs' in neigh_instructions:
					neighborhood_kwargs = neigh_instructions['neighborhood_kwargs']
					if 'compute_cum_sum' in neighborhood_kwargs:
						self.cum_presence_btn.setChecked(neighborhood_kwargs['compute_cum_sum'])
					if 'symmetrize' in neighborhood_kwargs:
						self.symmetrize_btn.setChecked(neighborhood_kwargs['symmetrize'])
					if 'status' in neighborhood_kwargs:
						status_options = neighborhood_kwargs['status']
						status_options = ['--' if s is None else s for s in status_options]
						idx0 = self.ref_pop_status_cb.findText(status_options[0])
						self.ref_pop_status_cb.setCurrentIndex(idx0)
						idx1 = self.neigh_pop_status_cb.findText(status_options[1])
						self.neigh_pop_status_cb.setCurrentIndex(idx1)
					if 'not_status_option' in neighborhood_kwargs:
						not_status_option = neighborhood_kwargs['not_status_option']
						if not_status_option[0]:
							self.ref_not_gate_btn.click()
						if not_status_option[1]:
							self.neigh_not_gate_btn.click()



