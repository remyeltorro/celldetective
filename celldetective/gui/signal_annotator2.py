import copy

from PyQt5.QtWidgets import QMainWindow, QComboBox, QLabel, QRadioButton, QLineEdit, QFileDialog, QApplication, \
	QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QAction, QShortcut, QLineEdit, QTabWidget, \
	QButtonGroup, QGridLayout, QSlider, QCheckBox, QToolButton
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QKeySequence
from matplotlib.collections import LineCollection
from celldetective.gui import SignalAnnotator, Styles
from celldetective.gui.gui_utils import center_window, QHSeperationLine, FilterChoice
from superqt import QLabeledDoubleSlider, QLabeledDoubleRangeSlider, QLabeledSlider, QSearchableComboBox
from celldetective.utils import extract_experiment_channels, get_software_location, _get_img_num_per_channel
from celldetective.io import auto_load_number_of_frames, load_frames
from celldetective.gui.gui_utils import FigureCanvas, color_from_status, color_from_class
import json
import numpy as np
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import os
from glob import glob
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
import gc
from matplotlib.animation import FuncAnimation
from matplotlib.cm import tab10
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class SignalAnnotator2(QMainWindow,Styles):

	"""
	UI to set tracking parameters for bTrack.

	"""

	def __init__(self, parent=None):

		super().__init__()
		self.parent_window = parent
		self.setWindowTitle("Signal annotator")

		self.pos = self.parent_window.parent_window.pos
		self.exp_dir = self.parent_window.exp_dir
		print(f'{self.pos=} {self.exp_dir=}')

		self.soft_path = get_software_location()
		self.recently_modified = False
		self.n_signals = 3
		self.target_selection = []
		self.effector_selection = []
		self.reference_selection = []
		self.neighbor_selection = []
		self.pair_selection = []

		# Read instructions from target block for now...
		self.mode = "neighborhood"
		self.instructions_path = self.exp_dir + os.sep.join(['configs', 'signal_annotator_config_neighborhood.json'])

		# default params
		self.target_class_name = 'class'
		self.target_time_name = 't0'
		self.target_status_name = 'status'

		center_window(self)
		
		# Locate stack
		self.locate_stack()
		self.load_annotator_config()

		# Locate tracks
		self.locate_target_tracks()
		self.locate_effector_tracks()

		self.neighborhood_cols = []
		if self.df_targets is not None:
			self.neighborhood_cols.extend([c for c in list(self.df_targets.columns) if c.startswith('neighborhood')])
		if self.df_effectors is not None:
			print(self.df_effectors.columns)
			self.neighborhood_cols.extend([c for c in list(self.df_effectors.columns) if c.startswith('neighborhood')])
		print(f"The following neighborhoods were detected: {self.neighborhood_cols=}")
		self.locate_relative_tracks()
		
		# Prepare stack
		self.prepare_stack()

		self.generate_signal_choices()
		self.frame_lbl = QLabel('frame: ')
		self.looped_animation()
		self.create_cell_signal_canvas()

		self.populate_widget()

		# Widget settings
		self.screen_height = self.parent_window.parent_window.parent_window.screen_height
		self.screen_width = self.parent_window.parent_window.parent_window.screen_width
		self.setMinimumWidth(int(0.8*self.screen_width))
		self.setMinimumHeight(int(0.8*self.screen_height))
		self.setAttribute(Qt.WA_DeleteOnClose)

	def populate_widget(self):

		"""
		Create the multibox design.

		"""

		self.button_widget = QWidget()
		main_layout = QHBoxLayout()
		self.button_widget.setLayout(main_layout)

		main_layout.setContentsMargins(30,30,30,30)
		self.left_panel = QVBoxLayout()
		self.left_panel.setContentsMargins(30,30,30,30)
		self.left_panel.setSpacing(10)

		self.right_panel = QVBoxLayout()

		#NEIGHBORHOOD
		neigh_hbox = QHBoxLayout()
		neigh_hbox.addWidget(QLabel('neighborhood: '), 25)
		self.neighborhood_choice_cb = QComboBox()
		self.neighborhood_choice_cb.addItems(self.neighborhood_cols)
		self.neighborhood_choice_cb.setCurrentIndex(0)
		neigh_hbox.addWidget(self.neighborhood_choice_cb, 88)
		self.left_panel.addLayout(neigh_hbox)

		class_hbox = QHBoxLayout()
		class_hbox.addWidget(QLabel('relative event: '), 25)
		self.relative_class_choice_cb = QComboBox()
		self.relative_class_choice_cb.addItems(self.relative_class_cols)
		self.relative_class_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors_pair)
		self.relative_class_choice_cb.setCurrentIndex(0)

		class_hbox.addWidget(self.relative_class_choice_cb, 70)

		self.set_reference_and_neighbor_populations()
		self.neighborhood_choice_cb.currentIndexChanged.connect(self.neighborhood_changed)
		self.compute_status_and_colors_pair()

		self.relative_add_class_btn = QPushButton('')
		self.relative_add_class_btn.setStyleSheet(self.button_select_all)
		self.relative_add_class_btn.setIcon(icon(MDI6.plus,color="black"))
		self.relative_add_class_btn.setToolTip("Add a new event class")
		self.relative_add_class_btn.setIconSize(QSize(20, 20))
		self.relative_add_class_btn.clicked.connect(self.create_new_relative_event_class)
		class_hbox.addWidget(self.relative_add_class_btn, 5)

		self.relative_del_class_btn = QPushButton('')
		self.relative_del_class_btn.setStyleSheet(self.button_select_all)
		self.relative_del_class_btn.setIcon(icon(MDI6.delete,color="black"))
		self.relative_del_class_btn.setToolTip("Delete an event class")
		self.relative_del_class_btn.setIconSize(QSize(20, 20))
		self.relative_del_class_btn.clicked.connect(self.del_relative_event_class)
		class_hbox.addWidget(self.relative_del_class_btn, 5)


		self.left_panel.addLayout(class_hbox)
		self.cell_events_hbox = QHBoxLayout()
		self.cell_events_hbox.addWidget(QLabel('reference event: '), 25)
		self.reference_event_choice_cb = QComboBox()
		if self.reference_population=='targets':
			self.reference_event_choice_cb.addItems(self.target_class_cols)
			self.reference_event_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors_targets)

		else:
			self.reference_event_choice_cb.addItems(self.effector_class_cols)
			self.reference_event_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors_effectors)
		self.cell_events_hbox.addWidget(self.reference_event_choice_cb, 70)
		if 'self' not in self.neighborhood_choice_cb.currentText():
			self.neigh_lab=QLabel('neighbor event: ')
			self.cell_events_hbox.addWidget(self.neigh_lab, 25)
			self.neighbor_event_choice_cb = QComboBox()
			if self.neighbor_population=='targets':
				self.neighbor_event_choice_cb.addItems(self.target_class_cols)
				self.neighbor_event_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors_targets)
			else:
				self.neighbor_event_choice_cb.addItems(self.effector_class_cols)
				self.neighbor_event_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors_effectors)


			self.cell_events_hbox.addWidget(self.neighbor_event_choice_cb, 70)

		self.left_panel.addLayout(self.cell_events_hbox)
		self.cell_info_hbox=QHBoxLayout()
		self.reference_cell_info = QLabel('')
		self.neighbor_cell_info= QLabel('')
		self.cell_info_hbox.addWidget(self.reference_cell_info)
		self.cell_info_hbox.addWidget(self.neighbor_cell_info)
		self.left_panel.addLayout(self.cell_info_hbox)

		# Annotation buttons
		options_hbox = QHBoxLayout()
		options_hbox.setContentsMargins(150, 30, 50, 0)
		self.event_btn = QRadioButton('event')
		self.event_btn.setStyleSheet(self.button_style_sheet_2)
		self.event_btn.toggled.connect(self.enable_time_of_interest)

		self.no_event_btn = QRadioButton('no event')
		self.no_event_btn.setStyleSheet(self.button_style_sheet_2)
		self.no_event_btn.toggled.connect(self.enable_time_of_interest)

		self.else_btn = QRadioButton('else')
		self.else_btn.setStyleSheet(self.button_style_sheet_2)
		self.else_btn.toggled.connect(self.enable_time_of_interest)

		self.suppr_btn = QRadioButton('mark for\nsuppression')
		self.suppr_btn.setStyleSheet(self.button_style_sheet_2)
		self.suppr_btn.toggled.connect(self.enable_time_of_interest)

		options_hbox.addWidget(self.event_btn, 25)
		options_hbox.addWidget(self.no_event_btn, 25)
		options_hbox.addWidget(self.else_btn, 25)
		options_hbox.addWidget(self.suppr_btn, 25)
		self.left_panel.addLayout(options_hbox)

		time_option_hbox = QHBoxLayout()
		time_option_hbox.setContentsMargins(100, 30, 100, 30)
		self.time_of_interest_label = QLabel('time of interest: ')
		time_option_hbox.addWidget(self.time_of_interest_label, 30)
		self.time_of_interest_le = QLineEdit()
		time_option_hbox.addWidget(self.time_of_interest_le, 70)
		self.left_panel.addLayout(time_option_hbox)

		main_action_hbox = QHBoxLayout()
		self.correct_btn = QPushButton('correct')
		self.correct_btn.setIcon(icon(MDI6.redo_variant, color="white"))
		self.correct_btn.setIconSize(QSize(20, 20))
		self.correct_btn.setStyleSheet(self.button_style_sheet)
		self.correct_btn.clicked.connect(self.show_annotation_buttons)
		self.correct_btn.setEnabled(False)
		main_action_hbox.addWidget(self.correct_btn)

		self.cancel_btn = QPushButton('cancel')
		self.cancel_btn.setStyleSheet(self.button_style_sheet_2)
		self.cancel_btn.setShortcut(QKeySequence("Esc"))
		self.cancel_btn.setEnabled(False)
		self.cancel_btn.clicked.connect(self.cancel_selection)
		main_action_hbox.addWidget(self.cancel_btn)
		self.left_panel.addLayout(main_action_hbox)

		self.annotation_btns_to_hide = [self.event_btn, self.no_event_btn,
										self.else_btn, self.time_of_interest_label,
										self.time_of_interest_le, self.suppr_btn]
		self.hide_annotation_buttons()

		self.del_shortcut = QShortcut(Qt.Key_Delete, self) #QKeySequence("s")
		self.del_shortcut.activated.connect(self.shortcut_suppr)
		self.del_shortcut.setEnabled(False)

		self.no_event_shortcut = QShortcut(QKeySequence("n"), self) #QKeySequence("s")
		self.no_event_shortcut.activated.connect(self.shortcut_no_event)
		self.no_event_shortcut.setEnabled(False)


		# Cell signals
		self.left_panel.addWidget(self.cell_fcanvas)

		plot_buttons_hbox = QHBoxLayout()
		plot_buttons_hbox.setContentsMargins(0,0,0,0)
		self.normalize_features_btn = QPushButton('')
		self.normalize_features_btn.setStyleSheet(self.button_select_all)
		self.normalize_features_btn.setIcon(icon(MDI6.arrow_collapse_vertical,color="black"))
		self.normalize_features_btn.setIconSize(QSize(25, 25))
		self.normalize_features_btn.setFixedSize(QSize(30, 30))
		#self.normalize_features_btn.setShortcut(QKeySequence('n'))
		self.normalize_features_btn.clicked.connect(self.normalize_features)

		plot_buttons_hbox.addWidget(QLabel(''), 90)
		plot_buttons_hbox.addWidget(self.normalize_features_btn, 5)
		self.normalized_signals = False

		self.log_btn = QPushButton()
		self.log_btn.setIcon(icon(MDI6.math_log,color="black"))
		self.log_btn.setStyleSheet(self.button_select_all)
		self.log_btn.clicked.connect(self.switch_to_log)
		plot_buttons_hbox.addWidget(self.log_btn, 5)

		self.left_panel.addLayout(plot_buttons_hbox)
		signal_choice_grid = QGridLayout()
		signal_choice_grid.setContentsMargins(30,0,30,50)
		target_label = QPushButton()
		target_label.setStyleSheet(self.button_select_all)
		target_label.setIcon(icon(MDI6.close_circle_outline,color='black'))
		target_label.setToolTip('reference cell')
		#self.correct_btn.setIcon(icon(MDI6.redo_variant, color="white"))
		#icon1 = qta.icon('MDI.close_circle_outline',color='black')
		#pixmap=icon1.pixmap
		#target_label.setPixmap(pixmap)
		effector_label = QPushButton()
		effector_label.setStyleSheet(self.button_select_all)
		effector_label.setIcon(icon(MDI6.triangle_outline,color='black'))
		effector_label.setToolTip('neighbor cell')

		relative_label = QPushButton()
		relative_label.setStyleSheet(self.button_select_all)
		relative_label.setIcon(icon(MDI6.vector_line,color='black'))
		relative_label.setToolTip('pair')

		signal_choice_grid.addWidget(target_label, 0, 0,alignment=Qt.AlignHCenter)
		signal_choice_grid.addWidget(effector_label, 0, 1,alignment=Qt.AlignHCenter)
		signal_choice_grid.addWidget(relative_label, 0, 2,alignment=Qt.AlignHCenter)
		signal_choice_grid.addWidget(self.target_button1, 1, 0,alignment=Qt.AlignHCenter)
		signal_choice_grid.addWidget(self.effector_button1, 1, 1,alignment=Qt.AlignHCenter)
		signal_choice_grid.addWidget(self.relative_button1, 1, 2,alignment=Qt.AlignHCenter)
		signal_choice_grid.addWidget(self.target_button2, 2, 0,alignment=Qt.AlignHCenter)
		signal_choice_grid.addWidget(self.effector_button2, 2, 1,alignment=Qt.AlignHCenter)
		signal_choice_grid.addWidget(self.relative_button2, 2, 2,alignment=Qt.AlignHCenter)
		signal_choice_grid.addWidget(self.target_button3, 3, 0,alignment=Qt.AlignHCenter)
		signal_choice_grid.addWidget(self.effector_button3, 3, 1,alignment=Qt.AlignHCenter)
		signal_choice_grid.addWidget(self.relative_button3, 3, 2,alignment=Qt.AlignHCenter)

		signal_choice_vbox = QVBoxLayout()
		signal_choice_vbox.setContentsMargins(30,0,30,50)
		for i in range(len(self.signal_choices)):
			signal_choice_grid.addWidget(self.signal_choices[i],i+1,3)
			# hlayout = QHBoxLayout()
			#
			# #hlayout.addLayout(self.signal_labels[i], 20)
			# #hlayout.addLayout(self.signal_choices[i], 75)
			# # if i==0:
			# #     hlayout.addWidget(self.signal_choices[i], 75,alignment=Qt.AlignBottom)
			# # else:
			# hlayout.addWidget(self.signal_choices[i], 75)
			# #hlayout.addWidget(self.log_btns[i], 5)
			# signal_choice_vbox.addLayout(hlayout)

		self.target_button1.clicked.connect(self.signal_button_changed1)
		self.effector_button1.clicked.connect(self.signal_button_changed1)
		self.relative_button1.clicked.connect(self.signal_button_changed1)

		self.target_button2.clicked.connect(self.signal_button_changed2)
		self.effector_button2.clicked.connect(self.signal_button_changed2)
		self.relative_button2.clicked.connect(self.signal_button_changed2)

		self.target_button3.clicked.connect(self.signal_button_changed3)
		self.effector_button3.clicked.connect(self.signal_button_changed3)
		self.relative_button3.clicked.connect(self.signal_button_changed3)

			# self.log_btns[i].setIcon(icon(MDI6.math_log,color="black"))
			# self.log_btns[i].setStyleSheet(self.parent.parent.parent.button_select_all)
			# self.log_btns[i].clicked.connect(lambda ch, i=i: self.switch_to_log(i))
		#signal_choice_hbox.addLayout(signal_choice_vbox,alignment=Qt.AlignCenter)
		self.left_panel.addLayout(signal_choice_grid)

		btn_hbox = QHBoxLayout()
		self.save_btn = QPushButton('Save')
		self.save_btn.setStyleSheet(self.button_style_sheet)
		self.save_btn.clicked.connect(self.save_trajectories)
		btn_hbox.addWidget(self.save_btn, 90)

		self.export_btn = QPushButton('')
		self.export_btn.setStyleSheet(self.button_select_all)
		self.export_btn.clicked.connect(self.export_signals)
		self.export_btn.setIcon(icon(MDI6.export,color="black"))
		self.export_btn.setIconSize(QSize(25, 25))
		btn_hbox.addWidget(self.export_btn, 10)
		self.left_panel.addLayout(btn_hbox)

		# Animation
		animation_buttons_box = QHBoxLayout()


		animation_buttons_box.addWidget(self.frame_lbl, 20, alignment=Qt.AlignLeft)

		self.first_frame_btn = QPushButton()
		self.first_frame_btn.clicked.connect(self.set_first_frame)
		self.first_frame_btn.setShortcut(QKeySequence('f'))
		self.first_frame_btn.setIcon(icon(MDI6.page_first,color="black"))
		self.first_frame_btn.setStyleSheet(self.button_select_all)
		self.first_frame_btn.setFixedSize(QSize(60, 60))
		self.first_frame_btn.setIconSize(QSize(30, 30))



		self.last_frame_btn = QPushButton()
		self.last_frame_btn.clicked.connect(self.set_last_frame)
		self.last_frame_btn.setShortcut(QKeySequence('l'))
		self.last_frame_btn.setIcon(icon(MDI6.page_last,color="black"))
		self.last_frame_btn.setStyleSheet(self.button_select_all)
		self.last_frame_btn.setFixedSize(QSize(60, 60))
		self.last_frame_btn.setIconSize(QSize(30, 30))

		self.stop_btn = QPushButton()
		self.stop_btn.clicked.connect(self.stop)
		self.stop_btn.setIcon(icon(MDI6.stop,color="black"))
		self.stop_btn.setStyleSheet(self.button_select_all)
		self.stop_btn.setFixedSize(QSize(60, 60))
		self.stop_btn.setIconSize(QSize(30, 30))


		self.start_btn = QPushButton()
		self.start_btn.clicked.connect(self.start)
		self.start_btn.setIcon(icon(MDI6.play,color="black"))
		self.start_btn.setFixedSize(QSize(60, 60))
		self.start_btn.setStyleSheet(self.button_select_all)
		self.start_btn.setIconSize(QSize(30, 30))
		self.start_btn.hide()

		animation_buttons_box.addWidget(self.first_frame_btn, 5, alignment=Qt.AlignRight)
		animation_buttons_box.addWidget(self.stop_btn,5, alignment=Qt.AlignRight)
		animation_buttons_box.addWidget(self.start_btn,5, alignment=Qt.AlignRight)
		animation_buttons_box.addWidget(self.last_frame_btn, 5, alignment=Qt.AlignRight)


		self.right_panel.addLayout(animation_buttons_box, 5)


		self.right_panel.addWidget(self.fcanvas, 90)

		if not self.rgb_mode:
			contrast_hbox = QHBoxLayout()
			contrast_hbox.setContentsMargins(150,5,150,5)
			self.contrast_slider = QLabeledDoubleRangeSlider()
			# self.contrast_slider.setSingleStep(0.001)
			# self.contrast_slider.setTickInterval(0.001)
			self.contrast_slider.setOrientation(1)
			print('range: ', [np.nanpercentile(self.stack.flatten(), 0.001), np.nanpercentile(self.stack.flatten(), 99.999)])
			self.contrast_slider.setRange(
				*[np.nanpercentile(self.stack, 0.001), np.nanpercentile(self.stack, 99.999)])
			self.contrast_slider.setValue(
				[np.nanpercentile(self.stack, 1), np.nanpercentile(self.stack, 99.99)])
			self.contrast_slider.valueChanged.connect(self.contrast_slider_action)
			contrast_hbox.addWidget(QLabel('contrast: '))
			contrast_hbox.addWidget(self.contrast_slider,90)
			self.right_panel.addLayout(contrast_hbox, 5)

		# speed_hbox = QHBoxLayout()
		# speed_hbox.setContentsMargins(150,5,150,5)
		# self.interval_slider = QLabeledSlider()
		# self.interval_slider.setSingleStep(1)
		# self.interval_slider.setTickInterval(1)
		# self.interval_slider.setOrientation(1)
		# self.interval_slider.setRange(1, 10000)
		# self.interval_slider.setValue(self.speed)
		# self.interval_slider.valueChanged.connect(self.interval_slider_action)
		# speed_hbox.addWidget(QLabel('interval (ms): '))
		# speed_hbox.addWidget(self.interval_slider,90)
		# self.right_panel.addLayout(speed_hbox, 10)

		#self.selected_populationulate_left_panel()
		#grid.addLayout(self.left_side, 0, 0, 1, 1)

		main_layout.addLayout(self.left_panel, 35)
		main_layout.addLayout(self.right_panel, 65)
		self.button_widget.adjustSize()
		self.compute_status_and_colors_targets()


		self.setCentralWidget(self.button_widget)
		self.show()

		QApplication.processEvents()

	def del_target_event_class(self):

		msgBox = QMessageBox()
		msgBox.setIcon(QMessageBox.Warning)
		msgBox.setText(f"You are about to delete event class {self.target_class_choice_cb.currentText()}. The associated time and\nstatus will also be deleted. Do you still want to proceed?")
		msgBox.setWindowTitle("Warning")
		msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
		returnValue = msgBox.exec()
		if returnValue == QMessageBox.No:
			return None
		else:
			class_to_delete = self.target_class_choice_cb.currentText()
			time_to_delete = class_to_delete.replace('class','t')
			status_to_delete = class_to_delete.replace('class', 'status')
			cols_to_delete = [class_to_delete, time_to_delete, status_to_delete]
			for c in cols_to_delete:
				try:
					self.df_targets = self.df_targets.drop([c], axis=1)
				except Exception as e:
					print(e)
			item_idx = self.target_class_choice_cb.findText(class_to_delete)
			self.target_class_choice_cb.removeItem(item_idx)

	def del_effector_event_class(self):

		msgBox = QMessageBox()
		msgBox.setIcon(QMessageBox.Warning)
		msgBox.setText(f"You are about to delete event class {self.effector_class_choice_cb.currentText()}. The associated time and\nstatus will also be deleted. Do you still want to proceed?")
		msgBox.setWindowTitle("Warning")
		msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
		returnValue = msgBox.exec()
		if returnValue == QMessageBox.No:
			return None
		else:
			class_to_delete = self.effector_class_choice_cb.currentText()
			time_to_delete = class_to_delete.replace('class','t')
			status_to_delete = class_to_delete.replace('class', 'status')
			cols_to_delete = [class_to_delete, time_to_delete, status_to_delete]
			for c in cols_to_delete:
				try:
					self.df_effectors = self.df_effectors.drop([c], axis=1)
				except Exception as e:
					print(e)
			item_idx = self.effector_class_choice_cb.findText(class_to_delete)
			self.effector_class_choice_cb.removeItem(item_idx)

	def del_relative_event_class(self):

		msgBox = QMessageBox()
		msgBox.setIcon(QMessageBox.Warning)
		msgBox.setText(f"You are about to delete event class {self.relative_class_choice_cb.currentText()}. The associated time and\nstatus will also be deleted. Do you still want to proceed?")
		msgBox.setWindowTitle("Warning")
		msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
		returnValue = msgBox.exec()
		if returnValue == QMessageBox.No:
			return None
		else:
			class_to_delete = self.relative_class_choice_cb.currentText()
			time_to_delete = class_to_delete.replace('class','t')
			status_to_delete = class_to_delete.replace('class', 'status')
			cols_to_delete = [class_to_delete, time_to_delete, status_to_delete]
			for c in cols_to_delete:
				try:
					self.df_relative = self.df_relative.drop([c], axis=1)
				except Exception as e:
					print(e)
			item_idx = self.relative_class_choice_cb.findText(class_to_delete)
			self.relative_class_choice_cb.removeItem(item_idx)

	def update_cell_events(self):
		if 'self' in self.neighborhood_choice_cb.currentText():
			try:
				self.neighbor_event_choice_cb.hide()
				self.neigh_lab.hide()
			except:
				pass
			self.reference_event_choice_cb.disconnect()
			self.reference_event_choice_cb.clear()
			if self.reference_population=='targets':
				self.reference_event_choice_cb.addItems(self.target_class_cols)
				self.reference_event_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors_targets)
			else:
				self.reference_event_choice_cb.addItems(self.effector_class_cols)
				self.reference_event_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors_effectors)

		else:
			try:
				self.neighbor_event_choice_cb.show()
				self.neigh_lab.show()
			except:
				pass
			self.reference_event_choice_cb.disconnect()
			self.reference_event_choice_cb.clear()

			if self.reference_population=='targets':
				self.reference_event_choice_cb.addItems(self.target_class_cols)
				self.reference_event_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors_targets)

			else:
				self.reference_event_choice_cb.addItems(self.effector_class_cols)
				self.reference_event_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors_effectors)

			self.neighbor_event_choice_cb.disconnect()
			self.neighbor_event_choice_cb.clear()

			if self.neighbor_population=='targets':
				self.neighbor_event_choice_cb.addItems(self.target_class_cols)
				self.neighbor_event_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors_targets)

			else:
				self.neighbor_event_choice_cb.addItems(self.effector_class_cols)
				self.neighbor_event_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors_effectors)




	def create_new_target_event_class(self):

		# display qwidget to name the event
		self.newClassWidget = QWidget()
		self.newClassWidget.setWindowTitle('Create new event class')

		layout = QVBoxLayout()
		self.newClassWidget.setLayout(layout)
		name_hbox = QHBoxLayout()
		name_hbox.addWidget(QLabel('event name: '), 25)
		self.target_class_name_le = QLineEdit('event')
		name_hbox.addWidget(self.target_class_name_le, 75)
		layout.addLayout(name_hbox)

		class_labels = ['event', 'no event', 'else']
		layout.addWidget(QLabel('prefill: '))
		radio_box = QHBoxLayout()
		self.class_option_rb = [QRadioButton() for i in range(3)]
		for i,c in enumerate(self.class_option_rb):
			if i==0:
				c.setChecked(True)
			c.setText(class_labels[i])
			radio_box.addWidget(c, 33, alignment=Qt.AlignCenter)
		layout.addLayout(radio_box)

		btn_hbox = QHBoxLayout()
		submit_btn = QPushButton('submit')
		cancel_btn = QPushButton('cancel')
		btn_hbox.addWidget(cancel_btn, 50)
		btn_hbox.addWidget(submit_btn, 50)
		layout.addLayout(btn_hbox)

		submit_btn.clicked.connect(self.write_new_target_event_class)
		cancel_btn.clicked.connect(self.close_without_new_class)

		self.newClassWidget.show()
		center_window(self.newClassWidget)


		# Prefill with class value
		# write in table
	def create_new_effector_event_class(self):

		# display qwidget to name the event
		self.newClassWidget = QWidget()
		self.newClassWidget.setWindowTitle('Create new event class')

		layout = QVBoxLayout()
		self.newClassWidget.setLayout(layout)
		name_hbox = QHBoxLayout()
		name_hbox.addWidget(QLabel('event name: '), 25)
		self.effector_class_name_le = QLineEdit('event')
		name_hbox.addWidget(self.effector_class_name_le, 75)
		layout.addLayout(name_hbox)

		class_labels = ['event', 'no event', 'else']
		layout.addWidget(QLabel('prefill: '))
		radio_box = QHBoxLayout()
		self.class_option_rb = [QRadioButton() for i in range(3)]
		for i, c in enumerate(self.class_option_rb):
			if i == 0:
				c.setChecked(True)
			c.setText(class_labels[i])
			radio_box.addWidget(c, 33, alignment=Qt.AlignCenter)
		layout.addLayout(radio_box)

		btn_hbox = QHBoxLayout()
		submit_btn = QPushButton('submit')
		cancel_btn = QPushButton('cancel')
		btn_hbox.addWidget(cancel_btn, 50)
		btn_hbox.addWidget(submit_btn, 50)
		layout.addLayout(btn_hbox)

		submit_btn.clicked.connect(self.write_new_effector_event_class)
		cancel_btn.clicked.connect(self.close_without_new_class)

		self.newClassWidget.show()
		center_window(self.newClassWidget)

	# Prefill with class value
	# write in table
	def create_new_relative_event_class(self):

		# display qwidget to name the event
		self.newClassWidget = QWidget()
		self.newClassWidget.setWindowTitle('Create new event class')

		layout = QVBoxLayout()
		self.newClassWidget.setLayout(layout)
		name_hbox = QHBoxLayout()
		name_hbox.addWidget(QLabel('event name: '), 25)
		self.relative_class_name_le = QLineEdit('event')
		name_hbox.addWidget(self.relative_class_name_le, 75)
		layout.addLayout(name_hbox)

		class_labels = ['event', 'no event', 'else']
		layout.addWidget(QLabel('prefill: '))
		radio_box = QHBoxLayout()
		self.class_option_rb = [QRadioButton() for i in range(3)]
		for i,c in enumerate(self.class_option_rb):
			if i==0:
				c.setChecked(True)
			c.setText(class_labels[i])
			radio_box.addWidget(c, 33, alignment=Qt.AlignCenter)
		layout.addLayout(radio_box)

		btn_hbox = QHBoxLayout()
		submit_btn = QPushButton('submit')
		cancel_btn = QPushButton('cancel')
		btn_hbox.addWidget(cancel_btn, 50)
		btn_hbox.addWidget(submit_btn, 50)
		layout.addLayout(btn_hbox)
		submit_btn.clicked.connect(self.write_new_relative_event_class)
		cancel_btn.clicked.connect(self.close_without_new_class)

		self.newClassWidget.show()
		center_window(self.newClassWidget)

	def write_new_target_event_class(self):


		if self.target_class_name_le.text()=='':
			self.target_class = 'class'
			self.target_time = 't0'
		else:
			self.target_class = 'class_'+self.target_class_name_le.text()
			self.target_time = 't_'+self.target_class_name_le.text()

		if self.target_class in list(self.df_targets.columns):

			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("This event name already exists. If you proceed,\nall annotated data will be rewritten. Do you wish to continue?")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.No:
				return None
			else:
				pass

		fill_option = np.where([c.isChecked() for c in self.class_option_rb])[0][0]
		self.df_targets.loc[:,self.target_class] = fill_option
		if fill_option==0:
			self.df_targets.loc[:,self.target_time] = 0.1
		else:
			self.df_targets.loc[:,self.target_time] = -1

		self.target_class_choice_cb.clear()
		cols = np.array(self.df_targets.columns)
		self.target_class_cols = np.array([c.startswith('class') for c in list(self.df_targets.columns)])
		self.target_class_cols = list(cols[self.target_class_cols])
		self.target_class_cols.remove('class_id')
		self.target_class_cols.remove('class_color')
		self.target_class_choice_cb.addItems(self.target_class_cols)
		idx = self.target_class_choice_cb.findText(self.target_class)
		self.target_class_choice_cb.setCurrentIndex(idx)

		self.newClassWidget.close()

	def write_new_effector_event_class(self):

		if self.effector_class_name_le.text() == '':
			self.effector_class = 'class'
			self.effector_time = 't0'
		else:
			self.effector_class = 'class_' + self.effector_class_name_le.text()
			self.effector_time = 't_' + self.effector_class_name_le.text()

		if self.effector_class in list(self.df_effectors.columns):

			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText(
				"This event name already exists. If you proceed,\nall annotated data will be rewritten. Do you wish to continue?")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.No:
				return None
			else:
				pass

		fill_option = np.where([c.isChecked() for c in self.class_option_rb])[0][0]
		self.df_effectors.loc[:, self.effector_class] = fill_option
		if fill_option == 0:
			self.df_effectors.loc[:, self.effector_time] = 0.1
		else:
			self.df_effectors.loc[:, self.effector_time] = -1

		self.effector_class_choice_cb.clear()
		cols = np.array(self.df_effectors.columns)
		self.effector_class_cols = np.array([c.startswith('class') for c in list(self.df_effectors.columns)])
		self.effector_class_cols = list(cols[self.effector_class_cols])
		self.effector_class_cols.remove('class_id')
		self.effector_class_cols.remove('class_color')
		self.effector_class_choice_cb.addItems(self.effector_class_cols)
		idx = self.effector_class_choice_cb.findText(self.effector_class)
		self.effector_class_choice_cb.setCurrentIndex(idx)

		self.newClassWidget.close()

	def write_new_relative_event_class(self):

		if self.relative_class_name_le.text()=='':
			self.relative_class = 'class'
			self.relative_time = 't0'
		else:
			self.relative_class = 'class_'+self.relative_class_name_le.text()
			self.relative_time = 't_'+self.relative_class_name_le.text()

		if self.relative_class in list(self.df_relative.columns):

			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("This event name already exists. If you proceed,\nall annotated data will be rewritten. Do you wish to continue?")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.No:
				return None
			else:
				pass
		fill_option = np.where([c.isChecked() for c in self.class_option_rb])[0][0]
		self.df_relative.loc[:,self.relative_class] = fill_option
		if fill_option==0:
			self.df_relative.loc[:,self.relative_time] = 0.1
		else:
			self.df_relative.loc[:,self.relative_time] = -1
		self.relative_class_choice_cb.disconnect()
		self.relative_class_choice_cb.clear()
		cols = np.array(self.df_relative.columns)
		self.relative_class_cols = np.array([c.startswith('class') for c in list(self.df_relative.columns)])
		self.relative_class_cols = list(cols[self.relative_class_cols])
		try:
			self.relative_class_cols.remove('class_color')
			self.relative_class_cols.remove('class_id')
		except:
			pass
		self.relative_class_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors_pair)
		self.relative_class_choice_cb.addItems(self.relative_class_cols)
		idx = self.relative_class_choice_cb.findText(self.relative_class)
		self.relative_class_choice_cb.setCurrentIndex(idx)

		self.newClassWidget.close()


	def close_without_new_class(self):
		self.newClassWidget.close()

	def compute_status_and_colors_targets(self):

		if self.df_targets is not None:
			if self.reference_population=='targets':
				self.target_class_name = self.reference_event_choice_cb.currentText()
			elif self.neighbor_population == 'targets':
				self.target_class_name = self.neighbor_event_choice_cb.currentText()
			else:
				self.target_class_name=''
			#self.target_class_name = self.target_class_choice_cb.currentText()
			#self.target_class_name='class_lowPI'
			self.expected_target_status = 'status_'
			suffix = self.target_class_name.replace('class','').replace('_','')
			if suffix!='':
				self.expected_target_status+='_'+suffix
				self.expected_target_time = 't_'+suffix
			else:
				self.expected_target_time = 't0'

			self.time_target_name = self.expected_target_time
			self.target_status_name = self.expected_target_status

			if self.time_target_name in self.df_targets.columns and self.target_class_name in self.df_targets.columns and not self.target_status_name in self.df_targets.columns:
				# only create the status column if it does not exist to not erase static classification results
				self.make_target_status_column()
			elif self.time_target_name in self.df_targets.columns and self.target_class_name in self.df_targets.columns:
				# all good, do nothing
				pass
			else:
				if not self.target_status_name in self.df_targets.columns:
					self.df_targets[self.target_status_name] = 0
					self.df_targets['status_color'] = color_from_status(0)
					self.df_targets['class_color'] = color_from_class(1)

			if not self.target_class_name in self.df_targets.columns:
				self.df_targets[self.target_class_name] = 1
			if not self.time_target_name in self.df_targets.columns:
				self.df_targets[self.time_target_name] = -1

			self.df_targets['status_color'] = [color_from_status(i) for i in self.df_targets[self.target_status_name].to_numpy()]
			self.df_targets['class_color'] = [color_from_class(i) for i in self.df_targets[self.target_class_name].to_numpy()]

			self.extract_scatter_from_target_trajectories()

	def compute_status_and_colors_effectors(self):

		if self.df_effectors is not None:
			if self.reference_population=='effectors':
				self.effector_class_name = self.reference_event_choice_cb.currentText()
			elif self.neighbor_population == 'effectors':
				self.effector_class_name = self.neighbor_event_choice_cb.currentText()
			else:
				self.effector_class_name=''
			#self.effector_class_name = self.effector_class_choice_cb.currentText()
			self.effector_expected_status = 'status'
			suffix = self.effector_class_name.replace('class','').replace('_','')
			if suffix!='':
				self.effector_expected_status+='_'+suffix
				self.effector_expected_time = 't_'+suffix
			else:
				self.effector_expected_time = 't0'

			self.effector_time_name = self.effector_expected_time
			self.effector_status_name = self.effector_expected_status

			print('selection and expected names: ', self.effector_class_name, self.effector_expected_time, self.effector_expected_status)

			if self.effector_time_name in self.df_effectors.columns and self.effector_class_name in self.df_effectors.columns and not self.effector_status_name in self.df_effectors.columns:
				# only create the status column if it does not exist to not erase static classification results
				self.make_effector_status_column()
			elif self.effector_time_name in self.df_effectors.columns and self.effector_class_name in self.df_effectors.columns:
				# all good, do nothing
				pass
			else:
				if not self.effector_status_name in self.df_effectors.columns:
					self.df_effectors[self.effector_status_name] = 0
					self.df_effectors['status_color'] = color_from_status(0)
					self.df_effectors['class_color'] = color_from_class(1)

			if not self.effector_class_name in self.df_effectors.columns:
				self.df_effectors[self.effector_class_name] = 1
			if not self.effector_time_name in self.df_effectors.columns:
				self.df_effectors[self.effector_time_name] = -1

			self.df_effectors['status_color'] = [color_from_status(i) for i in self.df_effectors[self.effector_status_name].to_numpy()]
			self.df_effectors['class_color'] = [color_from_class(i) for i in self.df_effectors[self.effector_class_name].to_numpy()]

			self.extract_scatter_from_effector_trajectories()

	def compute_status_and_colors_pair(self):

		self.pair_class_name=self.relative_class_choice_cb.currentText()
		self.pair_expected_status = 'status'
		suffix = self.pair_class_name.replace('class','').replace('_','',1)
		if suffix!='':
			self.pair_expected_status+='_'+suffix
			self.pair_expected_time = 't0_'+suffix
		else:
			self.pair_expected_time = 't0'

		self.pair_time_name = self.pair_expected_time
		self.pair_status_name = self.pair_expected_status

		print('selection and expected names: ', self.pair_class_name, self.pair_expected_time, self.pair_expected_status)

		#self.df_relative['status_color'] = [color_from_status(i) for i in self.df_relative[self.pair_status_name].to_numpy()]
		#self.df_effectors['class_color'] = [color_from_class(i) for i in self.df_effectors[self.effector_class_name].to_numpy()]
		if self.pair_time_name in self.df_relative.columns and self.pair_class_name in self.df_relative.columns and not self.pair_status_name in self.df_relative.columns:
			# only create the status column if it does not exist to not erase static classification results
			self.make_relative_status_column()
		elif self.pair_time_name in self.df_relative.columns and self.pair_class_name in self.df_relative.columns:
			# all good, do nothing
			pass
		else:
			if not self.pair_status_name in self.df_relative.columns:
				self.df_relative[self.pair_status_name] = 0
				self.df_relative['status_color'] = color_from_status(0)
				self.df_relative['class_color'] = color_from_class(1)

		if not self.pair_class_name in self.df_relative.columns:
			self.df_relative[self.pair_time_name] = 1
		if not self.pair_time_name in self.df_relative.columns:
			self.df_relative[self.pair_time_name] = -1

		self.df_relative['status_color'] = [color_from_status(i) for i in self.df_relative[self.pair_status_name].to_numpy()]
		self.df_relative['class_color'] = [color_from_class(i) for i in self.df_relative[self.pair_class_name].to_numpy()]

		self.extract_scatter_from_lines()
	def contrast_slider_action(self):

		"""
		Recontrast the imshow as the contrast slider is moved.
		"""

		self.vmin = self.contrast_slider.value()[0]
		self.vmax = self.contrast_slider.value()[1]
		self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
		self.fcanvas.canvas.draw_idle()


	def cancel_selection(self):

		print('Canceling selection...')

		self.hide_annotation_buttons()
		self.correct_btn.setEnabled(False)
		self.correct_btn.setText('correct')
		self.cancel_btn.setEnabled(False)
		self.correct_btn.disconnect()
		self.correct_btn.clicked.connect(self.show_annotation_buttons)

		self.reference_selection = []

		if len(self.pair_selection) > 0:
			self.cancel_pair_selection()

		if self.df_targets is not None:
			self.target_selection = []
		if self.df_effectors is not None:
			self.effector_selection = []

		if self.df_targets is not None:
			for k,(t,idx) in enumerate(zip(self.target_loc_t,self.target_loc_idx)):
				self.target_colors[t][idx, 0] = self.initial_target_colors[k][0]
				self.target_colors[t][idx, 1] = self.initial_target_colors[k][1]

			for (t,idx) in (zip(self.target_loc_t_not_picked,self.target_loc_idx_not_picked)):
				self.target_colors[t][idx, 0] = self.initial_target_colors[t][idx,0]
				self.target_colors[t][idx, 1] = self.initial_target_colors[t][idx,1]

			for t in range(len(self.target_colors)):
				for ind in range(len(self.target_colors[t])):
					self.target_colors[t][ind] = self.initial_target_colors[t][ind]

		if self.df_effectors is not None:

			for k,(t,idx) in enumerate(zip(self.reference_loc_t,self.reference_loc_idx)):
				self.effector_colors[t][idx,0] = self.initial_effector_colors[t][idx,0]
				self.effector_colors[t][idx,1] = self.initial_effector_colors[t][idx,1]

			for (t,idx) in (zip(self.reference_loc_t_not_picked,self.reference_loc_idx_not_picked)):
				self.effector_colors[t][idx, 0] = self.initial_effector_colors[t][idx,0]
				self.effector_colors[t][idx, 1] = self.initial_effector_colors[t][idx,1]

			for t in range(len(self.effector_colors)):
				for ind in range(len(self.effector_colors[t])):
					self.effector_colors[t][ind] = self.initial_effector_colors[t][ind]

		self.lines_data={}
		self.lines_list=[]
		self.lines_plot=[]

		self.selected_population = None


	def hide_annotation_buttons(self):

		for a in self.annotation_btns_to_hide:
			a.hide()
		for b in [self.event_btn, self.no_event_btn, self.else_btn, self.suppr_btn]:
			b.setChecked(False)
		self.time_of_interest_label.setEnabled(False)
		self.time_of_interest_le.setText('')
		self.time_of_interest_le.setEnabled(False)


	def enable_time_of_interest(self):

		if self.event_btn.isChecked():
			self.time_of_interest_label.setEnabled(True)
			self.time_of_interest_le.setEnabled(True)
		else:
			self.time_of_interest_label.setEnabled(False)
			self.time_of_interest_le.setEnabled(False)

	def cancel_pair_selection(self):

		# Unselect and recolor pair line
		self.pair_selection = []
		for t in range(len(self.lines_colors_status)):
			for idx in range(len(self.lines_colors_status[t])):
				if self.lines_colors_status[t][idx,2] == 'lime':
					self.lines_colors_status[t][idx,2]=self.initial_lines_colors_status[t][idx,2]
					self.lines_colors_class[t][idx,2]=self.initial_lines_colors_class[t][idx,2]

		# Unselect and recolor neighbor
		_, _, colors_neigh = self.get_neighbor_sets()
		for k,(t,idx) in enumerate(zip(self.neigh_cell_loc_t,self.neigh_cell_loc_idx)):
			colors_neigh[t][idx, 0] = self.neigh_previous_color[k][0]
			colors_neigh[t][idx, 1] = self.neigh_previous_color[k][1]

	def apply_modification(self):

		t0 = -1
		if self.event_btn.isChecked():
			cclass = 0
			try:
				t0 = float(self.time_of_interest_le.text().replace(',', '.'))
				self.line_dt.set_xdata([t0, t0])
				self.cell_fcanvas.canvas.draw_idle()
			except Exception as e:
				print(e)
				t0 = -1
				cclass = 2
		elif self.no_event_btn.isChecked():
			cclass = 1
		elif self.else_btn.isChecked():
			cclass = 2
		elif self.suppr_btn.isChecked():
			cclass = 42

		self.df_relative.loc[(self.df_relative['REFERENCE_ID'] == self.reference_track_of_interest)&(self.df_relative['NEIGHBOR_ID']==self.neighbor_track_of_interest), self.pair_class_name] = cclass
		self.df_relative.loc[(self.df_relative['REFERENCE_ID'] == self.reference_track_of_interest)&(self.df_relative['NEIGHBOR_ID']==self.neighbor_track_of_interest), self.pair_time_name] = t0

		indices = self.df_relative.loc[(self.df_relative['REFERENCE_ID'] == self.reference_track_of_interest)&(self.df_relative['NEIGHBOR_ID']==self.neighbor_track_of_interest), self.pair_class_name].index
		timeline = self.df_relative.loc[(self.df_relative['REFERENCE_ID'] == self.reference_track_of_interest)&(self.df_relative['NEIGHBOR_ID']==self.neighbor_track_of_interest), self.pair_time_name] .to_numpy()
		status = np.zeros_like(timeline)
		if t0 > 0:
			status[timeline >= t0] = 1.
		if cclass == 2:
			status[:] = 2
		if cclass > 2:
			status[:] = 42
		status_color = [color_from_status(s, recently_modified=True) for s in status]
		class_color = [color_from_class(cclass, recently_modified=True) for i in range(len(status))]

		self.df_relative.loc[indices, self.pair_status_name] = status
		self.df_relative.loc[indices, 'status_color'] = status_color
		self.df_relative.loc[indices, 'class_color'] = class_color

		# self.make_status_column()
		self.extract_scatter_from_lines()
		self.give_reference_cell_information()
		self.give_neighbor_cell_information()

		self.correct_btn.disconnect()
		self.correct_btn.clicked.connect(self.show_annotation_buttons)
			# self.cancel_btn.click()

		self.hide_annotation_buttons()
		self.correct_btn.setEnabled(False)
		self.correct_btn.setText('correct')
		self.cancel_btn.setEnabled(False)
		self.del_shortcut.setEnabled(False)
		self.no_event_shortcut.setEnabled(False)

		self.pair_selection=[]
		try:
			self.target_selection.pop(0)
		except:
			pass
		try:
			self.effector_selection.pop(0)
		except:
			pass


		#self.make_status_column()
		self.extract_scatter_from_target_trajectories()
		self.extract_scatter_from_effector_trajectories()
		#self.give_target_cell_information()
		#self.give_effector_cell_information()


		self.correct_btn.disconnect()
		self.correct_btn.clicked.connect(self.show_annotation_buttons)
		#self.cancel_btn.click()

		self.hide_annotation_buttons()
		self.correct_btn.setEnabled(False)
		self.correct_btn.setText('correct')
		self.cancel_btn.setEnabled(False)
		self.del_shortcut.setEnabled(False)
		self.no_event_shortcut.setEnabled(False)

		try:
			self.target_selection.pop(0)
			self.effector_selection.pop(0)
		except Exception as e:
			print(e)


		#self.fcanvas.canvas.draw()


	def locate_stack(self):

		"""
		Locate the target movie.

		"""

		movies = glob(self.pos + f"movie/{self.parent_window.parent_window.movie_prefix}*.tif")

		if len(movies)==0:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("No movies are detected in the experiment folder. Cannot load an image to test Haralick.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Yes:
				self.close()
		else:
			self.stack_path = movies[0]
			self.len_movie = self.parent_window.parent_window.len_movie
			len_movie_auto = auto_load_number_of_frames(self.stack_path)
			if len_movie_auto is not None:
				self.len_movie = len_movie_auto
			exp_config = self.exp_dir +"config.ini"
			self.channel_names, self.channels = extract_experiment_channels(exp_config)
			self.channel_names = np.array(self.channel_names)
			self.channels = np.array(self.channels)
			self.nbr_channels = len(self.channels)

	def locate_target_tracks(self):

		population = 'targets'
		self.target_trajectories_path = self.pos + os.sep.join(['output','tables', f'trajectories_{population}.pkl'])

		if not os.path.exists(self.target_trajectories_path):

			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("The target trajectories cannot be detected...")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			self.df_targets = None

		else:

			# Load and prep tracks
			self.df_targets = np.load(self.target_trajectories_path, allow_pickle=True)
			self.df_targets = self.df_targets.sort_values(by=['TRACK_ID', 'FRAME'])

			cols = np.array(self.df_targets.columns)
			self.target_class_cols = [c for c in list(self.df_targets.columns) if c.startswith('class')]

			try:
				self.target_class_cols.remove('class_id')
			except:
				pass
			try:
				self.target_class_cols.remove('class_color')
			except:
				pass

			if len(self.target_class_cols)>0:

				self.target_class_name = self.target_class_cols[0]
				self.target_expected_status = 'status'
				suffix = self.target_class_name.replace('class','').replace('_','')
				if suffix!='':
					self.target_expected_status+='_'+suffix
					self.target_expected_time = 't_'+suffix
				else:
					self.target_expected_time = 't0'
				self.target_time_name = self.target_expected_time
				self.target_status_name = self.target_expected_status
			else:
				self.target_class_name = 'class'
				self.target_time_name = 't0'
				self.target_status_name = 'status'

			if self.target_time_name in self.df_targets.columns and self.target_class_name in self.df_targets.columns and not self.target_status_name in self.df_targets.columns:
				# only create the status column if it does not exist to not erase static classification results
				self.make_target_status_column()
			elif self.target_time_name in self.df_targets.columns and self.target_class_name in self.df_targets.columns:
				# all good, do nothing
				pass
			else:
				if not self.target_status_name in self.df_targets.columns:
					self.df_targets[self.target_status_name] = 0
					self.df_targets['status_color'] = color_from_status(0)
					self.df_targets['class_color'] = color_from_class(1)

			if not self.target_class_name in self.df_targets.columns:
				self.df_targets[self.target_class_name] = 1
			if not self.target_time_name in self.df_targets.columns:
				self.df_targets[self.target_time_name] = -1

			self.df_targets['status_color'] = [color_from_status(i) for i in self.df_targets[self.target_status_name].to_numpy()]
			self.df_targets['class_color'] = [color_from_class(i) for i in self.df_targets[self.target_class_name].to_numpy()]


			self.df_targets = self.df_targets.dropna(subset=['POSITION_X', 'POSITION_Y'])
			self.df_targets['x_anim'] = self.df_targets['POSITION_X'] * self.fraction
			self.df_targets['y_anim'] = self.df_targets['POSITION_Y'] * self.fraction
			self.df_targets['x_anim'] = self.df_targets['x_anim'].astype(int)
			self.df_targets['y_anim'] = self.df_targets['y_anim'].astype(int)

			self.extract_scatter_from_target_trajectories()
			self.target_track_of_interest = self.df_targets['TRACK_ID'].min()

			self.loc_t = []
			self.loc_idx = []
			for t in range(len(self.target_tracks)):
				indices = np.where(self.target_tracks[t]==self.target_track_of_interest)[0]
				if len(indices)>0:
					self.loc_t.append(t)
					self.loc_idx.append(indices[0])


	def locate_effector_tracks(self):

		population = 'effectors'
		self.effector_trajectories_path =  self.pos + os.sep.join(['output','tables',f'trajectories_{population}.pkl'])

		if not os.path.exists(self.effector_trajectories_path):

			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("The effector trajectories cannot be detected...")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			self.df_effectors = None
		else:

			# Load and prep tracks
			self.df_effectors = np.load(self.effector_trajectories_path, allow_pickle=True)
			try:
				self.df_effectors = self.df_effectors.sort_values(by=['TRACK_ID', 'FRAME'])
			except:
				self.df_effectors = self.df_effectors.sort_values(by=['ID', 'FRAME'])


			cols = np.array(self.df_effectors.columns)
			self.effector_class_cols = np.array([c.startswith('class') for c in list(self.df_effectors.columns)])
			self.effector_class_cols = list(cols[self.effector_class_cols])
			try:
				self.effector_class_cols.remove('class_id')
			except:
				pass
			try:
				self.effector_class_cols.remove('class_color')
			except:
				pass
			if len(self.effector_class_cols)>0:
				self.effector_class_name = self.effector_class_cols[0]
				self.effector_expected_status = 'status'
				suffix = self.effector_class_name.replace('class','').replace('_','')
				if suffix!='':
					self.effector_expected_status+='_'+suffix
					self.effector_expected_time = 't_'+suffix
				else:
					self.effector_expected_time = 't0'
				self.effector_time_name = self.effector_expected_time
				self.effector_status_name = self.effector_expected_status
			else:
				self.effector_class_name = 'class'
				self.effector_time_name = 't0'
				self.effector_status_name = 'status'

			if self.effector_time_name in self.df_effectors.columns and self.effector_class_name in self.df_effectors.columns and not self.effector_status_name in self.df_effectors.columns:
				# only create the status column if it does not exist to not erase static classification results
				self.make_effector_status_column()
			elif self.effector_time_name in self.df_effectors.columns and self.effector_class_name in self.df_effectors.columns:
				# all good, do nothing
				pass
			else:
				if not self.effector_status_name in self.df_effectors.columns:
					self.df_effectors[self.effector_status_name] = 0
					self.df_effectors['status_color'] = color_from_status(0)
					self.df_effectors['class_color'] = color_from_class(1)

			if not self.effector_class_name in self.df_effectors.columns:
				self.df_effectors[self.effector_class_name] = 1
			if not self.effector_time_name in self.df_effectors.columns:
				self.df_effectors[self.effector_time_name] = -1

			self.df_effectors['status_color'] = [color_from_status(i) for i in self.df_effectors[self.effector_status_name].to_numpy()]
			self.df_effectors['class_color'] = [color_from_class(i) for i in self.df_effectors[self.effector_class_name].to_numpy()]


			self.df_effectors = self.df_effectors.dropna(subset=['POSITION_X', 'POSITION_Y'])
			self.df_effectors['x_anim'] = self.df_effectors['POSITION_X'] * self.fraction
			self.df_effectors['y_anim'] = self.df_effectors['POSITION_Y'] * self.fraction
			self.df_effectors['x_anim'] = self.df_effectors['x_anim'].astype(int)
			self.df_effectors['y_anim'] = self.df_effectors['y_anim'].astype(int)

			self.extract_scatter_from_effector_trajectories()
			try:
				self.effector_track_of_interest = self.df_effectors['TRACK_ID'].min()
			except:
				self.effector_track_of_interest = self.df_effectors['ID'].min()


			self.loc_t = []
			self.loc_idx = []
			for t in range(len(self.effector_tracks)):
				indices = np.where(self.effector_tracks[t]==self.effector_track_of_interest)[0]
				if len(indices)>0:
					self.loc_t.append(t)
					self.loc_idx.append(indices[0])

			# self.MinMaxScaler_effectors = MinMaxScaler()
			# self.columns_to_rescale_effectors = list(self.df_effectors.columns)

			# # is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
			# # is_number_test = is_number(self.df_tracks.dtypes)
			# # self.columns_to_rescale = [col for t,col in zip(is_number_test,self.df_tracks.columns) if t]
			# # print(self.columns_to_rescale)

			# cols_to_remove = ['status','status_color','class_color','TRACK_ID','ID', 'FRAME','x_anim','y_anim','t', 'state', 'generation', 'root', 'parent', 'class_id', 'class', 't0', 'POSITION_X', 'POSITION_Y','position','well','well_index','well_name','pos_name','index','concentration','cell_type','antibody','pharmaceutical_agent'] + self.effector_class_cols
			# cols = np.array(list(self.df_effectors.columns))
			# time_cols = np.array([c.startswith('t_') for c in cols])
			# time_cols = list(cols[time_cols])
			# cols_to_remove += time_cols

			# for tr in cols_to_remove:
			# 	try:
			# 		self.columns_to_rescale_effectors.remove(tr)
			# 	except:
			# 		pass
			# 		#print(f'column {tr} could not be found...')

			# x = self.df_effectors[self.columns_to_rescale_effectors].values
			# self.MinMaxScaler_effectors.fit(x)

			# #self.loc_t, self.loc_idx = np.where(self.tracks==self.track_of_interest)

	def locate_relative_tracks(self):

		population = 'relative'
		self.relative_trajectories_path = self.pos + os.sep.join(['output','tables','cell_pair_measurements.csv'])

		if not os.path.exists(self.relative_trajectories_path):

			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("The pair measurements cannot be detected... Please measure the pairs first.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			self.close()
		else:
			# Load and prep tracks
			self.df_relative = pd.read_csv(self.relative_trajectories_path)
			print(self.df_relative.columns)
			self.df_relative= self.df_relative.sort_values(by=['REFERENCE_ID','NEIGHBOR_ID','reference_population','neighbor_population','FRAME'])
			self.relative_cols = np.array(self.df_relative.columns)

		self.columns_to_rescale_relative = list(self.df_relative.columns)


		# is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
		# is_number_test = is_number(self.df_tracks.dtypes)
		# self.columns_to_rescale = [col for t,col in zip(is_number_test,self.df_tracks.columns) if t]
		# print(self.columns_to_rescale)

		cols_to_remove = ['REFERENCE_ID','NEIGHBOR_ID','FRAME','t0_arrival','ref_population','class_color', 'status_color','status','t0']
		cols = np.array(list(self.df_relative.columns))
		time_cols = np.array([c.startswith('t_') for c in cols])
		time_cols = list(cols[time_cols])
		cols_to_remove += time_cols
		self.relative_class_cols = np.array([c.startswith('class') for c in list(self.df_relative.columns)])
		# self.neighborhood_cols = np.array([c.startswith('neighborhood') for c in list(self.df_relative.columns)])

		# try:
		# 	self.neighborhood_cols = list(cols[self.neighborhood_cols])
		# except:
		# 	pass
		try:
			self.relative_class_cols = list(cols[self.relative_class_cols])
		except:
			pass
		try:
			self.relative_lass_cols.remove('class_id')
		except:
			pass
		try:
			self.relative_class_cols.remove('class_color')
		except:
			pass

		for tr in cols_to_remove:
			try:
				self.columns_to_rescale_relative.remove(tr)
			except:
				pass

		if len(self.relative_class_cols) > 0:
			self.relative_class_name = self.relative_class_cols[0]
			self.relative_expected_status = 'status'
			suffix = self.relative_class_name.replace('class', '').replace('_', '')
			if suffix != '':
				self.relative_expected_status += '_' + suffix
				self.relative_expected_time = 't_' + suffix
			else:
				self.relative_expected_time = 't0_arrival'
			self.relative_time_name = self.relative_expected_time
			self.relative_status_name = self.relative_expected_status
		else:
			self.relative_class_name = 'class'
			self.relative_time_name = 't0'
			self.relative_status_name = 'status'
		# print(f'column {tr} could not be found...')
			# for col in self.relative_cols:
			# 	print(col)
			# 	if col=="relxy":
			# 		self.relative_cols.remove(col)
			#self.target_class_cols = np.array([c.startswith('class') for c in list(self.df_targets.columns)])
			#self.relative_cols = list(cols[self.target_class_cols])
			#try:
				#self..remove('class_id')
			#except:
				#pass
			#try:
				#self.target_class_cols.remove('class_color')
			#except:
				#pass
			# if len(self.target_class_cols) > 0:
			# 	self.target_class_name = self.target_class_cols[0]
			# 	self.target_expected_status = 'status'
			# 	suffix = self.target_class_name.replace('class', '').replace('_', '')
			# 	if suffix != '':
			# 		self.target_expected_status += '_' + suffix
			# 		self.target_expected_time = 't_' + suffix
			# 	else:
			# 		self.target_expected_time = 't0'
			# 	self.target_time_name = self.target_expected_time
			# 	self.target_status_name = self.target_expected_status
			# else:
			# 	self.target_class_name = 'class'
			# 	self.target_time_name = 't0'
			# 	self.target_status_name = 'status'
			#
			# if self.target_time_name in self.df_targets.columns and self.target_class_name in self.df_targets.columns and not self.target_status_name in self.df_targets.columns:
			# 	# only create the status column if it does not exist to not erase static classification results
			# 	self.make_target_status_column()
			# elif self.target_time_name in self.df_targets.columns and self.target_class_name in self.df_targets.columns:
			# 	# all good, do nothing
			# 	pass
			# else:
			# 	if not self.target_status_name in self.df_targets.columns:
			# 		self.df_targets[self.target_status_name] = 0
			# 		self.df_targets['status_color'] = color_from_status(0)
			# 		self.df_targets['class_color'] = color_from_class(1)
			#
			# if not self.target_class_name in self.df_targets.columns:
			# 	self.df_targets[self.target_class_name] = 1
			# if not self.target_time_name in self.df_targets.columns:
			# 	self.df_targets[self.target_time_name] = -1
			#
			# self.df_targets['status_color'] = [color_from_status(i) for i in
			# 								   self.df_targets[self.target_status_name].to_numpy()]
			# self.df_targets['class_color'] = [color_from_class(i) for i in
			# 								  self.df_targets[self.target_class_name].to_numpy()]
			#
			# self.df_targets = self.df_targets.dropna(subset=['POSITION_X', 'POSITION_Y'])
			# self.df_targets['x_anim'] = self.df_targets['POSITION_X'] * self.fraction
			# self.df_targets['y_anim'] = self.df_targets['POSITION_Y'] * self.fraction
			# self.df_targets['x_anim'] = self.df_targets['x_anim'].astype(int)
			# self.df_targets['y_anim'] = self.df_targets['y_anim'].astype(int)
			#
			# self.extract_scatter_from_target_trajectories()
			# self.target_track_of_interest = self.df_targets['TRACK_ID'].min()
			#
			# self.loc_t = []
			# self.loc_idx = []
			# for t in range(len(self.target_tracks)):
			# 	indices = np.where(self.target_tracks[t] == self.target_track_of_interest)[0]
			# 	if len(indices) > 0:
			# 		self.loc_t.append(t)
			# 		self.loc_idx.append(indices[0])
			#
			# self.MinMaxScaler_targets = MinMaxScaler()
			# self.columns_to_rescale_targets = list(self.df_targets.columns)

			# is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
			# is_number_test = is_number(self.df_tracks.dtypes)
			# self.columns_to_rescale = [col for t,col in zip(is_number_test,self.df_tracks.columns) if t]
			# print(self.columns_to_rescale)

			#cols_to_remove = ['relxy']
			#cols = np.array(list(self.df_relative.columns))
			# time_cols = np.array([c.startswith('t_') for c in cols])
			# time_cols = list(cols[time_cols])
			# cols_to_remove += time_cols
			#
			# for tr in cols_to_remove:
			# 	try:
			# 		self.columns_to_rescale_targets.remove(tr)
			# 	except:
			# 		pass
			# # print(f'column {tr} could not be found...')
			#
			# x = self.df_targets[self.columns_to_rescale_targets].values
			# self.MinMaxScaler_targets.fit(x)

	# self.loc_t, self.loc_idx = np.where(self.tracks==self.track_of_interest)


	def set_reference_and_neighbor_populations(self):

		neigh = self.neighborhood_choice_cb.currentText()
		self.reference_population = self.df_relative.loc[~self.df_relative['status_'+neigh].isnull(), 'reference_population'].values[0]
		self.neighbor_population = self.df_relative.loc[~self.df_relative['status_'+neigh].isnull(), 'neighbor_population'].values[0]
		print(f'New reference population: {self.reference_population}')
		print(f'New neighbor population: {self.neighbor_population}')

		idx = self.relative_class_choice_cb.findText('class_'+neigh)
		if idx is not None:
			self.relative_class_choice_cb.setCurrentIndex(idx)

	def make_target_status_column(self):

		print('remaking the status column')
		for tid, group in self.df_targets.groupby('TRACK_ID'):

			indices = group.index
			t0 = group[self.target_time_name].to_numpy()[0]
			cclass = group[self.target_class_name].to_numpy()[0]
			timeline = group['FRAME'].to_numpy()
			status = np.zeros_like(timeline)
			if t0 > 0:
				status[timeline>=t0] = 1.
			if cclass==2:
				status[:] = 2
			if cclass>2:
				status[:] = 42
			status_color = [color_from_status(s) for s in status]
			class_color = [color_from_class(cclass) for i in range(len(status))]

			self.df_targets.loc[indices, self.target_status_name] = status
			self.df_targets.loc[indices, 'status_color'] = status_color
			self.df_targets.loc[indices, 'class_color'] = class_color

	def make_relative_status_column(self):
		# if 'TRACK_ID' in self.df_effectors.columns:
		#     id_type="TRACK_ID"
		# else:
		#     id_type="ID"
		print('remaking the status column')
		filtered=self.df_relative.loc(self.df_relative[f'{self.neighborhood_choice_cb.currentText()}'==1,])
		for tid, group in filtered.groupby('REFERENCE_ID'):

			indices = group.index
			t0 = group[self.pair_time_name].to_numpy()[0]
			cclass = group[self.pair_class_name].to_numpy()[0]
			timeline = group['FRAME'].to_numpy()
			status = np.zeros_like(timeline)
			if t0 > 0:
				status[timeline>=t0] = 1.
			if cclass==2:
				status[:] = 2
			if cclass>2:
				status[:] = 42
			status_color = [color_from_status(s) for s in status]
			class_color = [color_from_class(cclass) for i in range(len(status))]

			self.df_relative.loc[indices, self.pair_status_name] = status
			self.df_relative.loc[indices, 'status_color'] = status_color
			self.df_relative.loc[indices, 'class_color'] = class_color

	def make_effector_status_column(self):
		if 'TRACK_ID' in self.df_effectors.columns:
			id_type="TRACK_ID"
		else:
			id_type="ID"
		print('remaking the status column')
		for tid, group in self.df_effectors.groupby(id_type):

			indices = group.index
			t0 = group[self.effector_time_name].to_numpy()[0]
			cclass = group[self.effector_class_name].to_numpy()[0]
			timeline = group['FRAME'].to_numpy()
			status = np.zeros_like(timeline)
			if t0 > 0:
				status[timeline>=t0] = 1.
			if cclass==2:
				status[:] = 2
			if cclass>2:
				status[:] = 42
			status_color = [color_from_status(s) for s in status]
			class_color = [color_from_class(cclass) for i in range(len(status))]

			self.df_effectors.loc[indices, self.effector_status_name] = status
			self.df_effectors.loc[indices, 'status_color'] = status_color
			self.df_effectors.loc[indices, 'class_color'] = class_color

	def generate_signal_choices(self):

		self.signal_choices = []
		self.signal_labels = []
		if self.df_targets is not None:
			self.target_signals = list(self.df_targets.columns)
		else:
			self.target_signals = []
		if self.df_effectors is not None:
			self.effector_signals = list(self.df_effectors.columns)
		else:
			self.effector_signals = []
		self.relative_signals = list(self.relative_cols)
		to_remove = ['REFERENCE_ID', 'NEIGHBOR_ID', 'FRAME', 't0_arrival', 'TRACK_ID', 'class_color', 'status_color',
					 'FRAME', 'x_anim', 'y_anim', 't', 'state', 'generation', 'root', 'parent', 'class_id', 'class',
					 't0', 'POSITION_X', 'POSITION_Y', 'position', 'well', 'well_index', 'well_name', 'pos_name',
					 'index', 'relxy', 'tc', 'nk']

		for c in to_remove:
			if self.df_targets is not None:
				if c in self.target_signals:
					self.target_signals.remove(c)
			if self.df_effectors is not None:
				if c in self.effector_signals:
					self.effector_signals.remove(c)
			if c in self.relative_signals:
				self.relative_signals.remove(c)

		self.signal1 = QButtonGroup()
		self.signal2 = QButtonGroup()
		self.signal3 = QButtonGroup()

		self.target_signal_choice = QSearchableComboBox()
		self.effector_signal_choice = QSearchableComboBox()
		self.relative_signal_choice = QSearchableComboBox()
		self.target_buttons_layout = QVBoxLayout()
		self.effector_buttons_layout = QVBoxLayout()
		self.relative_buttons_layout = QVBoxLayout()
		self.signal_choice_names = QHBoxLayout()

		self.signal_buttons0 = QHBoxLayout()
		self.signal_buttons1 = QHBoxLayout()
		self.signal_buttons2 = QHBoxLayout()
		self.signal_buttons3 = QHBoxLayout()

		self.target_button1 = QRadioButton()
		self.effector_button1 = QRadioButton()
		self.relative_button1 = QRadioButton()

		self.target_button2 = QRadioButton()
		self.effector_button2 = QRadioButton()
		self.relative_button2 = QRadioButton()

		self.target_button3 = QRadioButton()
		self.effector_button3 = QRadioButton()
		self.relative_button3 = QRadioButton()

		self.signal1.addButton(self.target_button1)
		self.signal1.addButton(self.effector_button1)
		self.signal1.addButton(self.relative_button1)

		self.signal2.addButton(self.target_button2)
		self.signal2.addButton(self.effector_button2)
		self.signal2.addButton(self.relative_button2)

		self.signal3.addButton(self.target_button3)
		self.signal3.addButton(self.effector_button3)
		self.signal3.addButton(self.relative_button3)

		self.target_signal_choice_label = QLabel('target signal')
		self.effector_signal_choice_label = QLabel('effector signal')
		self.relative_signal_choice_label = QLabel('relative signal')

		self.target_signal_choice.addItems(['--'] + self.target_signals)
		self.effector_signal_choice.addItems(['--'] + self.effector_signals)
		self.relative_signal_choice.addItems(['--'] + self.relative_signals)

		self.effector_signal_choice.setCurrentIndex(1)
		self.target_signal_choice.setCurrentIndex(1)

		self.target_signal_choice.currentIndexChanged.connect(self.plot_signals)
		self.effector_signal_choice.currentIndexChanged.connect(self.plot_signals)
		self.relative_signal_choice.currentIndexChanged.connect(self.plot_signals)

		self.signal_choices.append(self.target_signal_choice)
		self.signal_choices.append(self.effector_signal_choice)
		self.signal_choices.append(self.relative_signal_choice)
		self.target_button1.setChecked(True)
		self.effector_button2.setChecked(True)
		self.relative_button3.setChecked(True)

		self.signal_labels.append(self.target_buttons_layout)
		self.signal_labels.append(self.effector_buttons_layout)
		self.signal_labels.append(self.relative_buttons_layout)




		# # EFFECTORS
		# self.signal_choice_effectors_cb = [QComboBox() for i in range(self.n_signals)]
		# self.signal_choice_effectors_label = [QLabel(f'signal {i+1}: ') for i in range(self.n_signals)]
		# #self.log_btns = [QPushButton() for i in range(self.n_signals)]

		# signals = list(self.df_effectors.columns)
		# to_remove = ['TRACK_ID', 'FRAME','x_anim','y_anim','t', 'state', 'generation', 'root', 'parent', 'class_id', 'class', 't0', 'POSITION_X', 'POSITION_Y', 'position', 'well', 'well_index', 'well_name', 'pos_name', 'index']

		# for c in to_remove:
		# 	if c in signals:
		# 		signals.remove(c)

		# for i in range(len(self.signal_choice_effectors_cb)):
		# 	self.signal_choice_effectors_cb[i].addItems(['--']+signals)
		# 	self.signal_choice_effectors_cb[i].setCurrentIndex(i+1)
		# 	self.signal_choice_effectors_cb[i].currentIndexChanged.connect(self.plot_signals)


	def plot_signals(self):
		self.dataframes = {
			'targets': self.df_targets,
			'effectors': self.df_effectors,
		}
		try:
			yvalues = []
			for i in range(len(self.signal_choices)):
				signal_choice = self.signal_choices[i].currentText()
				#self.lines[i].set_label(signal_choice)
				if signal_choice=="--":
					self.lines[i].set_xdata([])
					self.lines[i].set_ydata([])
					self.lines[i].set_label('')
				else:
					if i == 0:
						if self.target_button1.isChecked():
							df_ref=self.dataframes[self.reference_population]
							self.lines[i].set_label(f'{self.reference_population} '+signal_choice)
							print(f'plot signal {signal_choice} for cell {self.reference_track_of_interest}')
							xdata = df_ref.loc[df_ref['TRACK_ID']==self.reference_track_of_interest, 'FRAME'].to_numpy()
							ydata = df_ref.loc[df_ref['TRACK_ID']==self.reference_track_of_interest, signal_choice].to_numpy()

						if self.effector_button1.isChecked():
							df_neigh=self.dataframes[self.neighbor_population]
							self.lines[i].set_label(f'{self.neighbor_population} ' + signal_choice)
							print(f'plot signal {signal_choice} for cell {self.neighbor_track_of_interest}')
							xdata = df_neigh.loc[
								df_neigh['TRACK_ID'] == self.neighbor_track_of_interest, 'FRAME'].to_numpy()
							ydata = df_neigh.loc[df_neigh[
															'TRACK_ID'] == self.neighbor_track_of_interest, signal_choice].to_numpy()
							# except:
							#     xdata = self.df_effectors.loc[
							#         self.df_effectors['ID'] == self.effector_track_of_interest, 'FRAME'].to_numpy()
							#     ydata = self.df_effectors.loc[self.df_effectors[
							#                                   'ID'] == self.effector_track_of_interest, signal_choice].to_numpy()
						if self.relative_button1.isChecked():
							self.lines[i].set_label('relative ' + signal_choice)
							print(
								f'plot signal {signal_choice} for reference cell {self.reference_track_of_interest} and neighbor cell {self.neighbor_track_of_interest}')
							xdata = self.dataframes[self.reference_population].loc[self.dataframes[self.reference_population][
																		  'TRACK_ID'] == self.reference_track_of_interest, 'FRAME'].to_numpy()
							ydata = self.df_relative.loc[
								(self.df_relative['REFERENCE_ID'] == self.reference_track_of_interest) & (
											self.df_relative['NEIGHBOR_ID'] == self.neighbor_track_of_interest)&(self.df_relative[self.neighborhood_choice_cb.currentText()]==1),
								signal_choice].to_numpy()
					elif i == 1:
						if self.target_button2.isChecked():
							df_ref = self.dataframes[self.reference_population]
							self.lines[i].set_label(f'{self.reference_population} ' + signal_choice)
							print(f'plot signal {signal_choice} for cell {self.reference_track_of_interest}')
							xdata = df_ref.loc[
								df_ref['TRACK_ID'] == self.reference_track_of_interest, 'FRAME'].to_numpy()
							ydata = df_ref.loc[
								df_ref['TRACK_ID'] == self.reference_track_of_interest, signal_choice].to_numpy()
						if self.effector_button2.isChecked():
							df_neigh = self.dataframes[self.neighbor_population]
							self.lines[i].set_label(f'{self.neighbor_population} ' + signal_choice)
							print(f'plot signal {signal_choice} for cell {self.neighbor_track_of_interest}')
							xdata = df_neigh.loc[
								df_neigh['TRACK_ID'] == self.neighbor_track_of_interest, 'FRAME'].to_numpy()
							ydata = df_neigh.loc[df_neigh[
													 'TRACK_ID'] == self.neighbor_track_of_interest, signal_choice].to_numpy()

							# except:
							#     xdata = self.df_effectors.loc[
							#         self.df_effectors['ID'] == self.effector_track_of_interest, 'FRAME'].to_numpy()
							#     ydata = self.df_effectors.loc[self.df_effectors[
							#                                   'ID'] == self.effector_track_of_interest, signal_choice].to_numpy()
						if self.relative_button2.isChecked():
							self.lines[i].set_label('relative ' + signal_choice)
							print(
								f'plot signal {signal_choice} for target cell {self.reference_track_of_interest} and effector cell {self.neighbor_track_of_interest}')
							xdata = self.dataframes[self.reference_population].loc[self.dataframes[self.reference_population][
																		  'TRACK_ID'] == self.reference_track_of_interest, 'FRAME'].to_numpy()
							ydata = self.df_relative.loc[
								(self.df_relative['REFERENCE_ID'] == self.reference_track_of_interest) & (
											self.df_relative['NEIGHBOR_ID'] == self.neighbor_track_of_interest)&(self.df_relative[self.neighborhood_choice_cb.currentText()]==1),
								signal_choice].to_numpy()
					else:
						if self.target_button3.isChecked():
							df_ref = self.dataframes[self.reference_population]
							self.lines[i].set_label(f'{self.reference_population} ' + signal_choice)
							print(f'plot signal {signal_choice} for cell {self.reference_track_of_interest}')
							xdata = df_ref.loc[
								df_ref['TRACK_ID'] == self.reference_track_of_interest, 'FRAME'].to_numpy()
							ydata = df_ref.loc[
								df_ref['TRACK_ID'] == self.reference_track_of_interest, signal_choice].to_numpy()
						if self.effector_button3.isChecked():
							df_neigh = self.dataframes[self.neighbor_population]
							self.lines[i].set_label(f'{self.neighbor_population} ' + signal_choice)
							print(f'plot signal {signal_choice} for cell {self.effector_track_of_interest}')
							xdata = df_neigh.loc[
								df_neigh['TRACK_ID'] == self.neighbor_track_of_interest, 'FRAME'].to_numpy()
							ydata = df_neigh.loc[df_neigh[
													 'TRACK_ID'] == self.neighbor_track_of_interest, signal_choice].to_numpy()
							# except:
							#     xdata = self.df_effectors.loc[
							#         self.df_effectors['ID'] == self.effector_track_of_interest, 'FRAME'].to_numpy()
							#     ydata = self.df_effectors.loc[self.df_effectors[
							#                                   'ID'] == self.effector_track_of_interest, signal_choice].to_numpy()
						if self.relative_button3.isChecked():
							self.lines[i].set_label('relative ' + signal_choice)
							print(
								f'plot signal {signal_choice} for reference cell {self.reference_track_of_interest} and neighbor cell {self.neighbor_track_of_interest}')
							xdata = self.dataframes[self.reference_population].loc[self.dataframes[self.reference_population][
																		  'TRACK_ID'] == self.reference_track_of_interest, 'FRAME'].to_numpy()
							ydata = self.df_relative.loc[
								(self.df_relative['REFERENCE_ID'] == self.reference_track_of_interest) & (
											self.df_relative['NEIGHBOR_ID'] == self.neighbor_track_of_interest)&(self.df_relative[self.neighborhood_choice_cb.currentText()]==1),
								signal_choice].to_numpy()

						# if ydata!=[]:
						# 	print(ydata.shape)
						# else:
						# 	ydata=np.zeros(30)
						# print(ydata)

					#xdata = xdata[ydata == ydata]  # remove nan
					xdata = xdata[ydata == ydata]
					ydata = ydata[ydata == ydata]
					yvalues.extend(ydata)
					t0 = self.df_relative.loc[(self.df_relative['REFERENCE_ID']==self.reference_track_of_interest)&(self.df_relative['NEIGHBOR_ID']==self.neighbor_track_of_interest),f't0_{self.neighborhood_choice_cb.currentText()}'].to_numpy()

					self.line_dt.set_xdata([t0, t0])
					min_val=np.min(ydata)
					max_val=np.max(ydata)

					self.line_dt.set_ydata([min_val, max_val])
					self.lines[i].set_xdata(xdata)
					self.lines[i].set_ydata(ydata)
					self.lines[i].set_color(tab10(i / 3.))




			self.configure_ylims()


			min_val,max_val = self.cell_ax.get_ylim()
			t0 = self.df_relative.loc[(self.df_relative['REFERENCE_ID'] == self.reference_track_of_interest) & (
						self.df_relative['NEIGHBOR_ID'] == self.neighbor_track_of_interest), f't0_{self.neighborhood_choice_cb.currentText()}'].to_numpy()

			if t0!=[]:
				t0=t0[0]
				self.line_dt.set_xdata([t0, t0])
				self.line_dt.set_ydata([min_val,max_val])

			self.cell_ax.legend()
			self.cell_fcanvas.canvas.draw()
		except Exception as e:
				print(f"{e=}")

	def extract_scatter_from_lines(self):

		self.lines_list = []
		self.lines_tracks=[]
		self.lines_colors_status = []
		self.initial_lines_colors_status=[]
		self.lines_colors_class = []
		self.initial_lines_colors_class=[]

		for t in np.arange(self.len_movie):

			# Append frame_positions to self.line_positions
			self.lines_tracks.append(self.df_relative.loc[(self.df_relative['FRAME'] == t)&(~self.df_relative['status_'+self.neighborhood_choice_cb.currentText()].isnull()), ['REFERENCE_ID', 'NEIGHBOR_ID']].to_numpy())
			self.initial_lines_colors_status.append(self.df_relative.loc[(self.df_relative['FRAME'] == t)&(~self.df_relative['status_'+self.neighborhood_choice_cb.currentText()].isnull()), ['REFERENCE_ID', 'NEIGHBOR_ID','status_color']].to_numpy())
			self.lines_colors_status.append(self.df_relative.loc[(self.df_relative['FRAME'] == t)&(~self.df_relative['status_'+self.neighborhood_choice_cb.currentText()].isnull()), ['REFERENCE_ID', 'NEIGHBOR_ID','status_color']].to_numpy())
			self.initial_lines_colors_class.append(self.df_relative.loc[(self.df_relative['FRAME'] == t)&(~self.df_relative['status_'+self.neighborhood_choice_cb.currentText()].isnull()), ['REFERENCE_ID', 'NEIGHBOR_ID','class_color']].to_numpy())
			self.lines_colors_class.append(self.df_relative.loc[(self.df_relative['FRAME'] == t)&(~self.df_relative['status_'+self.neighborhood_choice_cb.currentText()].isnull()), ['REFERENCE_ID', 'NEIGHBOR_ID','class_color']].to_numpy())

	def extract_scatter_from_target_trajectories(self):

		self.target_positions = []
		self.target_colors = []
		self.target_tracks = []
		self.initial_target_colors = []

		for t in np.arange(self.len_movie):

			if self.df_targets is not None:
				self.target_positions.append(self.df_targets.loc[self.df_targets['FRAME']==t,['x_anim', 'y_anim']].to_numpy())
				self.target_colors.append(self.df_targets.loc[self.df_targets['FRAME']==t,['class_color', 'status_color']].to_numpy())
				self.initial_target_colors.append(
					self.df_targets.loc[self.df_targets['FRAME'] == t, ['class_color', 'status_color']].to_numpy())
				self.target_tracks.append(self.df_targets.loc[self.df_targets['FRAME']==t, 'TRACK_ID'].to_numpy())


	def extract_scatter_from_effector_trajectories(self):

		self.effector_positions = []
		self.effector_colors = []
		self.initial_effector_colors=[]
		self.effector_tracks = []

		for t in np.arange(self.len_movie):

			if self.df_effectors is not None:

				self.effector_positions.append(self.df_effectors.loc[self.df_effectors['FRAME']==t,['x_anim', 'y_anim']].to_numpy())
				self.effector_colors.append(self.df_effectors.loc[self.df_effectors['FRAME']==t,['class_color', 'status_color']].to_numpy())
				self.initial_effector_colors.append(self.df_effectors.loc[self.df_effectors['FRAME'] == t, ['class_color', 'status_color']].to_numpy())
				try:
					self.effector_tracks.append(self.df_effectors.loc[self.df_effectors['FRAME']==t, 'TRACK_ID'].to_numpy())
				except:
					self.effector_tracks.append(
						self.df_effectors.loc[self.df_effectors['FRAME'] == t, 'ID'].to_numpy())

	def load_annotator_config(self):

		"""
		Load settings from config or set default values.
		"""

		print('Reading instructions..')
		if os.path.exists(self.instructions_path):
			with open(self.instructions_path, 'r') as f:

				instructions = json.load(f)
				print(f'Reading instructions: {instructions}')

				if 'rgb_mode' in instructions:
					self.rgb_mode = instructions['rgb_mode']
				else:
					self.rgb_mode = False

				if 'percentile_mode' in instructions:
					self.percentile_mode = instructions['percentile_mode']
				else:
					self.percentile_mode = True

				if 'channels' in instructions:
					self.target_channels = instructions['channels']
				else:
					self.target_channels = [[self.channel_names[0], 0.01, 99.99]]

				if 'fraction' in instructions:
					self.fraction = float(instructions['fraction'])
				else:
					self.fraction = 0.25

				if 'interval' in instructions:
					self.anim_interval = int(instructions['interval'])
				else:
					self.anim_interval = 1

				if 'log' in instructions:
					self.log_option = instructions['log']
				else:
					self.log_option = False
		else:
			self.rgb_mode = False
			self.log_option = False
			self.percentile_mode = True
			self.target_channels = [[self.channel_names[0], 0.01, 99.99]]
			self.fraction = 0.25
			self.anim_interval = 1

	def prepare_stack(self):

		self.img_num_channels = _get_img_num_per_channel(self.channels, self.len_movie, self.nbr_channels)
		self.stack = []
		for ch in tqdm(self.target_channels, desc="channel"):
			target_ch_name = ch[0]
			if self.percentile_mode:
				normalize_kwargs = {"percentiles": (ch[1], ch[2]), "values": None}
			else:
				normalize_kwargs = {"values": (ch[1], ch[2]), "percentiles": None}

			if self.rgb_mode:
				normalize_kwargs.update({'amplification': 255., 'clip': True})

			chan = []
			indices = self.img_num_channels[self.channels[np.where(self.channel_names==target_ch_name)][0]]
			for t in tqdm(range(len(indices)),desc='FRAME'):
				if self.rgb_mode:
					f = load_frames(indices[t], self.stack_path, scale=self.fraction, normalize_input=True, normalize_kwargs=normalize_kwargs)
					f = f.astype(np.uint8)
				else:
					f = load_frames(indices[t], self.stack_path, scale=self.fraction, normalize_input=False)
				chan.append(f[:,:,0])

			self.stack.append(chan)

		self.stack = np.array(self.stack)
		if self.rgb_mode:
			self.stack = np.moveaxis(self.stack, 0, -1)
		else:
			self.stack = self.stack[0]
			if self.log_option:
				self.stack[np.where(self.stack>0.)] = np.log(self.stack[np.where(self.stack>0.)])

		print(f'Load stack of shape: {self.stack.shape}.')

	def neighborhood_changed(self):
		self.set_reference_and_neighbor_populations()
		self.cancel_selection()
		self.update_cell_events()
		self.extract_scatter_from_lines()
		# self.draw_frame(self.framedata)
		self.plot_signals()


	def closeEvent(self, event):

		self.stop()
		# result = QMessageBox.question(self,
		# 			  "Confirm Exit...",
		# 			  "Are you sure you want to exit ?",
		# 			  QMessageBox.Yes| QMessageBox.No,
		# 			  )
		del self.stack
		gc.collect()

	def looped_animation(self):

		"""
		Load an image.

		"""

		self.framedata = 0

		self.fig, self.ax = plt.subplots(tight_layout=True)
		self.fcanvas = FigureCanvas(self.fig, interactive=True)
		self.ax.clear()

		if not hasattr(self, 'lines'):
			self.lines_data = {}

		self.im = self.ax.imshow(self.stack[0], cmap='gray', vmin=np.nanpercentile(self.stack, 1), vmax=np.nanpercentile(self.stack, 99.99))
		if self.df_targets is not None:
			self.target_status_scatter = self.ax.scatter(self.target_positions[0][:,0], self.target_positions[0][:,1], marker="x", c=self.target_colors[0][:,1], s=50, picker=True, pickradius=10)
			self.target_class_scatter = self.ax.scatter(self.target_positions[0][:,0], self.target_positions[0][:,1], marker='o', facecolors='none',edgecolors=self.target_colors[0][:,0], s=200)
		else:
			self.target_status_scatter = self.ax.scatter([],[], marker="x", s=50, picker=True, pickradius=10)
			self.target_class_scatter = self.ax.scatter([],[], marker='o', facecolors='none', s=200)

		if self.df_effectors is not None:
			self.effector_status_scatter = self.ax.scatter(self.effector_positions[0][:,0], self.effector_positions[0][:,1], marker="x", c=self.effector_colors[0][:,1], s=50, picker=True, pickradius=10)
			self.effector_class_scatter = self.ax.scatter(self.effector_positions[0][:,0], self.effector_positions[0][:,1], marker='^', facecolors='none',edgecolors=self.effector_colors[0][:,0], s=200)
		else:
			self.effector_status_scatter = self.ax.scatter([], [], marker="x", s=50, picker=True, pickradius=10)
			self.effector_class_scatter = self.ax.scatter([],[], marker='^', facecolors='none', s=200)

		self.ax.set_xticks([])
		self.ax.set_yticks([])
		self.ax.set_aspect('equal')

		self.fig.set_facecolor('none')  # or 'None'
		self.fig.canvas.setStyleSheet("background-color: black;")

		self.anim = FuncAnimation(
							   self.fig,
							   self.draw_frame,
							   frames = self.len_movie, # better would be to cast np.arange(len(movie)) in case frame column is incomplete
							   interval = self.anim_interval, # in ms
							   blit=True,
							   )

		self.fig.canvas.mpl_connect('pick_event', self.on_scatter_pick)
		self.fcanvas.canvas.draw()


	def create_cell_signal_canvas(self):

		self.cell_fig, self.cell_ax = plt.subplots()
		self.cell_fcanvas = FigureCanvas(self.cell_fig, interactive=False)
		self.cell_ax.clear()

		spacing = 0.5
		minorLocator = MultipleLocator(1)
		self.cell_ax.xaxis.set_minor_locator(minorLocator)
		self.cell_ax.xaxis.set_major_locator(MultipleLocator(5))
		self.cell_ax.grid(which = 'major')
		self.cell_ax.set_xlabel("time [frame]")
		self.cell_ax.set_ylabel("signal")

		self.cell_fig.set_facecolor('none')  # or 'None'
		self.cell_fig.canvas.setStyleSheet("background-color: transparent;")

		self.lines = [self.cell_ax.plot([np.linspace(0,self.len_movie-1,self.len_movie)],[np.zeros((self.len_movie))])[0] for i in range(len(self.signal_choices))]
		for i in range(len(self.lines)):
			self.lines[i].set_label(f'signal {i}')

		min_val,max_val = self.cell_ax.get_ylim()
		self.line_dt, = self.cell_ax.plot([-1,-1],[min_val,max_val],c="k",linestyle="--")

		self.cell_ax.set_xlim(0,self.len_movie)
		self.cell_ax.legend()
		self.cell_fcanvas.canvas.draw()

		self.plot_signals()


	def on_scatter_pick(self, event):
		
		self.identify_closest_marker(event)

		_, tracks, _, _ = self.get_reference_sets()

		if self.selected_population == self.reference_population:

			if self.index is not None:
				toi = tracks[self.framedata][self.index]

			if len(self.reference_selection)==0:

				self.reference_track_of_interest = toi
				self.reference_selection.append(self.reference_track_of_interest)

				self.get_neighbors_of_selected_cell(self.reference_track_of_interest)
				print(f'You selected track {self.reference_track_of_interest} with {len(self.neighbors)} neighbors...')

				self.give_reference_cell_information()
				#self.plot_signals()

				self.recolor_selection()
				self.trace_neighbors()

			elif len(self.reference_selection) > 0 and toi in self.reference_selection:

				self.cancel_btn.click()
				self.cancel_selection()

			elif len(self.reference_selection) > 0 and toi in self.neighbors and self.neighbor_population==self.reference_population:
				if len(self.pair_selection)==0:
					self.highlight_the_pair(toi)
				else:
					self.cancel_pair_selection()
			else:
				print('one cell already selected... skip... ')
				pass
		else:
			_, tracks, _ = self.get_neighbor_sets()
			if self.index is not None:
				toi = tracks[self.framedata][self.index]
			if toi in self.neighbors:
				print('you picked a neighbor!!')

		if self.pair_selected:

			print('You selected a pair...')
			print(f'{self.pair_selection=}')

			artist = event.artist

			if self.index is not None:

				selected_point = artist.get_offsets()[self.index]
				connect = self.connections[(selected_point[0], selected_point[1])]

				self.neigh = connect[0][1]
				self.neigh = -1

				if len(self.pair_selection) == 0 and ((selected_point[0],selected_point[1]) in self.connections.keys()):

					connect = self.connections[(selected_point[0], selected_point[1])]
					
					self.correct_btn.setEnabled(True)
					self.cancel_btn.setEnabled(True)
					self.del_shortcut.setEnabled(True)
					self.no_event_shortcut.setEnabled(True)
					self.pair_selection.append(self.index)

					self.neighbor_loc_t=[]
					self.neighbor_loc_idx=[]
					self.neigh_previous_color=[]

					for t in range(self.len_movie):
						self.lines_colors_status[t][:, :2] = self.lines_colors_status[t][:, :2].astype(float)

						indices1 = np.where((self.lines_colors_status[t][:, 0] == connect[0][0]) &
										   (self.lines_colors_status[t][:, 1] == connect[0][1]))[0]
						self.lines_colors_class[t][:, :2] = self.lines_colors_class[t][:, :2].astype(float)

						indices2 = np.where((self.lines_colors_class[t][:, 0] == connect[0][0]) &
										   (self.lines_colors_class[t][:, 1] == connect[0][1]))[0]

						self.lines_colors_status[t][indices1, 2] = 'lime'
						self.lines_colors_class[t][indices2, 2] = 'lime'


					self.reference_track_of_interest = connect[0][0]
					self.neighbor_track_of_interest = connect[0][1]

					positions, tracks, colors = self.get_neighbor_sets()
					
					for t in range(len(tracks)):
						indices_picked = np.where(tracks[t] == self.neighbor_track_of_interest)[0]

						if len(indices_picked) > 0:
							self.neighbor_loc_t.append(t)
							self.neighbor_loc_idx.append(indices_picked[0])

					for t, idx in zip(self.neighbor_loc_t, self.neighbor_loc_idx):
						self.neigh_previous_color.append(colors[t][idx].copy())
						colors[t][idx] = 'lime'

					self.give_reference_cell_information()
					self.give_neighbor_cell_information()
					#self.plot_signals()

				elif len(self.pair_selection)==1: # and self.neigh!=self.neighbor_track_of_interest
					print('Length of pair selection is larger than one, trying to cancel the pair selection...')
					self.cancel_pair_selection()
				else:
					print('something else')
					self.cancel_pair_selection()
			else:
				pass
		else:
			pass

	def highlight_the_pair(self, neighbor_cell):

		# 1) recolor the neighbor marker
		print(f'Reference cell: {self.reference_track_of_interest}, neighbor cell: {neighbor_cell}')

		_, tracks, colors = self.get_neighbor_sets()
		self.neigh_cell_loc_idx = []
		self.neigh_cell_loc_t = []
		self.neigh_previous_color = []

		for t in range(len(tracks)):
			indices_picked = np.where(tracks[t]==neighbor_cell)[0]
			if len(indices_picked)>0:
				self.neigh_cell_loc_t.append(t)
				self.neigh_cell_loc_idx.append(indices_picked[0])

		for t,idx in zip(self.neigh_cell_loc_t,self.neigh_cell_loc_idx):
			self.neigh_previous_color.append(colors[t][idx].copy())
			colors[t][idx] = 'lime'

		# 2) identify the pair line and recolor it
		for t in range(self.len_movie):

			self.lines_colors_status[t][:, :2] = self.lines_colors_status[t][:, :2].astype(float)
			indices1 = np.where((self.lines_colors_status[t][:, 0] == self.reference_track_of_interest)&(self.lines_colors_status[t][:, 1] == neighbor_cell))[0]

			self.lines_colors_class[t][:, :2] = self.lines_colors_class[t][:, :2].astype(float)
			indices2 = np.where((self.lines_colors_class[t][:, 0] == self.reference_track_of_interest)&(self.lines_colors_class[t][:, 1] == neighbor_cell))[0]

			self.lines_colors_status[t][indices1, 2] = 'lime'
			self.lines_colors_class[t][indices2, 2] = 'lime'
			# Maybe do the symmetrical neighborhood when same populations?

		self.pair_selection.append(tuple([self.reference_track_of_interest, neighbor_cell]))

	def get_neighbor_sets(self):

		if self.reference_population != self.neighbor_population:
			if self.reference_population=='effectors':
				return self.target_positions, self.target_tracks, self.target_colors
			elif self.reference_population=='targets':
				return self.effector_positions, self.effector_tracks, self.effector_colors
		else:
			if self.reference_population=='effectors':
				return self.effector_positions, self.effector_tracks, self.effector_colors
			elif self.reference_population=='targets':
				return self.target_positions, self.target_tracks, self.target_colors		

	def get_reference_sets(self):

		if self.reference_population == 'effectors':
			return self.effector_positions, self.effector_tracks, self.effector_colors, self.initial_effector_colors
		elif self.reference_population == 'targets':
			return self.target_positions, self.target_tracks, self.target_colors, self.initial_target_colors	

	def trace_neighbors(self):

		self.lines_data = {}
		self.points_data={}
		self.connections={}
		self.line_connections={}

		positions, tracks, colors = self.get_neighbor_sets()

		# Look for neighbors
		for neigh in self.neighbors:

			self.neighbor_loc_t = []
			self.neighbor_loc_idx = []

			for t in range(len(tracks)):
				indices = np.where(tracks[t]==neigh)[0]
				if len(indices)>0:
					self.neighbor_loc_t.append(t)
					self.neighbor_loc_idx.append(indices[0])

			self.neighbor_previous_color = []
			for t, idx in zip(self.neighbor_loc_t, self.neighbor_loc_idx):

				neigh_x = positions[t][idx, 0]
				neigh_y = positions[t][idx, 1]
				x_m_point = (self.reference_x[t] + neigh_x) / 2
				y_m_point = (self.reference_y[t] + neigh_y) / 2

				if t not in self.lines_data.keys():
					self.lines_data[t]=[([self.reference_x[t], neigh_x], [self.reference_y[t], neigh_y])]
					self.points_data[t]=[(x_m_point, y_m_point)]
				else:
					self.lines_data[t].append(([self.reference_x[t], neigh_x], [self.reference_y[t], neigh_y]))
					self.points_data[t].append((x_m_point, y_m_point))

				self.connections[(x_m_point, y_m_point)] = [(self.reference_track_of_interest, neigh)]
				self.line_connections[(self.reference_x[t], neigh_x, self.reference_y[t], neigh_y)]=[(self.reference_track_of_interest, neigh)]

				self.neighbor_previous_color.append(colors[t][idx].copy())
				colors[t][idx] = 'salmon'

			# for t in range(len(colors)):
			# 	for idx in range(len(colors[t])):
			# 		if colors[t][idx].any() != 'salmon':
			# 			if colors[t][idx].any() != 'magenta':
			# 				#init_color[t][idx] = colors[t][idx].copy()
			# 				colors[t][idx] = 'black'

	def recolor_selection(self):

		positions, tracks, colors, init_colors = self.get_reference_sets()
		
		self.reference_loc_t = []
		self.reference_loc_idx = []
		self.reference_loc_t_not_picked = []
		self.reference_loc_idx_not_picked=[]

		for t in range(len(tracks)):
			
			indices_picked = np.where(tracks[t]==self.reference_track_of_interest)[0]
			indices_not_picked = np.where(tracks[t]!=self.reference_track_of_interest)[0]
			self.reference_loc_t_not_picked.append(t)
			self.reference_loc_idx_not_picked.append(indices_not_picked)
			if len(indices_picked)>0:
				self.reference_loc_t.append(t)
				self.reference_loc_idx.append(indices_picked[0])

		self.reference_previous_color = []
		self.reference_not_picked_initial_colors=[]
		self.reference_x = []
		self.reference_y = []

		# Recolor selected cell
		for t,idx in zip(self.reference_loc_t,self.reference_loc_idx):
			self.reference_x.append(positions[t][idx, 0])
			self.reference_y.append(positions[t][idx, 1])
			self.reference_previous_color.append(colors[t][idx].copy())
			colors[t][idx] = 'lime'

		# Recolor all other cells in black
		for t, idx in zip(self.reference_loc_t_not_picked, self.reference_loc_idx_not_picked):
			self.reference_not_picked_initial_colors.append(colors[t][idx].copy())
			init_colors[t][idx] = colors[t][idx].copy()
			colors[t][idx] = 'black'


	def get_neighbors_of_selected_cell(self, selected_cell):
		
		self.neighbors = self.df_relative.loc[(self.df_relative['REFERENCE_ID'] == selected_cell)&(~self.df_relative['status_'+self.neighborhood_choice_cb.currentText()].isnull()),'NEIGHBOR_ID']
		self.neighbors = np.unique(self.neighbors)
		if len(self.neighbors)>0:
			first_neighbor = np.min(self.neighbors)
			self.neighbor_track_of_interest = first_neighbor
		else:
			self.neighbor_track_of_interest = None

	def identify_closest_marker(self, event):
		
		ind = event.ind
		label = event.artist.get_label()

		# Identify the nature of the selected object (target/effector/pair)
		self.pair_selected = False
		if label == '_child1':
			self.selected_population = 'targets'
		elif label == '_child3':
			self.selected_population = 'effectors'
		else:
			number = int(label.split('_child')[1])
			if number>4:
				self.pair_selected = True
			else:
				self.pair_selected = False
				return None

		if self.selected_population=='effectors':
			positions = self.effector_positions
		elif self.selected_population=='targets':
			positions = self.target_positions

		if len(ind)==1:
			self.index = ind[0]
		elif len(ind)>1:
			# More than one point in vicinity
			datax,datay = [positions[self.framedata][i,0] for i in ind],[positions[self.framedata][i,1] for i in ind]
			msx, msy = event.mouseevent.xdata, event.mouseevent.ydata
			dist = np.sqrt((np.array(datax)-msx)**2+(np.array(datay)-msy)**2)
			self.index = ind[np.argmin(dist)]
		else:
			self.index = None


	def show_annotation_buttons(self):

		for a in self.annotation_btns_to_hide:
			a.show()
		try:
			cclass = self.df_relative.loc[(self.df_relative['REFERENCE_ID'] == self.reference_track_of_interest)&(self.df_relative['NEIGHBOR_ID']==self.neighbor_track_of_interest)&
									  (self.df_relative[f'{self.neighborhood_choice_cb.currentText()}']==1), self.pair_class_name].to_numpy()[0]
			t0 = self.df_relative.loc[(self.df_relative['REFERENCE_ID'] == self.reference_track_of_interest)&(self.df_relative['NEIGHBOR_ID']==self.neighbor_track_of_interest)&
									  (self.df_relative[f'{self.neighborhood_choice_cb.currentText()}']==1), self.pair_time_name].to_numpy()[0]
		except:
			print('chet ne nashel')
			pass
		if cclass == 0:
			self.event_btn.setChecked(True)
			self.time_of_interest_le.setText(str(t0))
		elif cclass == 1:
			self.no_event_btn.setChecked(True)
		elif cclass == 2:
			self.else_btn.setChecked(True)
		elif cclass > 2:
			self.suppr_btn.setChecked(True)

		self.enable_time_of_interest()
		self.correct_btn.setText('submit')

		self.correct_btn.disconnect()
		self.correct_btn.clicked.connect(self.apply_modification)

	# def hide_annotation_buttons(self):
	#
	#     for a in self.annotation_btns_to_hide:
	#         a.hide()
	#     for b in [self.event_btn, self.no_event_btn, self.else_btn, self.suppr_btn]:
	#         b.setChecked(False)
	#     self.time_of_interest_label.setEnabled(False)
	#     self.time_of_interest_le.setText('')
	#     self.time_of_interest_le.setEnabled(False)
	def shortcut_suppr(self):
		self.correct_btn.click()
		self.suppr_btn.click()
		self.correct_btn.click()

	def shortcut_no_event(self):
		self.correct_btn.click()
		self.no_event_btn.click()
		self.correct_btn.click()

	def configure_ylims(self):

		try:
			min_values = []
			max_values = []
			for i in range(len(self.signal_choices)):
				signal = self.signal_choices[i].currentText()
				if signal=='--':
					continue
				else:
					if i==0:
						if self.target_button1.isChecked():
							df_ref=self.dataframes[self.reference_population]
							maxx_target = np.nanpercentile(df_ref.loc[:,signal].to_numpy().flatten(),99)
							minn_target = np.nanpercentile(df_ref.loc[:,signal].to_numpy().flatten(),1)
							min_values.append(minn_target)
							max_values.append(maxx_target)
						if self.effector_button1.isChecked():
							df_neigh=self.dataframes[self.neighbor_population]
							maxx_target = np.nanpercentile(df_neigh.loc[:, signal].to_numpy().flatten(), 99)
							minn_target = np.nanpercentile(df_neigh.loc[:, signal].to_numpy().flatten(), 1)
							min_values.append(minn_target)
							max_values.append(maxx_target)
						if self.relative_button1.isChecked():
							maxx_relative = np.nanpercentile(self.df_relative.loc[:, signal].to_numpy().flatten(), 99)
							minn_relative = np.nanpercentile(self.df_relative.loc[:, signal].to_numpy().flatten(), 1)
							min_values.append(minn_relative)
							max_values.append(maxx_relative)
					elif i==1:
						if self.target_button2.isChecked():
							df_ref=self.dataframes[self.reference_population]
							maxx_effector = np.nanpercentile(df_ref.loc[:,signal].to_numpy().flatten(),99)
							minn_effector = np.nanpercentile(df_ref.loc[:,signal].to_numpy().flatten(),1)
							min_values.append(minn_effector)
							max_values.append(maxx_effector)
						if self.effector_button2.isChecked():
							df_neigh=self.dataframes[self.neighbor_population]
							maxx_effector = np.nanpercentile(df_neigh.loc[:, signal].to_numpy().flatten(), 99)
							minn_effector = np.nanpercentile(df_neigh.loc[:, signal].to_numpy().flatten(), 1)
							min_values.append(minn_effector)
							max_values.append(maxx_effector)
						if self.relative_button2.isChecked():
							maxx_relative = np.nanpercentile(self.df_relative.loc[:, signal].to_numpy().flatten(), 99)
							minn_relative = np.nanpercentile(self.df_relative.loc[:, signal].to_numpy().flatten(), 1)
							min_values.append(minn_relative)
							max_values.append(maxx_relative)
					else:
						if self.target_button3.isChecked():
							df_ref=self.dataframes[self.reference_population]
							maxx_relative = np.nanpercentile(df_ref.loc[:,signal].to_numpy().flatten(),99)
							minn_relative = np.nanpercentile(df_ref.loc[:,signal].to_numpy().flatten(),1)
							min_values.append(minn_relative)
							max_values.append(maxx_relative)
						if self.effector_button3.isChecked():
							df_neigh=self.dataframes[self.neighbor_population]

							maxx_relative = np.nanpercentile(df_neigh.loc[:, signal].to_numpy().flatten(), 99)
							minn_relative = np.nanpercentile(df_neigh.loc[:, signal].to_numpy().flatten(), 1)
							min_values.append(minn_relative)
							max_values.append(maxx_relative)
						if self.relative_button3.isChecked():
							maxx_relative = np.nanpercentile(self.df_relative.loc[:, signal].to_numpy().flatten(), 99)
							minn_relative = np.nanpercentile(self.df_relative.loc[:, signal].to_numpy().flatten(), 1)
							min_values.append(minn_relative)
							max_values.append(maxx_relative)

			if len(min_values)>0:
				self.cell_ax.set_ylim(np.amin(min_values), np.amax(max_values))
		except Exception as e:
			print(e)

	def draw_frame(self, framedata):

		"""
		Update plot elements at each timestep of the loop.
		"""

		self.framedata = framedata
		self.frame_lbl.setText(f'frame: {self.framedata}')
		self.im.set_array(self.stack[self.framedata])
		#if self.reference_population=='targets':
		if self.df_targets is not None:
			self.target_status_scatter.set_visible(True)
			self.target_class_scatter.set_visible(True)
			# self.target_status_scatter.set_alpha(1)
			self.target_status_scatter.set_picker(True)
			# self.target_class_scatter.set_alpha(1)
			self.target_status_scatter.set_offsets(self.target_positions[self.framedata])
			self.target_status_scatter.set_color(self.target_colors[self.framedata][:,1])
			self.target_class_scatter.set_offsets(self.target_positions[self.framedata])
			self.target_class_scatter.set_edgecolor(self.target_colors[self.framedata][:,0])
		#if self.reference_population!=self.neighbor_population:
		if self.df_effectors is not None:
			self.effector_status_scatter.set_visible(True)
			self.effector_class_scatter.set_visible(True)
			# self.effector_status_scatter.set_alpha(1)
			self.effector_status_scatter.set_picker(True)
			# self.effector_class_scatter.set_alpha(1)
			self.effector_status_scatter.set_offsets(self.effector_positions[self.framedata])
			self.effector_status_scatter.set_color(self.effector_colors[self.framedata][:,1])

			self.effector_class_scatter.set_offsets(self.effector_positions[self.framedata])
			self.effector_class_scatter.set_edgecolor(self.effector_colors[self.framedata][:,0])
		#else:
		# if self.df_effectors is not None:
		# 	self.effector_status_scatter.set_visible(False)
		# 	self.effector_class_scatter.set_visible(False)
		# 	# self.effector_status_scatter.set_alpha(0)
		# 	self.effector_status_scatter.set_picker(None)
		# 	# self.effector_class_scatter.set_alpha(0)
		#else:
		if self.df_effectors is not None:
			self.effector_status_scatter.set_visible(True)
			self.effector_status_scatter.set_picker(True)
			self.effector_class_scatter.set_visible(True)
			self.effector_status_scatter.set_offsets(self.effector_positions[self.framedata])
			self.effector_status_scatter.set_color(self.effector_colors[self.framedata][:, 1])

			self.effector_class_scatter.set_offsets(self.effector_positions[self.framedata])
			self.effector_class_scatter.set_edgecolor(self.effector_colors[self.framedata][:, 0])
		#if self.reference_population != self.neighbor_population:
		if self.df_targets is not None:
			self.target_status_scatter.set_visible(True)
			self.target_status_scatter.set_picker(True)
			self.target_class_scatter.set_visible(True)
			self.target_status_scatter.set_offsets(self.target_positions[self.framedata])
			self.target_status_scatter.set_color(self.target_colors[self.framedata][:, 1])

			self.target_class_scatter.set_offsets(self.target_positions[self.framedata])
			self.target_class_scatter.set_edgecolor(self.target_colors[self.framedata][:, 0])
		#else:
		if self.df_targets is not None:
			self.target_status_scatter.set_visible(False)
			self.target_status_scatter.set_picker(None)
			self.target_class_scatter.set_visible(False)
		self.lines_list=[]
		self.point_list=[]

		try:
			for key in self.lines_data:
				if key==self.framedata:
					for line in self.lines_data[key]:
						x_coords, y_coords = line
						pair=self.line_connections[x_coords[0],x_coords[1],y_coords[0],y_coords[1]]

						this_frame=self.lines_colors_class[self.framedata]

						this_pair=this_frame[(this_frame[:, 0] == pair[0][0]) & (this_frame[:, 1] == pair[0][1])]
						self.lines_plot=self.ax.plot(x_coords, y_coords, alpha=1, linewidth=2,color=this_pair[0][2])

						#self.ax.draw_artist(self.lines)
						# for l in self.lines:
						#     self.ax.draw_artist(l)
						self.lines_list.append(self.lines_plot[0])

					# Plot points
					for point in self.points_data[key]:
						x, y = point
						pair=self.connections[x,y]

						this_frame=self.lines_colors_status[self.framedata]
						this_pair=this_frame[(this_frame[:, 0] == pair[0][0]) & (this_frame[:, 1] == pair[0][1])]
						self.points=self.ax.scatter(x, y, marker="x", color=this_pair[0][2],picker=True)
						self.ax.draw_artist(self.points)
						#self.ax.draw_artist(self.points)
						self.point_list.append(self.points)
					# for coords_line in self.lines_data[key]:
					#     line.set_alpha(1)
					#     self.ax.draw_artist(line)
					#     self.lines_list.append(line)
					# for coords_point in self.points_data[key]:
					#     point.set_alpha(1)
					#     self.ax.draw_artist(point)
					#     self.point_list.append(point)
		except:
			pass

		if self.lines_list!=[]:
			return [self.im,self.target_status_scatter,self.target_class_scatter,self.effector_status_scatter,self.effector_class_scatter] +self.lines_list +self.point_list
		else:
			return [self.im, self.target_status_scatter, self.target_class_scatter, self.effector_status_scatter,
					self.effector_class_scatter,]

	def stop(self):
		# # On stop we disconnect all of our events.
		self.stop_btn.hide()
		self.start_btn.show()
		self.anim.pause()
		self.stop_btn.clicked.connect(self.start)


	def start(self):
		'''
		Starts interactive animation. Adds the draw frame command to the GUI
		handler, calls show to start the event loop.
		'''
		self.start_btn.setShortcut(QKeySequence(""))

		self.last_frame_btn.setEnabled(True)
		self.last_frame_btn.clicked.connect(self.set_last_frame)

		self.first_frame_btn.setEnabled(True)
		self.first_frame_btn.clicked.connect(self.set_first_frame)


		self.start_btn.hide()
		self.stop_btn.show()

		self.anim.event_source.start()
		self.stop_btn.clicked.connect(self.stop)


	def give_target_cell_information(self):

		target_cell_selected = f"target cell: {self.target_track_of_interest}\n"
		target_cell_class = f"class: {self.df_targets.loc[self.df_targets['TRACK_ID']==self.target_track_of_interest, self.target_class_name].to_numpy()[0]}\n"
		target_cell_time = f"time of interest: {self.df_targets.loc[self.df_targets['TRACK_ID']==self.target_track_of_interest, self.target_time_name].to_numpy()[0]}\n"

		self.target_cell_info.setText(target_cell_selected+target_cell_class+target_cell_time)#+effector_cell_selected+effector_cell_class+effector_cell_time)

	def give_reference_cell_information(self):

		reference_cell_selected = f"reference cell: {self.reference_track_of_interest}\n"
		reference_cell_population = f"population: {self.reference_population}\n"
		self.reference_cell_info.setText(reference_cell_selected+reference_cell_population)#+effector_cell_selected+effector_cell_class+effector_cell_time+target_cell_class+target_cell_time))

	def give_neighbor_cell_information(self):
		neighbor_cell_selected = f"neighbor cell: {self.neighbor_track_of_interest}\n"
		neighbor_cell_population = f"population: {self.neighbor_population}\n"
		neighbor_cell_time = f"time of interest: {self.df_relative.loc[(self.df_relative['REFERENCE_ID']==self.reference_track_of_interest)&(self.df_relative['NEIGHBOR_ID']==self.neighbor_track_of_interest), self.pair_time_name].to_numpy()[0]}\n"
		neighbor_cell_class = f"class: {self.df_relative.loc[(self.df_relative['REFERENCE_ID']==self.reference_track_of_interest)&(self.df_relative['NEIGHBOR_ID']==self.neighbor_track_of_interest), self.pair_class_name].to_numpy()[0]}\n"
		#reference_cell_class = f"class: {self.df_targets.loc[self.df_targets['TRACK_ID']==self.target_track_of_interest, self.target_class_name].to_numpy()[0]}\n"
		#target_cell_time = f"time of interest: {self.df_targets.loc[self.df_targets['TRACK_ID']==self.target_track_of_interest, self.target_time_name].to_numpy()[0]}\n"

		self.neighbor_cell_info.setText(neighbor_cell_selected+neighbor_cell_population+neighbor_cell_class+neighbor_cell_time)#+effector_cell_selected+effector_cell_class+effector_cell_time+target_cell_class+target_cell_time))

	#def hide_neighbor_cell_info(self):
		#neighbor_cell_selected.hide()
		#neighbor_cell_population.hide()
	def hide_target_cell_info(self):

		self.target_cell_info.setText('')

	def give_effector_cell_information(self):
		self.effector_cell_info.setSpacing(0)
		self.effector_cell_info.setContentsMargins(0, 20, 0, 30)
		self.neigh_eff_combo=QComboBox()
		#self.neighb_eff_combo.addItems(self.df_relative.loc[(self.df_relative['target']==self.target_track_of_interest),'effecor'])
		neighs=self.df_relative.loc[(self.df_relative['REFERENCE_ID']==self.target_track_of_interest),'NEIGHBOR_ID'].to_numpy()
		neighs=np.unique(neighs)
		for effector in neighs:
			self.neigh_eff_combo.addItem(str(effector))
		if self.effector_track_of_interest not in neighs:
			self.neigh_eff_combo.addItem(str(self.effector_track_of_interest))
		self.neigh_eff_combo.setCurrentText(str(self.effector_track_of_interest))
		self.eff_cell_sel=QHBoxLayout()
		#effector_cell_selected = f"effector cell: {self.effector_track_of_interest}"
		self.effector_cell_selected = f"effector cell: "
		self.eff_cell = QLabel(self.effector_cell_selected)
		# self.eff_cell_sel.removeWidget(self.eff_cell)
		# self.eff_cell_sel.removeWidget(self.neigh_eff_combo)
		self.eff_cell_sel.addWidget(self.eff_cell)
		self.eff_cell_sel.addWidget(self.neigh_eff_combo, alignment=Qt.AlignLeft)
		try:
			self.effector_cell_class = f"class: {self.df_effectors.loc[self.df_effectors['TRACK_ID']==self.effector_track_of_interest, self.effector_class_name].to_numpy()[0]}"
		except:
			self.effector_cell_class = f"class: {self.df_effectors.loc[self.df_effectors['ID'] == self.effector_track_of_interest, self.effector_class_name].to_numpy()[0]}"

		self.eff_cls = QLabel(self.effector_cell_class)
		try:
			self.effector_cell_time = f"time of interest: {self.df_effectors.loc[self.df_effectors['TRACK_ID']==self.effector_track_of_interest, self.effector_time_name].to_numpy()[0]}"
		except:
			self.effector_cell_time = f"time of interest: {self.df_effectors.loc[self.df_effectors['ID']==self.effector_track_of_interest, self.effector_time_name].to_numpy()[0]}"

		self.eff_tm=QLabel(self.effector_cell_time)
		# try:
		#     self.effector_probabilty = f"probability: {self.df_relative.loc[(self.df_relative['REFERENCE_ID']==self.target_track_of_interest)&(self.df_relative['NEIGHBOR_ID']==self.effector_track_of_interest),'probability'].to_numpy()[0]}"
		# except:
		#     self.effector_probabilty=f"probability: 0"
		# self.eff_prb=QLabel(self.effector_probabilty)
		#self.effector_cell_info.setText(effector_cell_selected+effector_cell_class+effector_cell_time+effector_probabilty)
		# self.effector_cell_info.removeWidget(self.eff_cls)
		# self.effector_cell_info.removeWidget(self.eff_tm)
		# self.effector_cell_info.removeWidget(self.eff_prb)
		self.effector_cell_info.addLayout(self.eff_cell_sel)
		self.effector_cell_info.addWidget(self.eff_cls)
		self.effector_cell_info.addWidget(self.eff_tm)
		#self.effector_cell_info.addWidget(self.eff_prb)
		self.neigh_eff_combo.currentIndexChanged.connect(self.update_effector_info)
		self.eff_info_to_hide=[self.eff_cell,self.neigh_eff_combo,self.eff_cls,self.eff_tm]#self.eff_prb





	def hide_effector_cell_info(self):
		self.eff_cls.clear()
		self.eff_tm.clear()
		#self.eff_prb.clear()

		for info in self.eff_info_to_hide:
			info.hide()


	def save_trajectories(self):

		#pass

		# if self.normalized_signals:
		# 	self.normalize_features_btn.click()
		# if self.selection:
		# 	self.cancel_selection()
		self.relative_class_name = self.relative_class_choice_cb.currentText()
		self.df_relative = self.df_relative.drop(self.df_relative[self.df_relative[self.relative_class_name]>2].index)
		self.df_relative.to_csv(self.relative_trajectories_path, index=False)
		print('relative table saved.')

		#self.extract_scatter_from_effector_trajectories()
		#self.give_cell_information()


	# def interval_slider_action(self):

	# 	print(dir(self.anim.event_source))

	# 	self.anim.event_source.interval = self.interval_slider.value()
	# 	self.anim.event_source._timer_set_interval()

	def set_last_frame(self):

		self.last_frame_btn.setEnabled(False)
		self.last_frame_btn.disconnect()

		self.last_key = len(self.stack) - 1
		while len(np.where(self.stack[self.last_key].flatten()==0)[0]) > 0.99*len(self.stack[self.last_key].flatten()):
			self.last_key -= 1
		print(f'Last frame is {len(self.stack) - 1}; last not black is {self.last_key}')
		self.anim._drawn_artists = self.draw_frame(self.last_key)
		self.anim._drawn_artists = sorted(self.anim._drawn_artists, key=lambda x: x.get_zorder())
		for a in self.anim._drawn_artists:
			a.set_visible(True)

		self.fig.canvas.draw()
		self.anim.event_source.stop()

		#self.cell_plot.draw()
		self.stop_btn.hide()
		self.start_btn.show()
		self.stop_btn.clicked.connect(self.start)
		self.start_btn.setShortcut(QKeySequence("l"))

	def set_first_frame(self):

		self.first_frame_btn.setEnabled(False)
		self.first_frame_btn.disconnect()

		self.first_key = 0
		print(f'First frame is {0}')
		self.anim._drawn_artists = self.draw_frame(0)
		self.anim._drawn_artists = sorted(self.anim._drawn_artists, key=lambda x: x.get_zorder())
		for a in self.anim._drawn_artists:
			a.set_visible(True)

		self.fig.canvas.draw()
		self.anim.event_source.stop()

		#self.cell_plot.draw()
		self.stop_btn.hide()
		self.start_btn.show()
		self.stop_btn.clicked.connect(self.start)
		self.start_btn.setShortcut(QKeySequence("f"))

	def export_signals(self):

		auto_dataset_name = self.pos.split(os.sep)[-4] + '_' + self.pos.split(os.sep)[-2] + '.npy'

		if self.normalized_signals:
			self.normalize_features_btn.click()

		training_set = []
		cols_relative=self.df_relative.columns
		relative_filtered=self.df_relative[self.df_relative[f'{self.neighborhood_choice_cb.currentText()}'] == 1]
		tracks_reference=np.unique(relative_filtered["REFERENCE_ID"].to_numpy())
		# if self.reference_population=='targets':
		#     cols_reference= self.df_targets.columns
		#     tracks_reference = np.unique(self.df_targets["TRACK_ID"].to_numpy())
		# else:
		#     cols_reference= self.df_effectors.columns
		#     tracks_reference = np.unique(self.df_effectors["TRACK_ID"].to_numpy())
		# if self.neighbor_population=='targets':
		#     cols_neigh = self.df_targets.columns
		#     tracks_neigh = np.unique(self.df_targets["TRACK_ID"].to_numpy())
		# else:
		#     cols_neigh = self.df_effectors.columns
		#     tracks_neigh = np.unique(self.df_effectors["TRACK_ID"].to_numpy())
		if self.reference_population == 'targets':
			df_ref=self.df_targets
			cols_ref = self.df_targets.columns

		else:
			df_ref=self.df_effectors
			cols_ref = self.df_effectors.columns
		if self.neighbor_population == 'targets':
			df_neigh=self.df_targets
			cols_neigh = self.df_targets.columns
		else:
			df_neigh=self.df_effectors
			cols_neigh = self.df_effectors.columns
		for track in tracks_reference:
			# Add all signals at given track

			tracks_neigh = np.unique(relative_filtered.loc[relative_filtered["REFERENCE_ID"]==track,'NEIGHBOR_ID'].to_numpy())
			for neigh in tracks_neigh:
				signals = {}
				for c in cols_relative:
					signals.update({'relative_'+c: relative_filtered.loc[(relative_filtered["REFERENCE_ID"] == track)&(relative_filtered["NEIGHBOR_ID"]==neigh), c].to_numpy()})
				time_of_interest = relative_filtered.loc[(relative_filtered["REFERENCE_ID"] == track)&(relative_filtered["NEIGHBOR_ID"] == neigh), self.pair_time_name].to_numpy()[0]
				cclass = relative_filtered.loc[(relative_filtered["REFERENCE_ID"] == track)&(relative_filtered["NEIGHBOR_ID"]==neigh), self.pair_class_name].to_numpy()[0]
				signals.update({"time_of_interest": time_of_interest, "class": cclass})
				for c in cols_ref:
					signals.update({'reference_'+c: df_ref.loc[(df_ref["TRACK_ID"] == track), c].to_numpy()})
				for c in cols_neigh:
					signals.update({'neighbor_' + c: df_neigh.loc[(df_neigh["TRACK_ID"] == neigh), c].to_numpy()})

			# Here auto add all available channels
				training_set.append(signals)

		#print(training_set)

		pathsave = QFileDialog.getSaveFileName(self, "Select file name", self.exp_dir + auto_dataset_name, ".npy")[0]
		if pathsave != '':
		   if not pathsave.endswith(".npy"):
			   pathsave += ".npy"
		   try:
			   np.save(pathsave, training_set)
			   print(f'File successfully written in {pathsave}.')
		   except Exception as e:
			   print(f"Error {e}...")

	def update_effector_info(self):
		# Clear existing labels
		self.eff_cls.clear()
		self.eff_tm.clear()
		#self.eff_prb.clear()
		self.effector_loc_t=[]
		self.effector_loc_idx=[]
		for t in range(len(self.effector_tracks)):
			indices = np.where(self.effector_tracks[t] == self.effector_track_of_interest)[0]
			if len(indices) > 0:
				self.effector_loc_t.append(t)
				self.effector_loc_idx.append(indices[0])

		self.effector_previous_color = []
		for t, idx in zip(self.effector_loc_t, self.effector_loc_idx):
			self.effector_previous_color.append(self.effector_colors[t][idx].copy())
			self.effector_colors[t][idx] = 'salmon'
		# Get the selected effector cell
		self.effector_track_of_interest = float(self.neigh_eff_combo.currentText())

		# Get information for the selected effector cell
		try:
			effector_class = self.df_effectors.loc[self.df_effectors['TRACK_ID'] == self.effector_track_of_interest, self.effector_class_name].to_numpy()[0]
		except:
			effector_class = 0
		try:
			effector_time = \
			self.df_effectors.loc[self.df_effectors['TRACK_ID'] == self.effector_track_of_interest, self.effector_time_name].to_numpy()[0]
		except:
			effector_time = \
			self.df_effectors.loc[self.df_effectors['ID'] == self.effector_track_of_interest, self.effector_time_name].to_numpy()[0]

		# try:
		#     effector_probability = self.df_relative.loc[
		#         (self.df_relative['REFERENCE_ID'] == self.target_track_of_interest) & (
		#                     self.df_relative['NEIGHBOR_ID'] == self.effector_track_of_interest), 'probability'].to_numpy()[0]
		# except IndexError:
		#     effector_probability = 0

		# Update labels with new information
		self.eff_cls.setText(f"class: {effector_class}")
		self.eff_tm.setText(f"time of interest: {effector_time}")
		#self.eff_prb.setText(f"probability: {effector_probability}")
		self.effector_loc_t=[]
		self.effector_loc_idx=[]
		for t in range(len(self.effector_tracks)):
			indices = np.where(self.effector_tracks[t] == self.effector_track_of_interest)[0]
			if len(indices) > 0:
				self.effector_loc_t.append(t)
				self.effector_loc_idx.append(indices[0])

		self.effector_previous_color = []
		for t, idx in zip(self.effector_loc_t, self.effector_loc_idx):
			self.effector_previous_color.append(self.effector_colors[t][idx].copy())
			self.effector_colors[t][idx] = 'magenta'

		self.plot_signals()


		# auto_dataset_name = self.pos.split(os.sep)[-4]+'_'+self.pos.split(os.sep)[-2]+'.npy'

		# if self.normalized_signals:
		# 	self.normalize_features_btn.click()

		# training_set = []
		# cols = self.df_tracks.columns
		# tracks = np.unique(self.df_tracks["TRACK_ID"].to_numpy())

		# for track in tracks:
		# 	# Add all signals at given track
		# 	signals = {}
		# 	for c in cols:
		# 		signals.update({c: self.df_tracks.loc[self.df_tracks["TRACK_ID"]==track, c].to_numpy()})
		# 	time_of_interest = self.df_tracks.loc[self.df_tracks["TRACK_ID"]==track, self.time_name].to_numpy()[0]
		# 	cclass = self.df_tracks.loc[self.df_tracks["TRACK_ID"]==track, self.class_name].to_numpy()[0]
		# 	signals.update({"time_of_interest": time_of_interest,"class": cclass})
		# 	# Here auto add all available channels
		# 	training_set.append(signals)

		# print(training_set)

		# pathsave = QFileDialog.getSaveFileName(self, "Select file name", self.exp_dir+auto_dataset_name, ".npy")[0]
		# if pathsave!='':
		# 	if not pathsave.endswith(".npy"):
		# 		pathsave += ".npy"
		# 	try:
		# 		np.save(pathsave,training_set)
		# 		print(f'File successfully written in {pathsave}.')
		# 	except Exception as e:
		# 		print(f"Error {e}...")

	# def normalize_features(self):
	# 	if self.df_targets is None or self.df_effectors is None:
	# 		print("Error: DataFrame is not initialized.")
	# 		return
	# 	x_targets = self.df_targets[self.columns_to_rescale_targets].values
	# 	x_effectors = self.df_effectors[self.columns_to_rescale_effectors].values
	#
	# 	if not self.normalized_signals:
	# 		x_target_normalized = self.MinMaxScaler_targets.fit_transform(x_targets)
	# 		x_effector_normalized = self.MinMaxScaler_effectors.fit_transform(x_effectors)
	# 		self.df_targets[self.columns_to_rescale_targets]=x_target_normalized
	# 		self.df_effectors[self.columns_to_rescale_effectors]=x_effector_normalized
	# 		self.plot_signals()
	# 		self.normalized_signals = True
	# 		self.normalize_features_btn.setIcon(icon(MDI6.arrow_collapse_vertical,color="#1565c0"))
	# 		self.normalize_features_btn.setIconSize(QSize(25, 25))
	# 		print("Features normalized.")
	# 	else:
	# 		x_target_inverse = self.MinMaxScaler_targets.inverse_transform(x_targets)
	# 		x_effectors_inverse = self.MinMaxScaler_effectors.inverse_transform(x_effectors)
	# 		self.df_targets[self.columns_to_rescale_targets] = x_target_inverse
	# 		self.df_effectors[self.columns_to_rescale_effectors] = x_effectors_inverse
	# 		self.plot_signals()
	# 		self.normalized_signals = False
	# 		self.normalize_features_btn.setIcon(icon(MDI6.arrow_collapse_vertical,color="black"))
	# 		self.normalize_features_btn.setIconSize(QSize(25, 25))
	# 		print("Features unnormalized.")

	def normalize_features(self):
		self.MinMaxScaler = MinMaxScaler()
		if self.df_targets is None or self.df_effectors is None:
			print("Error: DataFrame is not initialized.")
			return
		df_tracks1_renamed = self.df_targets[self.columns_to_rescale_targets].rename(columns=lambda x: x + '_1')
		df_tracks2_renamed = self.df_effectors[self.columns_to_rescale_effectors].rename(columns=lambda x: x + '_2')

		columns_to_rescale1_renamed = [col + '_1' for col in self.columns_to_rescale_targets]
		columns_to_rescale2_renamed = [col + '_2' for col in self.columns_to_rescale_effectors]
		columns_to_rescale_renamed = []
		for i in columns_to_rescale1_renamed:
			columns_to_rescale_renamed.append(i)
		for i in columns_to_rescale2_renamed:
			columns_to_rescale_renamed.append(i)
		for i in self.columns_to_rescale_relative:
			columns_to_rescale_renamed.append(i)
		self.merged_df = pd.concat([df_tracks1_renamed, df_tracks2_renamed,self.df_relative[self.columns_to_rescale_relative]], axis=1)

		x = self.merged_df[columns_to_rescale_renamed].values
		if not self.normalized_signals:
			self.MinMaxScaler.fit(self.merged_df.values)
			x_normalized = self.MinMaxScaler_targets.fit_transform(x)
			self.merged_df[columns_to_rescale_renamed] = x_normalized
			self.df_targets[self.columns_to_rescale_targets] = self.merged_df[columns_to_rescale1_renamed]
			self.df_effectors[self.columns_to_rescale_effectors] = self.merged_df[columns_to_rescale2_renamed]
			self.df_relative[self.columns_to_rescale_relative]=self.merged_df[self.columns_to_rescale_relative]
			self.plot_signals()
			self.normalized_signals = True
			self.normalize_features_btn.setIcon(icon(MDI6.arrow_collapse_vertical,color="#1565c0"))
			self.normalize_features_btn.setIconSize(QSize(25, 25))
			print("Features normalized.")
		else:
			x_inverse=self.MinMaxScaler_targets.inverse_transform(x)
			self.merged_df[columns_to_rescale_renamed]=x_inverse
			self.df_targets[self.columns_to_rescale_targets]=self.merged_df[columns_to_rescale1_renamed]
			self.df_effectors[self.columns_to_rescale_effectors]=self.merged_df[columns_to_rescale2_renamed]
			self.df_relative[self.columns_to_rescale_relative]=self.merged_df[self.columns_to_rescale_relative]
			self.plot_signals()
			self.normalized_signals = False
			self.normalize_features_btn.setIcon(icon(MDI6.arrow_collapse_vertical,color="black"))
			self.normalize_features_btn.setIconSize(QSize(25, 25))
			print("Features unnormalized.")


	def switch_to_log(self):

		"""
		Better would be to create a log(quantity) and plot it...
		"""

		try:
			if self.cell_ax.get_yscale()=='linear':
				self.cell_ax.set_yscale('log')
				self.log_btn.setIcon(icon(MDI6.math_log,color="#1565c0"))
			else:
				self.cell_ax.set_yscale('linear')
				self.log_btn.setIcon(icon(MDI6.math_log,color="black"))
		except Exception as e:
			print(e)

		#self.cell_ax.autoscale()
		self.cell_fcanvas.canvas.draw_idle()

	def signal_button_changed1(self):
		self.target_signal_choice.clear()
		if self.target_button1.isChecked():
			self.target_signal_choice.addItems(['--']+self.target_signals)
		if self.effector_button1.isChecked():
			self.target_signal_choice.addItems(['--']+self.effector_signals)
		if self.relative_button1.isChecked():
			self.target_signal_choice.addItems(['--']+self.relative_signals)

	def signal_button_changed2(self):
		self.effector_signal_choice.clear()
		if self.target_button2.isChecked():
			self.effector_signal_choice.addItems(['--'] + self.target_signals)
		if self.effector_button2.isChecked():
			self.effector_signal_choice.addItems(['--'] + self.effector_signals)
		if self.relative_button2.isChecked():
			self.effector_signal_choice.addItems(['--'] + self.relative_signals)

	def signal_button_changed3(self):
		self.relative_signal_choice.clear()
		if self.target_button3.isChecked():
			self.relative_signal_choice.addItems(['--'] + self.target_signals)
		if self.effector_button3.isChecked():
			self.relative_signal_choice.addItems(['--'] + self.effector_signals)
		if self.relative_button3.isChecked():
			self.relative_signal_choice.addItems(['--'] + self.relative_signals)