from PyQt5.QtWidgets import QMainWindow, QComboBox, QLabel, QRadioButton, QLineEdit,QFileDialog, QApplication, QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QAction, QShortcut, QLineEdit
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QKeySequence
from celldetective.gui.gui_utils import center_window, QHSeperationLine, FilterChoice
from superqt import QLabeledDoubleSlider, QLabeledDoubleRangeSlider, QLabeledSlider
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

class SignalAnnotator2(QMainWindow):
	
	"""
	UI to set tracking parameters for bTrack.

	"""

	def __init__(self, parent=None):
		
		super().__init__()
		self.parent = parent
		self.setWindowTitle("Signal annotator")

		self.pos = self.parent.parent.pos
		self.exp_dir = self.parent.exp_dir
		print(f'{self.pos=} {self.exp_dir=}')

		self.soft_path = get_software_location()
		self.recently_modified = False
		self.n_signals = 3
		self.target_selection = []
		self.effector_selection = []

		# Read instructions from target block for now...
		self.mode = "targets"
		self.instructions_path = self.exp_dir + "configs/signal_annotator_config_targets.json"
		#self.trajectories_path = self.pos+'output/tables/trajectories_targets.csv'

		self.screen_height = self.parent.parent.parent.screen_height
		self.screen_width = self.parent.parent.parent.screen_width

		# default params
		self.target_class_name = 'class'
		self.target_time_name = 't0'
		self.target_status_name = 'status'

		center_window(self)

		self.locate_stack()
		self.load_annotator_config()

		self.locate_target_tracks()
		self.locate_effector_tracks()

		self.prepare_stack()

		self.generate_signal_choices()
		self.frame_lbl = QLabel('frame: ')
		self.looped_animation()
		self.create_cell_signal_canvas()


		self.populate_widget()

		self.setMinimumWidth(int(0.8*self.screen_width))
		# self.setMaximumHeight(int(0.8*self.screen_height))
		self.setMinimumHeight(int(0.8*self.screen_height))
		# self.setMaximumHeight(int(0.8*self.screen_height))

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


		# TARGETS
		class_hbox = QHBoxLayout()
		class_hbox.addWidget(QLabel('event: '), 25)
		self.target_class_choice_cb = QComboBox()

		cols = np.array(self.df_targets.columns)
		self.target_class_cols = np.array([c.startswith('class') for c in list(self.df_targets.columns)])
		self.target_class_cols = list(cols[self.target_class_cols])
		try:
			self.target_class_cols.remove('class_id')
		except Exception:
			pass
		try:
			self.target_class_cols.remove('class_color')
		except Exception:
			pass

		self.target_class_choice_cb.addItems(self.target_class_cols)
		self.target_class_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors_targets)
		self.target_class_choice_cb.setCurrentIndex(0)

		class_hbox.addWidget(self.target_class_choice_cb, 70)

		# self.target_add_class_btn = QPushButton('')
		# self.target_add_class_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		# self.target_add_class_btn.setIcon(icon(MDI6.plus,color="black"))
		# self.target_add_class_btn.setToolTip("Add a new event class")
		# self.target_add_class_btn.setIconSize(QSize(20, 20))
		# self.target_add_class_btn.clicked.connect(self.create_new_event_class)
		# class_hbox.addWidget(self.add_class_btn, 5)

		# self.del_class_btn = QPushButton('')
		# self.del_class_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		# self.del_class_btn.setIcon(icon(MDI6.delete,color="black"))
		# self.del_class_btn.setToolTip("Delete an event class")
		# self.del_class_btn.setIconSize(QSize(20, 20))
		# self.del_class_btn.clicked.connect(self.del_event_class)
		# class_hbox.addWidget(self.del_class_btn, 5)

		self.left_panel.addLayout(class_hbox)


		# EFFECTORS
		class_hbox = QHBoxLayout()
		class_hbox.addWidget(QLabel('event: '), 25)
		self.effector_class_choice_cb = QComboBox()

		cols = np.array(self.df_effectors.columns)
		self.effector_class_cols = np.array([c.startswith('class') for c in list(self.df_effectors.columns)])
		self.effector_class_cols = list(cols[self.effector_class_cols])
		try:
			self.effector_class_cols.remove('class_id')
		except Exception:
			pass
		try:
			self.effector_class_cols.remove('class_color')
		except Exception:
			pass

		self.effector_class_choice_cb.addItems(self.effector_class_cols)
		self.effector_class_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors_effectors)
		self.effector_class_choice_cb.setCurrentIndex(0)

		class_hbox.addWidget(self.effector_class_choice_cb, 70)

		# self.add_class_btn = QPushButton('')
		# self.add_class_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		# self.add_class_btn.setIcon(icon(MDI6.plus,color="black"))
		# self.add_class_btn.setToolTip("Add a new event class")
		# self.add_class_btn.setIconSize(QSize(20, 20))
		# self.add_class_btn.clicked.connect(self.create_new_event_class)
		# class_hbox.addWidget(self.add_class_btn, 5)

		# self.del_class_btn = QPushButton('')
		# self.del_class_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		# self.del_class_btn.setIcon(icon(MDI6.delete,color="black"))
		# self.del_class_btn.setToolTip("Delete an event class")
		# self.del_class_btn.setIconSize(QSize(20, 20))
		# self.del_class_btn.clicked.connect(self.del_event_class)
		# class_hbox.addWidget(self.del_class_btn, 5)

		self.left_panel.addLayout(class_hbox)

		self.cell_info = QLabel('')
		self.left_panel.addWidget(self.cell_info)

		# Annotation buttons
		options_hbox = QHBoxLayout()
		options_hbox.setContentsMargins(150,30,50,0)
		self.event_btn = QRadioButton('event')
		self.event_btn.setStyleSheet(self.parent.parent.parent.button_style_sheet_2)
		self.event_btn.toggled.connect(self.enable_time_of_interest)

		self.no_event_btn = QRadioButton('no event')
		self.no_event_btn.setStyleSheet(self.parent.parent.parent.button_style_sheet_2)
		self.no_event_btn.toggled.connect(self.enable_time_of_interest)

		self.else_btn = QRadioButton('else')
		self.else_btn.setStyleSheet(self.parent.parent.parent.button_style_sheet_2)
		self.else_btn.toggled.connect(self.enable_time_of_interest)

		self.suppr_btn = QRadioButton('mark for\nsuppression')
		self.suppr_btn.setStyleSheet(self.parent.parent.parent.button_style_sheet_2)
		self.suppr_btn.toggled.connect(self.enable_time_of_interest)

		options_hbox.addWidget(self.event_btn, 25)
		options_hbox.addWidget(self.no_event_btn, 25)
		options_hbox.addWidget(self.else_btn, 25)
		options_hbox.addWidget(self.suppr_btn, 25)
		self.left_panel.addLayout(options_hbox)

		time_option_hbox = QHBoxLayout()
		time_option_hbox.setContentsMargins(100,30,100,30)
		self.time_of_interest_label = QLabel('time of interest: ')
		time_option_hbox.addWidget(self.time_of_interest_label, 30)
		self.time_of_interest_le = QLineEdit()
		time_option_hbox.addWidget(self.time_of_interest_le, 70)
		self.left_panel.addLayout(time_option_hbox)

		main_action_hbox = QHBoxLayout()
		self.correct_btn = QPushButton('correct')
		self.correct_btn.setIcon(icon(MDI6.redo_variant,color="white"))
		self.correct_btn.setIconSize(QSize(20, 20))
		self.correct_btn.setStyleSheet(self.parent.parent.parent.button_style_sheet)
		self.correct_btn.clicked.connect(self.show_annotation_buttons)
		self.correct_btn.setEnabled(False)
		main_action_hbox.addWidget(self.correct_btn)

		self.cancel_btn = QPushButton('cancel')
		self.cancel_btn.setStyleSheet(self.parent.parent.parent.button_style_sheet_2)
		self.cancel_btn.setShortcut(QKeySequence("Esc"))
		self.cancel_btn.setEnabled(False)
		self.cancel_btn.clicked.connect(self.cancel_selection)
		main_action_hbox.addWidget(self.cancel_btn)
		self.left_panel.addLayout(main_action_hbox)

		self.annotation_btns_to_hide = [self.event_btn, self.no_event_btn, 
										self.else_btn, self.time_of_interest_label, 
										self.time_of_interest_le, self.suppr_btn]
		self.hide_annotation_buttons()
		#### End of annotation buttons


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
		self.normalize_features_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
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
		self.log_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.log_btn.clicked.connect(self.switch_to_log)
		plot_buttons_hbox.addWidget(self.log_btn, 5)

		self.left_panel.addLayout(plot_buttons_hbox)

		signal_choice_vbox = QVBoxLayout()
		signal_choice_vbox.setContentsMargins(30,0,30,50)
		for i in range(len(self.signal_choice_targets_cb)):
			
			hlayout = QHBoxLayout()
			hlayout.addWidget(self.signal_choice_targets_label[i], 20)
			hlayout.addWidget(self.signal_choice_targets_cb[i], 75)
			#hlayout.addWidget(self.log_btns[i], 5)
			signal_choice_vbox.addLayout(hlayout)

			# self.log_btns[i].setIcon(icon(MDI6.math_log,color="black"))
			# self.log_btns[i].setStyleSheet(self.parent.parent.parent.button_select_all)
			# self.log_btns[i].clicked.connect(lambda ch, i=i: self.switch_to_log(i))

		self.left_panel.addLayout(signal_choice_vbox)

		btn_hbox = QHBoxLayout()
		self.save_btn = QPushButton('Save')
		self.save_btn.setStyleSheet(self.parent.parent.parent.button_style_sheet)
		self.save_btn.clicked.connect(self.save_trajectories)
		btn_hbox.addWidget(self.save_btn, 90)

		self.export_btn = QPushButton('')
		self.export_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
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
		self.first_frame_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.first_frame_btn.setFixedSize(QSize(60, 60))
		self.first_frame_btn.setIconSize(QSize(30, 30))		



		self.last_frame_btn = QPushButton()
		self.last_frame_btn.clicked.connect(self.set_last_frame)
		self.last_frame_btn.setShortcut(QKeySequence('l'))
		self.last_frame_btn.setIcon(icon(MDI6.page_last,color="black"))
		self.last_frame_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.last_frame_btn.setFixedSize(QSize(60, 60))
		self.last_frame_btn.setIconSize(QSize(30, 30))		

		self.stop_btn = QPushButton()
		self.stop_btn.clicked.connect(self.stop)
		self.stop_btn.setIcon(icon(MDI6.stop,color="black"))
		self.stop_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
		self.stop_btn.setFixedSize(QSize(60, 60))
		self.stop_btn.setIconSize(QSize(30, 30))


		self.start_btn = QPushButton()
		self.start_btn.clicked.connect(self.start)
		self.start_btn.setIcon(icon(MDI6.play,color="black"))
		self.start_btn.setFixedSize(QSize(60, 60))
		self.start_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
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
			self.contrast_slider.setRange(*[np.nanpercentile(self.stack.flatten(), 0.001), np.nanpercentile(self.stack.flatten(), 99.999)])
			self.contrast_slider.setValue([np.percentile(self.stack.flatten(), 1), np.percentile(self.stack.flatten(), 99.99)])
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

		#self.populate_left_panel()
		#grid.addLayout(self.left_side, 0, 0, 1, 1)

		main_layout.addLayout(self.left_panel, 35)
		main_layout.addLayout(self.right_panel, 65)
		self.button_widget.adjustSize()

		self.setCentralWidget(self.button_widget)
		self.show()

		QApplication.processEvents()

	# def del_event_class(self):

	# 	msgBox = QMessageBox()
	# 	msgBox.setIcon(QMessageBox.Warning)
	# 	msgBox.setText(f"You are about to delete event class {self.class_choice_cb.currentText()}. The associated time and\nstatus will also be deleted. Do you still want to proceed?")
	# 	msgBox.setWindowTitle("Warning")
	# 	msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
	# 	returnValue = msgBox.exec()
	# 	if returnValue == QMessageBox.No:
	# 		return None
	# 	else:
	# 		class_to_delete = self.class_choice_cb.currentText()
	# 		time_to_delete = class_to_delete.replace('class','t')
	# 		status_to_delete = class_to_delete.replace('class', 'status')
	# 		cols_to_delete = [class_to_delete, time_to_delete, status_to_delete]
	# 		for c in cols_to_delete:
	# 			try:
	# 				self.df_tracks = self.df_tracks.drop([c], axis=1)
	# 			except Exception as e:
	# 				print(e)
	# 		item_idx = self.class_choice_cb.findText(class_to_delete)
	# 		self.class_choice_cb.removeItem(item_idx)



	def create_new_event_class(self):

		# display qwidget to name the event
		self.newClassWidget = QWidget()
		self.newClassWidget.setWindowTitle('Create new event class')
		
		layout = QVBoxLayout()
		self.newClassWidget.setLayout(layout)
		name_hbox = QHBoxLayout()
		name_hbox.addWidget(QLabel('event name: '), 25)
		self.class_name_le = QLineEdit('event')
		name_hbox.addWidget(self.class_name_le, 75)
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

		submit_btn.clicked.connect(self.write_new_event_class)
		cancel_btn.clicked.connect(self.close_without_new_class)

		self.newClassWidget.show()
		center_window(self.newClassWidget)


		# Prefill with class value
		# write in table

	# def write_new_event_class(self):
		

	# 	if self.class_name_le.text()=='':
	# 		self.target_class = 'class'
	# 		self.target_time = 't0'
	# 	else:
	# 		self.target_class = 'class_'+self.class_name_le.text()
	# 		self.target_time = 't_'+self.class_name_le.text()

	# 	if self.target_class in list(self.df_tracks.columns):

	# 		msgBox = QMessageBox()
	# 		msgBox.setIcon(QMessageBox.Warning)
	# 		msgBox.setText("This event name already exists. If you proceed,\nall annotated data will be rewritten. Do you wish to continue?")
	# 		msgBox.setWindowTitle("Warning")
	# 		msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
	# 		returnValue = msgBox.exec()
	# 		if returnValue == QMessageBox.No:
	# 			return None
	# 		else:
	# 			pass

	# 	fill_option = np.where([c.isChecked() for c in self.class_option_rb])[0][0]
	# 	self.df_tracks.loc[:,self.target_class] = fill_option
	# 	if fill_option==0:
	# 		self.df_tracks.loc[:,self.target_time] = 0.1
	# 	else:
	# 		self.df_tracks.loc[:,self.target_time] = -1
		
	# 	self.class_choice_cb.clear()
	# 	cols = np.array(self.df_tracks.columns)
	# 	self.class_cols = np.array([c.startswith('class') for c in list(self.df_tracks.columns)])
	# 	self.class_cols = list(cols[self.class_cols])
	# 	self.class_cols.remove('class_id')
	# 	self.class_cols.remove('class_color')
	# 	self.class_choice_cb.addItems(self.class_cols)
	# 	idx = self.class_choice_cb.findText(self.target_class)
	# 	self.class_choice_cb.setCurrentIndex(idx)

	# 	self.newClassWidget.close()	


	def close_without_new_class(self):
		self.newClassWidget.close()

	def compute_status_and_colors_targets(self, i):

		self.target_class_name = self.target_class_choice_cb.currentText()
		self.expected_target_status = 'status'
		suffix = self.target_class_name.replace('class','').replace('_','')
		if suffix!='':
			self.expected_target_status+='_'+suffix
			self.expected_target_time = 't_'+suffix
		else:
			self.expected_target_time = 't0'

		self.time_target_name = self.expected_target_time
		self.status_target_name = self.expected_target_status

		print('selection and expected names: ', self.target_class_name, self.target_expected_time, self.target_expected_status)

		if self.time_target_name in self.df_targets.columns and self.target_class_name in self.df_targets.columns and not self.status_target_name in self.df_targets.columns:
			# only create the status column if it does not exist to not erase static classification results
			self.make_status_column_targets()
		elif self.time_target_name in self.df_targets.columns and self.target_class_name in self.df_targets.columns:
			# all good, do nothing
			pass
		else:
			if not self.status_name in self.df_targets.columns:
				self.df_targets[self.status_target_name] = 0
				self.df_targets['status_color'] = color_from_status(0)
				self.df_targets['class_color'] = color_from_class(1)

		if not self.target_class_name in self.df_targets.columns:
			self.df_targets[self.target_class_name] = 1
		if not self.time_target_name in self.df_targets.columns:
			self.df_targets[self.time_target_name] = -1

		self.df_targets['status_color'] = [color_from_status(i) for i in self.df_targets[self.status_target_name].to_numpy()]					
		self.df_targets['class_color'] = [color_from_class(i) for i in self.df_targets[self.target_class_name].to_numpy()]

		self.extract_scatter_from_target_trajectories()

	def compute_status_and_colors_effectors(self, i):

		self.effector_class_name = self.effector_class_choice_cb.currentText()
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
			self.make_status_column_effectors()
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


	def contrast_slider_action(self):

		"""
		Recontrast the imshow as the contrast slider is moved.
		"""

		self.vmin = self.contrast_slider.value()[0]
		self.vmax = self.contrast_slider.value()[1]
		self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
		self.fcanvas.canvas.draw_idle()

	def cancel_selection(self):

		self.hide_annotation_buttons()
		self.correct_btn.setEnabled(False)
		self.correct_btn.setText('correct')
		self.cancel_btn.setEnabled(False)
		
		try:
			self.target_selection.pop(0)
		except Exception as e:
			print(e)
		try:
			self.effector_selection.pop(0)
		except Exception as e:
			print(e)

		try:
			for k,(t,idx) in enumerate(zip(self.target_loc_t,self.target_loc_idx)):
				self.target_colors[t][idx,0] = self.target_previous_color[k][0]
				self.target_colors[t][idx,1] = self.target_previous_color[k][1]
			if self.neighbors != {}:
				for key in self.neighbors.keys():
					for value in self.neighbors[key]:
						self.effector_colors[key][value, 0] = self.initial_effector_colors[key][value, 0]
						self.effector_colors[key][value, 1] = self.initial_effector_colors[key][value, 1]
		except Exception as e:
			print(f'{e=}')

		try:
			for k,(t,idx) in enumerate(zip(self.effector_loc_t,self.effector_loc_idx)):
				self.effector_colors[t][idx,0] = self.initial_effector_colors[t][idx,0]
				self.effector_colors[t][idx,1] = self.initial_effector_colors[t][idx,1]



		except Exception as e:
			print(f'{e=}')

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

	def show_annotation_buttons(self):
		
		for a in self.annotation_btns_to_hide:
			a.show()

		cclass = self.df_tracks.loc[self.df_tracks['TRACK_ID']==self.track_of_interest, self.class_name].to_numpy()[0]
		t0 = self.df_tracks.loc[self.df_tracks['TRACK_ID']==self.track_of_interest, self.time_name].to_numpy()[0]

		if cclass==0:
			self.event_btn.setChecked(True)
			self.time_of_interest_le.setText(str(t0))
		elif cclass==1:
			self.no_event_btn.setChecked(True)
		elif cclass==2:
			self.else_btn.setChecked(True)
		elif cclass>2:
			self.suppr_btn.setChecked(True)

		self.enable_time_of_interest()
		self.correct_btn.setText('submit')

		self.correct_btn.disconnect()
		self.correct_btn.clicked.connect(self.apply_modification)

	def apply_modification(self):

		t0 = -1
		if self.event_btn.isChecked():
			cclass = 0
			try:
				t0 = float(self.time_of_interest_le.text().replace(',','.'))
				self.line_dt.set_xdata([t0,t0])
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

		self.df_tracks.loc[self.df_tracks['TRACK_ID']==self.track_of_interest, self.class_name] = cclass
		self.df_tracks.loc[self.df_tracks['TRACK_ID']==self.track_of_interest, self.time_name] = t0

		indices = self.df_tracks.loc[self.df_tracks['TRACK_ID']==self.track_of_interest, self.class_name].index
		timeline = self.df_tracks.loc[self.df_tracks['TRACK_ID']==self.track_of_interest, 'FRAME'].to_numpy()
		status = np.zeros_like(timeline)
		if t0 > 0:
			status[timeline>=t0] = 1.
		if cclass==2:
			status[:] = 2
		if cclass>2:
			status[:] = 42
		status_color = [color_from_status(s, recently_modified=True) for s in status]
		class_color = [color_from_class(cclass, recently_modified=True) for i in range(len(status))]

		self.df_tracks.loc[indices, self.status_name] = status
		self.df_tracks.loc[indices, 'status_color'] = status_color
		self.df_tracks.loc[indices, 'class_color'] = class_color

		#self.make_status_column()
		self.extract_scatter_from_target_trajectories()
		self.give_cell_information()

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
			self.selection.pop(0)
		except Exception as e:
			print(e)


		#self.fcanvas.canvas.draw()


	def locate_stack(self):
		
		"""
		Locate the target movie.

		"""

		movies = glob(self.pos + f"movie/{self.parent.parent.movie_prefix}*.tif")

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
			self.len_movie = self.parent.parent.len_movie
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
		self.target_trajectories_path = self.pos+f'output/tables/trajectories_{population}.csv'
		self.neigh_trajectories_path = self.pos+f'output/tables/trajectories_{population}.pkl'

		if not os.path.exists(self.target_trajectories_path):

			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("The trajectories cannot be detected.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Yes:
				self.close()
		else:

			# Load and prep tracks
			self.df_targets = pd.read_csv(self.target_trajectories_path)
			self.df_targets = self.df_targets.sort_values(by=['TRACK_ID', 'FRAME'])

			cols = np.array(self.df_targets.columns)
			self.target_class_cols = np.array([c.startswith('class') for c in list(self.df_targets.columns)])
			self.target_class_cols = list(cols[self.target_class_cols])
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

			self.MinMaxScaler = MinMaxScaler()
			self.columns_to_rescale = list(self.df_targets.columns)

			# is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
			# is_number_test = is_number(self.df_tracks.dtypes)
			# self.columns_to_rescale = [col for t,col in zip(is_number_test,self.df_tracks.columns) if t]
			# print(self.columns_to_rescale)
			
			cols_to_remove = ['status','status_color','class_color','TRACK_ID', 'FRAME','x_anim','y_anim','t', 'state', 'generation', 'root', 'parent', 'class_id', 'class', 't0', 'POSITION_X', 'POSITION_Y','position','well','well_index','well_name','pos_name','index','concentration','cell_type','antibody','pharmaceutical_agent'] + self.target_class_cols
			cols = np.array(list(self.df_targets.columns))
			time_cols = np.array([c.startswith('t_') for c in cols])
			time_cols = list(cols[time_cols])
			cols_to_remove += time_cols			

			for tr in cols_to_remove:
				try:
					self.columns_to_rescale.remove(tr)
				except:
					pass
					#print(f'column {tr} could not be found...')

			x = self.df_targets[self.columns_to_rescale].values
			self.MinMaxScaler.fit(x)

			#self.loc_t, self.loc_idx = np.where(self.tracks==self.track_of_interest)

	def locate_effector_tracks(self):

		population = 'effectors'
		self.effector_trajectories_path =  self.pos+f'output/tables/trajectories_{population}.csv'

		if not os.path.exists(self.effector_trajectories_path):

			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("The trajectories cannot be detected.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Yes:
				self.close()
		else:

			# Load and prep tracks
			self.df_effectors = pd.read_csv(self.effector_trajectories_path)
			self.df_effectors = self.df_effectors.sort_values(by=['TRACK_ID', 'FRAME'])

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
			self.effector_track_of_interest = self.df_effectors['TRACK_ID'].min()

			self.loc_t = []
			self.loc_idx = []
			for t in range(len(self.effector_tracks)):
				indices = np.where(self.effector_tracks[t]==self.effector_track_of_interest)[0]
				if len(indices)>0:
					self.loc_t.append(t)
					self.loc_idx.append(indices[0])

			self.MinMaxScaler = MinMaxScaler()
			self.columns_to_rescale = list(self.df_effectors.columns)

			# is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
			# is_number_test = is_number(self.df_tracks.dtypes)
			# self.columns_to_rescale = [col for t,col in zip(is_number_test,self.df_tracks.columns) if t]
			# print(self.columns_to_rescale)
			
			cols_to_remove = ['status','status_color','class_color','TRACK_ID', 'FRAME','x_anim','y_anim','t', 'state', 'generation', 'root', 'parent', 'class_id', 'class', 't0', 'POSITION_X', 'POSITION_Y','position','well','well_index','well_name','pos_name','index','concentration','cell_type','antibody','pharmaceutical_agent'] + self.effector_class_cols
			cols = np.array(list(self.df_effectors.columns))
			time_cols = np.array([c.startswith('t_') for c in cols])
			time_cols = list(cols[time_cols])
			cols_to_remove += time_cols			

			for tr in cols_to_remove:
				try:
					self.columns_to_rescale.remove(tr)
				except:
					pass
					#print(f'column {tr} could not be found...')

			x = self.df_effectors[self.columns_to_rescale].values
			self.MinMaxScaler.fit(x)

			#self.loc_t, self.loc_idx = np.where(self.tracks==self.track_of_interest)



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


	def make_effector_status_column(self):

		print('remaking the status column')
		for tid, group in self.df_effectors.groupby('TRACK_ID'):
			
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
		
		# TARGETS
		self.signal_choice_targets_cb = [QComboBox() for i in range(self.n_signals)]
		self.signal_choice_targets_label = [QLabel(f'signal {i+1}: ') for i in range(self.n_signals)]
		#self.log_btns = [QPushButton() for i in range(self.n_signals)]

		signals = list(self.df_targets.columns)
		to_remove = ['TRACK_ID', 'FRAME','x_anim','y_anim','t', 'state', 'generation', 'root', 'parent', 'class_id', 'class', 't0', 'POSITION_X', 'POSITION_Y', 'position', 'well', 'well_index', 'well_name', 'pos_name', 'index']

		for c in to_remove:
			if c in signals:
				signals.remove(c)

		for i in range(len(self.signal_choice_targets_cb)):
			self.signal_choice_targets_cb[i].addItems(['--']+signals)
			self.signal_choice_targets_cb[i].setCurrentIndex(i+1)
			self.signal_choice_targets_cb[i].currentIndexChanged.connect(self.plot_signals)

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
		
		try:
			yvalues = []
			for i in range(len(self.signal_choice_targets_cb)):
				
				signal_choice = self.signal_choice_targets_cb[i].currentText()
				self.lines[i].set_label(signal_choice)

				if signal_choice=="--":
					self.lines[i].set_xdata([])
					self.lines[i].set_ydata([])
				else:
					print(f'plot signal {signal_choice} for cell {self.target_track_of_interest}')
					xdata = self.df_targets.loc[self.df_targets['TRACK_ID']==self.target_track_of_interest, 'FRAME'].to_numpy()
					ydata = self.df_targets.loc[self.df_targets['TRACK_ID']==self.target_track_of_interest, signal_choice].to_numpy()

					xdata = xdata[ydata==ydata] # remove nan
					ydata = ydata[ydata==ydata]

					yvalues.extend(ydata)
					self.lines[i].set_xdata(xdata)
					self.lines[i].set_ydata(ydata)
					self.lines[i].set_color(tab10(i/3.))
			
			self.configure_ylims()

			min_val,max_val = self.cell_ax.get_ylim()
			t0 = self.df_targets.loc[self.df_targets['TRACK_ID']==self.target_track_of_interest, self.target_expected_time].to_numpy()[0]
			self.line_dt.set_xdata([t0, t0])
			self.line_dt.set_ydata([min_val,max_val])

			self.cell_ax.legend()
			self.cell_fcanvas.canvas.draw()
		except Exception as e:
			print(f"{e=}")


	def extract_scatter_from_target_trajectories(self):

		self.target_positions = []
		self.target_colors = []
		self.target_tracks = []

		for t in np.arange(self.len_movie):

			self.target_positions.append(self.df_targets.loc[self.df_targets['FRAME']==t,['x_anim', 'y_anim']].to_numpy())
			self.target_colors.append(self.df_targets.loc[self.df_targets['FRAME']==t,['class_color', 'status_color']].to_numpy())
			self.target_tracks.append(self.df_targets.loc[self.df_targets['FRAME']==t, 'TRACK_ID'].to_numpy())


	def extract_scatter_from_effector_trajectories(self):

		self.effector_positions = []
		self.effector_colors = []
		self.initial_effector_colors=[]
		self.effector_tracks = []

		for t in np.arange(self.len_movie):

			self.effector_positions.append(self.df_effectors.loc[self.df_effectors['FRAME']==t,['x_anim', 'y_anim']].to_numpy())
			self.effector_colors.append(self.df_effectors.loc[self.df_effectors['FRAME']==t,['class_color', 'status_color']].to_numpy())
			self.initial_effector_colors.append(self.df_effectors.loc[self.df_effectors['FRAME'] == t, ['class_color', 'status_color']].to_numpy())
			self.effector_tracks.append(self.df_effectors.loc[self.df_effectors['FRAME']==t, 'TRACK_ID'].to_numpy())


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
			for t in tqdm(range(len(indices)),desc='frame'):
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

		self.im = self.ax.imshow(self.stack[0], cmap='gray')
		self.target_status_scatter = self.ax.scatter(self.target_positions[0][:,0], self.target_positions[0][:,1], marker="x", c=self.target_colors[0][:,1], s=50, picker=True, pickradius=10)
		self.target_class_scatter = self.ax.scatter(self.target_positions[0][:,0], self.target_positions[0][:,1], marker='o', facecolors='none',edgecolors=self.target_colors[0][:,0], s=200)
		
		self.effector_status_scatter = self.ax.scatter(self.effector_positions[0][:,0], self.effector_positions[0][:,1], marker="x", c=self.effector_colors[0][:,1], s=50, picker=True, pickradius=10)
		self.effector_class_scatter = self.ax.scatter(self.effector_positions[0][:,0], self.effector_positions[0][:,1], marker='^', facecolors='none',edgecolors=self.effector_colors[0][:,0], s=200)
		
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

		self.lines = [self.cell_ax.plot([np.linspace(0,self.len_movie-1,self.len_movie)],[np.zeros((self.len_movie))])[0] for i in range(len(self.signal_choice_targets_cb))]
		for i in range(len(self.lines)):
			self.lines[i].set_label(f'signal {i}')

		min_val,max_val = self.cell_ax.get_ylim()
		self.line_dt, = self.cell_ax.plot([-1,-1],[min_val,max_val],c="k",linestyle="--")

		self.cell_ax.set_xlim(0,self.len_movie)
		self.cell_ax.legend()
		self.cell_fcanvas.canvas.draw()

		self.plot_signals()


	def on_scatter_pick(self, event):
		
		ind = event.ind

		label = event.artist.get_label()
		print(f'{label=}')

		if label == '_child1':
			pop = 'targets'
		elif label == '_child3':
			pop = 'effectors'
		else:
			return None


		if pop=='targets':
			if len(ind)>1:
				# More than one point in vicinity
				datax,datay = [self.target_positions[self.framedata][i,0] for i in ind],[self.target_positions[self.framedata][i,1] for i in ind]
				msx, msy = event.mouseevent.xdata, event.mouseevent.ydata
				dist = np.sqrt((np.array(datax)-msx)**2+(np.array(datay)-msy)**2)
				ind = [ind[np.argmin(dist)]]
			

			if len(ind)>0 and (len(self.target_selection)==0):
				ind = ind[0]
				self.target_selection.append(ind)
				self.correct_btn.setEnabled(True)
				self.cancel_btn.setEnabled(True)
				self.del_shortcut.setEnabled(True)
				self.no_event_shortcut.setEnabled(True)

				self.target_track_of_interest = self.target_tracks[self.framedata][ind]
				print(f'You selected track {self.target_track_of_interest}.')
				self.give_cell_information()
				self.plot_signals()

				self.target_loc_t = []
				self.target_loc_idx = []
				self.effector_loc_t = []
				self.effector_loc_idx = []
				#self.effector_previous_color = []
				for t in range(len(self.target_tracks)):
					indices = np.where(self.target_tracks[t]==self.target_track_of_interest)[0]
					if len(indices)>0:
						self.target_loc_t.append(t)
						self.target_loc_idx.append(indices[0])


				self.target_previous_color = []
				self.neighbors={}
				for t,idx in zip(self.target_loc_t,self.target_loc_idx):
					neigh=pd.read_pickle(self.neigh_trajectories_path)
					#print(neigh)
					columns_of_interest = neigh.filter(like='neighborhood_2_circle_200_px').columns
					first_column = next((col for col in columns_of_interest if col.startswith('neighborhood_2_circle_200_px')),
										None)
					effect = neigh.loc[(neigh['TRACK_ID'] == self.target_track_of_interest) &
									   (neigh['FRAME'] == t),
					first_column]
					#print(effect.iloc[0])
					#print(len(effect.iloc[0]))
					#print(neigh.columns)
					eff_dict1=[]
					indices=[]
					if effect.iloc[0]!=[]:
						for i in range(0,len(effect.iloc[0])):
							eff_dict1.append(effect.iloc[0][i])
						for dictionn in eff_dict1:
							#print(dictionn)
							indices.append(np.where(self.effector_tracks[t] == dictionn['id'])[0])
						#print(eff_dict1)
						#eff_dict=effect.iloc[0][0]
						#indices = np.where(self.effector_tracks[t] == eff_dict['id'])[0]
						#print(eff_dict1)
						#print(eff_dict)
						if len(indices) > 0:
							indices2=[]
							for i in indices:
								indices2.append(i[0])
							self.effector_loc_t.append(t)
							self.neighbors[t]=indices2
							#print(self.effector_loc_t)

					self.target_previous_color.append(self.target_colors[t][idx].copy())
					self.target_colors[t][idx] = 'lime'
				# print(self.effector_loc_t)
				# print(self.effector_loc_idx)
				self.effector_previous_color = []
				#print(neighbors)
				# for t_eff, idx_eff in zip(self.effector_loc_t, self.effector_loc_idx):
				# 	self.effector_previous_color.append(self.effector_colors[t_eff][idx_eff].copy())
				# 	self.effector_colors[t_eff][idx_eff] = 'darkorange'
				for key in self.neighbors.keys():
					for value in self.neighbors[key]:
						self.effector_previous_color.append(self.effector_colors[key][value].copy())
						self.effector_colors[key][value] = 'darkorange'


			elif len(ind)>0 and len(self.target_selection)==1:
				self.cancel_btn.click()
			else:
				pass

		elif pop=='effectors':
			if len(ind)>1:
				# More than one point in vicinity
				datax,datay = [self.effector_positions[self.framedata][i,0] for i in ind],[self.effector_positions[self.framedata][i,1] for i in ind]
				msx, msy = event.mouseevent.xdata, event.mouseevent.ydata
				dist = np.sqrt((np.array(datax)-msx)**2+(np.array(datay)-msy)**2)
				ind = [ind[np.argmin(dist)]]
			

			if len(ind)>0 and (len(self.effector_selection)==0):
				ind = ind[0]
				self.effector_selection.append(ind)
				self.correct_btn.setEnabled(True)
				self.cancel_btn.setEnabled(True)
				self.del_shortcut.setEnabled(True)
				self.no_event_shortcut.setEnabled(True)

				self.effector_track_of_interest = self.effector_tracks[self.framedata][ind]
				print(f'You selected track {self.effector_track_of_interest}.')
				self.give_cell_information()
				self.plot_signals()

				self.effector_loc_t = []
				self.effector_loc_idx = []
				for t in range(len(self.effector_tracks)):
					indices = np.where(self.effector_tracks[t]==self.effector_track_of_interest)[0]
					if len(indices)>0:
						self.effector_loc_t.append(t)
						self.effector_loc_idx.append(indices[0])


				self.effector_previous_color = []
				for t,idx in zip(self.effector_loc_t,self.effector_loc_idx):
					self.effector_previous_color.append(self.effector_colors[t][idx].copy())
					self.effector_colors[t][idx] = 'magenta'

			elif len(ind)>0 and len(self.effector_selection)==1:
				self.cancel_btn.click()
			else:
				pass
							
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
			for i in range(len(self.signal_choice_targets_cb)):
				signal = self.signal_choice_targets_cb[i].currentText()
				if signal=='--':
					continue
				else:
					maxx = np.nanpercentile(self.df_targets.loc[:,signal].to_numpy().flatten(),99)
					minn = np.nanpercentile(self.df_targets.loc[:,signal].to_numpy().flatten(),1)
					min_values.append(minn)
					max_values.append(maxx)

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

		self.target_status_scatter.set_offsets(self.target_positions[self.framedata])
		self.target_status_scatter.set_color(self.target_colors[self.framedata][:,1])

		self.target_class_scatter.set_offsets(self.target_positions[self.framedata])
		self.target_class_scatter.set_edgecolor(self.target_colors[self.framedata][:,0])

		self.effector_status_scatter.set_offsets(self.effector_positions[self.framedata])
		self.effector_status_scatter.set_color(self.effector_colors[self.framedata][:,1])

		self.effector_class_scatter.set_offsets(self.effector_positions[self.framedata])
		self.effector_class_scatter.set_edgecolor(self.effector_colors[self.framedata][:,0])


		return (self.im,self.target_status_scatter,self.target_class_scatter,self.effector_status_scatter,self.effector_class_scatter,)


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


	def give_cell_information(self):

		target_cell_selected = f"cell: {self.target_track_of_interest}\n"
		target_cell_class = f"class: {self.df_targets.loc[self.df_targets['TRACK_ID']==self.target_track_of_interest, self.target_class_name].to_numpy()[0]}\n"
		target_cell_time = f"time of interest: {self.df_targets.loc[self.df_targets['TRACK_ID']==self.target_track_of_interest, self.target_time_name].to_numpy()[0]}\n"

		effector_cell_selected = f"cell: {self.effector_track_of_interest}\n"
		effector_cell_class = f"class: {self.df_effectors.loc[self.df_effectors['TRACK_ID']==self.effector_track_of_interest, self.effector_class_name].to_numpy()[0]}\n"
		effector_cell_time = f"time of interest: {self.df_effectors.loc[self.df_effectors['TRACK_ID']==self.effector_track_of_interest, self.effector_time_name].to_numpy()[0]}\n"

		self.cell_info.setText(target_cell_selected+target_cell_class+target_cell_time+effector_cell_selected+effector_cell_class+effector_cell_time)


	def save_trajectories(self):

		pass

		# if self.normalized_signals:
		# 	self.normalize_features_btn.click()
		# if self.selection:
		# 	self.cancel_selection()

		# self.df_tracks = self.df_tracks.drop(self.df_tracks[self.df_tracks[self.class_name]>2].index)
		# self.df_tracks.to_csv(self.trajectories_path, index=False)
		# print('table saved.')
		# self.extract_scatter_from_trajectories()
		# #self.give_cell_information()


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

		pass
		
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

	def normalize_features(self):

		pass

		# x = self.df_tracks[self.columns_to_rescale].values

		# if not self.normalized_signals:
		# 	x = self.MinMaxScaler.transform(x)
		# 	self.df_tracks[self.columns_to_rescale] = x
		# 	self.plot_signals()
		# 	self.normalized_signals = True
		# 	self.normalize_features_btn.setIcon(icon(MDI6.arrow_collapse_vertical,color="#1565c0"))
		# 	self.normalize_features_btn.setIconSize(QSize(25, 25))
		# else:
		# 	x = self.MinMaxScaler.inverse_transform(x)
		# 	self.df_tracks[self.columns_to_rescale] = x
		# 	self.plot_signals()
		# 	self.normalized_signals = False			
		# 	self.normalize_features_btn.setIcon(icon(MDI6.arrow_collapse_vertical,color="black"))
		# 	self.normalize_features_btn.setIconSize(QSize(25, 25))

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
