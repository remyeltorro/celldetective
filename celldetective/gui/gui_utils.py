import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox, QFrame, QSizePolicy, QWidget, QLineEdit, QListWidget, QVBoxLayout, QComboBox, \
	QPushButton, QLabel, QHBoxLayout, QCheckBox, QFileDialog
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QDoubleValidator, QIntValidator

from celldetective.gui import Styles
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.pyplot as plt
import celldetective.extra_properties as extra_properties
from inspect import getmembers, isfunction
from celldetective.filters import *
from os import sep

def center_window(window):
	"""
	Center window in the middle of the screen.
	"""

	frameGm = window.frameGeometry()
	screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
	centerPoint = QApplication.desktop().screenGeometry(screen).center()
	frameGm.moveCenter(centerPoint)
	window.move(frameGm.topLeft())

class ExportPlotBtn(QPushButton, Styles):
	
	def __init__(self, fig, export_dir=None):

		super().__init__()

		self.export_dir = export_dir
		self.fig = fig

		self.setText('')
		self.setIcon(icon(MDI6.content_save,color="black"))
		self.setStyleSheet(self.button_select_all)
		self.setToolTip('Export figure.')
		self.setIconSize(QSize(20, 20))
		self.clicked.connect(self.save_plot)
	
	def save_plot(self):
		if self.export_dir is not None:
			guess_dir = self.export_dir+sep+'plot.png'
		else:
			guess_dir = 'plot.png'
		fileName, _ = QFileDialog.getSaveFileName(self, 
			"Save Image", guess_dir, "Images (*.png *.xpm *.jpg *.svg)") #, options=options
		if fileName:
			self.fig.tight_layout()
			self.fig.savefig(fileName, bbox_inches='tight', dpi=300)


class QHSeperationLine(QFrame):
	'''
	a horizontal seperation line\n
	'''

	def __init__(self):
		super().__init__()
		self.setMinimumWidth(1)
		self.setFixedHeight(20)
		self.setFrameShape(QFrame.HLine)
		self.setFrameShadow(QFrame.Sunken)
		self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)


class FeatureChoice(QWidget):

	def __init__(self, parent_window):
		super().__init__()
		self.parent_window = parent_window
		self.setWindowTitle("Add feature")
		# Create the QComboBox and add some items
		self.combo_box = QComboBox(self)
		center_window(self)

		standard_measurements = ["area",
								 "area_bbox",
								 "area_convex",
								 "area_filled",
								 "major_axis_length",
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
								 "intensity_mean",
								 "intensity_max",
								 "intensity_min",
								 ]

		extra_props = getmembers(extra_properties, isfunction)
		extra_props = [extra_props[i][0] for i in range(len(extra_props))]
		if len(extra_props) > 0:
			standard_measurements.extend(extra_props)

		self.combo_box.addItems(standard_measurements)

		self.add_btn = QPushButton("Add")
		self.add_btn.clicked.connect(self.add_current_feature)

		# Create the layout
		layout = QVBoxLayout(self)
		layout.addWidget(self.combo_box)
		layout.addWidget(self.add_btn)

	def add_current_feature(self):
		filtername = self.combo_box.currentText()
		self.parent_window.list_widget.addItems([filtername])
		self.close()


class FilterChoice(QWidget):

	def __init__(self, parent_window):

		super().__init__()
		self.parent_window = parent_window
		self.setWindowTitle("Add filter")
		# Create the QComboBox and add some items
		center_window(self)

		self.default_params = {
			'gauss_filter': {'sigma': 2},
			'median_filter': {'size': 4},
			'maximum_filter': {'size': 4},
			'minimum_filter': {'size': 4},
			'percentile_filter': {'percentile': 99, 'size': 4},
			'variance_filter': {'size': 4},
			'std_filter': {'size': 4},
			'laplace_filter': None,
			'abs_filter': None,
			'ln_filter': None,
			'subtract_filter': {'value': 1},
			'dog_filter': {'sigma_low': 0.8, 'sigma_high': 1.6},
			'log_filter': {'sigma': 2},
			'tophat_filter': {'size': 4, 'connectivity': 4},
			'otsu_filter': None,
			'local_filter': {'block_size': 73, 'method': 'mean', 'offset': 0},
			'niblack_filter': {'window_size': 15, 'k': 0.2},
			# 'sauvola_filter': {'window_size': 15, 'k': 0.2}
		}

		layout = QVBoxLayout(self)
		self.combo_box = QComboBox(self)
		self.combo_box.addItems(list(self.default_params.keys()))
		self.combo_box.currentTextChanged.connect(self.update_arguments)
		layout.addWidget(self.combo_box)

		self.arguments_le = [QLineEdit() for i in range(3)]
		self.arguments_labels = [QLabel('') for i in range(3)]
		for i in range(2):
			hbox = QHBoxLayout()
			hbox.addWidget(self.arguments_labels[i], 20)
			hbox.addWidget(self.arguments_le[i], 80)
			layout.addLayout(hbox)

		self.add_btn = QPushButton("Add")
		self.add_btn.clicked.connect(self.add_current_feature)
		layout.addWidget(self.add_btn)

		self.combo_box.setCurrentIndex(0)
		self.update_arguments()

	def add_current_feature(self):

		filtername = self.combo_box.currentText()
		self.parent_window.list_widget.addItems([filtername])

		filter_instructions = [filtername.split('_')[0]]
		for a in self.arguments_le:
			arg = a.text()
			arg_num = arg
			if (arg != '') and arg_num.replace('.', '').replace(',', '').isnumeric():
				num = float(arg)
				if num.is_integer():
					num = int(num)
				filter_instructions.append(num)
			elif arg != '':
				filter_instructions.append(arg)

		print(f'You added filter {filter_instructions}.')

		self.parent_window.items.append(filter_instructions)
		self.close()

	def update_arguments(self):

		selected_filter = self.combo_box.currentText()
		arguments = self.default_params[selected_filter]
		if arguments is not None:
			args = list(arguments.keys())
			for i in range(len(args)):
				self.arguments_labels[i].setEnabled(True)
				self.arguments_le[i].setEnabled(True)

				self.arguments_labels[i].setText(args[i])
				self.arguments_le[i].setText(str(arguments[args[i]]))

			if len(args) < 2:
				for i in range(len(args), 2):
					self.arguments_labels[i].setEnabled(False)
					self.arguments_labels[i].setText('')
					self.arguments_le[i].setEnabled(False)
		else:
			for i in range(2):
				self.arguments_labels[i].setEnabled(False)
				self.arguments_le[i].setEnabled(False)
				self.arguments_labels[i].setText('')


class OperationChoice(QWidget):
	"""
	Mini window to select an operation from numpy to apply on the ROI.

	"""

	def __init__(self, parent_window):
		super().__init__()
		self.parent_window = parent_window
		self.setWindowTitle("Add feature")
		# Create the QComboBox and add some items
		self.combo_box = QComboBox(self)
		center_window(self)

		self.combo_box.addItems(["mean", "median", "average", "std", "var",
								 "nanmedian", "nanmean", "nanstd", "nanvar"])

		self.add_btn = QPushButton("Add")
		self.add_btn.clicked.connect(self.add_current_feature)

		# Create the layout
		layout = QVBoxLayout(self)
		layout.addWidget(self.combo_box)
		layout.addWidget(self.add_btn)

	def add_current_feature(self):
		filtername = self.combo_box.currentText()
		self.parent_window.list_widget.addItems([filtername])
		self.close()


class GeometryChoice(QWidget):

	def __init__(self, parent_window):

		super().__init__()
		self.parent_window = parent_window
		self.setWindowTitle("Set distances")
		center_window(self)

		# Create the QComboBox and add some items

		self.dist_label = QLabel('Distance [px]: ')
		self.dist_le = QLineEdit('10')

		self.dist_outer_label = QLabel('Max distance [px]')
		self.dist_outer_le = QLineEdit('100')
		self.outer_to_hide = [self.dist_outer_le, self.dist_outer_label]

		self.outer_btn = QCheckBox('outer distance')
		self.outer_btn.clicked.connect(self.activate_outer_value)

		self.add_btn = QPushButton("Add")
		self.add_btn.clicked.connect(self.add_current_feature)

		# Create the layout
		layout = QVBoxLayout(self)
		dist_layout = QHBoxLayout()
		dist_layout.addWidget(self.dist_label, 30)
		dist_layout.addWidget(self.dist_le, 70)

		self.dist_outer_layout = QHBoxLayout()
		self.dist_outer_layout.addWidget(self.dist_outer_label, 30)
		self.dist_outer_layout.addWidget(self.dist_outer_le, 70)

		layout.addLayout(dist_layout)
		layout.addLayout(self.dist_outer_layout)
		layout.addWidget(self.outer_btn)
		layout.addWidget(self.add_btn)

		for el in self.outer_to_hide:
			el.hide()

	def activate_outer_value(self):
		if self.outer_btn.isChecked():
			self.dist_label.setText('Min distance [px]: ')
			for el in self.outer_to_hide:
				el.show()
		else:
			self.dist_label.setText('Distance [px]: ')
			for el in self.outer_to_hide:
				el.hide()

	def add_current_feature(self):

		value = self.dist_le.text()
		if self.outer_btn.isChecked():
			value2 = self.dist_outer_le.text()
			values = [value + '-' + value2]
		else:
			values = [value]
		self.parent_window.list_widget.addItems(values)
		self.close()


class DistanceChoice(QWidget):

	def __init__(self, parent_window):
		super().__init__()
		self.parent_window = parent_window
		self.setWindowTitle("Set distances")
		center_window(self)

		# Create the QComboBox and add some items

		self.dist_label = QLabel('Distance [px]: ')
		self.dist_le = QLineEdit('10')

		self.add_btn = QPushButton("Add")
		self.add_btn.clicked.connect(self.add_current_feature)

		# Create the layout
		layout = QVBoxLayout(self)
		dist_layout = QHBoxLayout()
		dist_layout.addWidget(self.dist_label, 30)
		dist_layout.addWidget(self.dist_le, 70)

		layout.addLayout(dist_layout)
		layout.addWidget(self.add_btn)

	def add_current_feature(self):
		value = self.dist_le.text()
		values = [value]
		self.parent_window.list_widget.addItems(values)
		self.close()


class ListWidget(QWidget):
	"""
	Generic list widget.
	"""

	def __init__(self, parent_window, choiceWidget, initial_features, dtype=str, channel_names=None):

		super().__init__()
		self.parent_window = parent_window
		self.initial_features = initial_features
		self.choiceWidget = choiceWidget
		self.dtype = dtype
		self.items = []
		self.channel_names=channel_names

		self.setFixedHeight(80)

		# Initialize list widget
		self.list_widget = QListWidget()
		self.list_widget.addItems(initial_features)

		# Set up layout
		main_layout = QVBoxLayout()
		main_layout.addWidget(self.list_widget)
		self.setLayout(main_layout)
		center_window(self)

	def addItem(self):

		"""
		Add a new item.
		"""

		self.addItemWindow = self.choiceWidget(self)
		self.addItemWindow.show()

	def getItems(self):

		"""
		Get all the items as a list.
		"""

		items = []
		for x in range(self.list_widget.count()):
			if len(self.list_widget.item(x).text().split('-')) == 2:
				if self.list_widget.item(x).text()[0] == '-':
					items.append(self.dtype(self.list_widget.item(x).text()))
				else:
					minn, maxx = self.list_widget.item(x).text().split('-')
					to_add = [self.dtype(minn), self.dtype(maxx)]
					items.append(to_add)
			else:
				items.append(self.dtype(self.list_widget.item(x).text()))
		return items



	def removeSel(self):

		"""
		Remove selected items.
		"""

		listItems = self.list_widget.selectedItems()
		if not listItems: return
		for item in listItems:
			idx = self.list_widget.row(item)
			self.list_widget.takeItem(idx)
			if self.items:
				del self.items[idx]


class FigureCanvas(QWidget):
	"""
	Generic figure canvas.
	"""

	def __init__(self, fig, title="", interactive=True):
		super().__init__()
		self.fig = fig
		self.setWindowTitle(title)
		self.canvas = FigureCanvasQTAgg(self.fig)
		self.canvas.setStyleSheet("background-color: transparent;")
		if interactive:
			self.toolbar = NavigationToolbar2QT(self.canvas)
		self.layout = QVBoxLayout(self)
		self.layout.addWidget(self.canvas)
		if interactive:
			self.layout.addWidget(self.toolbar)

		center_window(self)
		self.setAttribute(Qt.WA_DeleteOnClose)

	def draw(self):
		self.canvas.draw()

	def closeEvent(self, event):
		""" Delete figure on closing window. """
		# self.canvas.ax.cla() # ****
		self.fig.clf()  # ****
		plt.close(self.fig)
		super(FigureCanvas, self).closeEvent(event)


class QuickSliderLayout(QHBoxLayout):
	
	"""docstring for ClassName"""
	
	def __init__(self, label=None, slider=None, layout_ratio=(0.25,0.75), slider_initial_value=1, slider_range=(0,1), slider_tooltip=None, decimal_option=True, precision=1.0E-03, *args):
		super().__init__(*args)

		if label is not None and isinstance(label,str):
			self.qlabel = QLabel(label)
			self.addWidget(self.qlabel, int(100*layout_ratio[0]))

		self.slider = slider
		self.slider.setOrientation(1)
		if decimal_option:
			self.slider.setSingleStep(precision)
			self.slider.setTickInterval(precision)
		else:
			self.slider.setSingleStep(1)
			self.slider.setTickInterval(1)

		self.slider.setRange(*slider_range)
		self.slider.setValue(slider_initial_value)
		if isinstance(slider_tooltip,str):
			self.slider.setToolTip(slider_tooltip)

		self.addWidget(self.slider, int(100*layout_ratio[1]))

class ThresholdLineEdit(QLineEdit):
	
	"""docstring for ClassName"""
	
	def __init__(self, init_value=2.0, connected_buttons=None, placeholder='px > thresh are masked',value_type='float',*args):
		super().__init__(*args)

		self.init_value = init_value
		self.value_type = value_type 
		self.connected_buttons = connected_buttons
		self.setPlaceholderText(placeholder)

		if self.value_type=="float":
			self.setValidator(QDoubleValidator())
		else:
			self.init_value = int(self.init_value)
			self.setValidator(QIntValidator())

		if self.connected_buttons is not None:
			self.textChanged.connect(self.enable_btn)
		self.set_threshold(self.init_value)

	def enable_btn(self):

		thresh = self.get_threshold(show_warning=False)
		if isinstance(self.connected_buttons, QPushButton):
			cbs = [self.connected_buttons]
		else:
			cbs = self.connected_buttons

		if thresh is None:
			for c in cbs:
				c.setEnabled(False)
		else:
			for c in cbs:
				c.setEnabled(True)

	def set_threshold(self, value):

		try:
			self.setText(str(value).replace('.',','))
		except:
			print('Please provide a valid threshold value...')

	def get_threshold(self, show_warning=True):
		
		try:
			if self.value_type=='float':
				thresh = float(self.text().replace(',','.'))
			else:
				thresh = int(self.text().replace(',','.'))
		except ValueError:
			if show_warning:
				msgBox = QMessageBox()
				msgBox.setWindowTitle('warning')
				msgBox.setIcon(QMessageBox.Critical)
				msgBox.setText("Please set a valid threshold value.")
				msgBox.setWindowTitle("")
				msgBox.setStandardButtons(QMessageBox.Ok)
				returnValue = msgBox.exec()
			thresh = None

		return thresh

# class BackgroundFitCorrectionLayout(QGridLayout):
	
# 	"""docstring for ClassName"""
	
# 	def __init__(self, parent=None, *args):
# 		super().__init__(*args)

# 		self.parent = parent
# 		self.channel_names = self.parent.channel_names # check this

# 		self.setContentsMargins(15,15,15,15)
# 		self.generate_widgets()
# 		self.add_to_layout()

# 	def generate_widgets(self):

# 		self.channel_lbl = QLabel('Channel: ')
# 		self.channels_cb = QComboBox()
# 		self.channels_cb.addItems(self.channel_names)
		
# 		self.thresh_lbl = QLabel('Threshold: ')
# 		self.thresh_lbl.setToolTip('Threshold on the STD-filtered image.\nPixel values above the threshold are\nconsidered as non-background and are\nmasked prior to background estimation.')
# 		self.threshold_viewer_btn = QPushButton()
# 		self.threshold_viewer_btn.setIcon(icon(MDI6.image_check, color="k"))
# 		self.threshold_viewer_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
# 		self.threshold_viewer_btn.clicked.connect(self.set_threshold_graphically)

# 		self.model_lbl = QLabel('Model: ')
# 		self.models_cb = QComboBox()
# 		self.models_cb.addItems(['Paraboloid', 'Place'])
		
# 		self.operation_lbl = QLabel('Operation: ')
# 		self.operation_group = QButtonGroup()
# 		self.subtract_btn = QRadioButton('Subtract')
# 		self.divide_btn = QRadioButton('Divide')
# 		self.subtract_btn.toggled.connect(self.activate_clipping_options)
# 		self.divide_btn.toggled.connect(self.activate_clipping_options)

# 		self.operation_group.addButton(self.subtract_btn)
# 		self.operation_group.addButton(self.divide_btn)

# 		self.clip_group = QButtonGroup()
# 		self.clip_btn = QRadioButton('Clip')
# 		self.clip_not_btn = QRadioButton('Do not clip')

# 		self.clip_group.addButton(self.clip_btn)
# 		self.clip_group.addButton(self.clip_not_btn)

# 		self.corrected_stack_viewer = QPushButton("")
# 		self.corrected_stack_viewer.setStyleSheet(self.parent.parent.parent.button_select_all)
# 		self.corrected_stack_viewer.setIcon(icon(MDI6.eye_outline, color="black"))
# 		self.corrected_stack_viewer.setToolTip("View corrected image")
# 		self.corrected_stack_viewer.setIconSize(QSize(20, 20))

# 		self.add_correction_btn = QPushButton('Add correction')
# 		self.add_correction_btn.setStyleSheet(self.parent.parent.parent.button_style_sheet_2)
# 		self.add_correction_btn.setIcon(icon(MDI6.plus, color="#1565c0"))
# 		self.add_correction_btn.setToolTip('Add correction.')
# 		self.add_correction_btn.setIconSize(QSize(25, 25))
# 		self.add_correction_btn.clicked.connect(self.add_instructions_to_parent_list)

# 		self.threshold_le = ThresholdLineEdit(init_value=2, connected_buttons=[self.threshold_viewer_btn,
# 																			   self.corrected_stack_viewer,
# 																			   self.add_correction_btn
# 																			   ])

# 	def add_to_layout(self):
		
# 		channel_layout = QHBoxLayout()
# 		channel_layout.addWidget(self.channel_lbl, 25)
# 		channel_layout.addWidget(self.channels_cb, 75)
# 		self.addLayout(channel_layout, 0, 0, 1, 3)

# 		threshold_layout = QHBoxLayout()
# 		threshold_layout.addWidget(self.thresh_lbl, 25)
# 		threshold_layout.addWidget(self.threshold_le, 70)
# 		threshold_layout.addWidget(self.threshold_viewer_btn, 5)
# 		self.addLayout(threshold_layout, 1, 0, 1, 3)

# 		model_layout = QHBoxLayout()
# 		model_layout.addWidget(self.model_lbl, 25)
# 		model_layout.addWidget(self.models_cb, 75)
# 		self.addLayout(model_layout, 2, 0, 1, 3)

# 		operation_layout = QHBoxLayout()
# 		operation_layout.addWidget(self.operation_lbl, 25)
# 		operation_layout.addWidget(self.subtract_btn, 75//2, alignment=Qt.AlignCenter)
# 		operation_layout.addWidget(self.divide_btn, 75//2, alignment=Qt.AlignCenter)
# 		self.addLayout(operation_layout, 3, 0, 1, 3)

# 		clip_layout = QHBoxLayout()
# 		clip_layout.addWidget(QLabel(''), 25)
# 		clip_layout.addWidget(self.clip_btn, 75//4, alignment=Qt.AlignCenter)
# 		clip_layout.addWidget(self.clip_not_btn, 75//4, alignment=Qt.AlignCenter)
# 		clip_layout.addWidget(QLabel(''), 75//2)
# 		self.addLayout(clip_layout, 4, 0, 1, 3)

# 		self.addWidget(self.corrected_stack_viewer, 4, 2, 1, 1)
# 		self.addWidget(self.add_correction_btn, 5, 0, 1, 3)

# 		self.subtract_btn.click()
# 		self.clip_not_btn.click()

# 		verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
# 		self.addItem(verticalSpacer, 5, 0, 1, 3)

# 	def add_instructions_to_parent_list(self):

# 		self.generate_instructions()
# 		self.parent.background_correction.append(self.instructions)
# 		correction_description = ""
# 		for index, (key, value) in enumerate(self.instructions.items()):
# 			if index > 0:
# 				correction_description += ", "
# 			correction_description += str(key) + " : " + str(value)
# 		self.parent.normalisation_list.addItem(correction_description)

# 	def generate_instructions(self):

# 		if self.subtract_btn.isChecked():
# 			operation = "subtract"
# 		else:
# 			operation = "divide"
# 			clip = None

# 		if self.clip_btn.isChecked() and self.subtract_btn.isChecked():
# 			clip = True
# 		else:
# 			clip = False

# 		self.instructions = {
# 					  "target_channel": self.channels_cb.currentText(),
# 					  "correction_type": "fit",
# 					  "threshold_on_std": self.threshold_le.get_threshold(),
# 					  "operation": operation,
# 					  "clip": clip
# 					 }

# 	def activate_clipping_options(self):
		
# 		if self.subtract_btn.isChecked():
# 			self.clip_btn.setEnabled(True)
# 			self.clip_not_btn.setEnabled(True)

# 		else:
# 			self.clip_btn.setEnabled(False)
# 			self.clip_not_btn.setEnabled(False)		

# 	def set_target_channel(self):

# 		channel_indices = _extract_channel_indices_from_config(self.parent.parent.exp_config, [self.channels_cb.currentText()])
# 		self.target_channel = channel_indices[0]

# 	def set_threshold_graphically(self):

# 		self.parent.locate_image()
# 		self.set_target_channel()
# 		thresh = self.threshold_le.get_threshold()

# 		if self.parent.current_stack is not None and thresh is not None:
# 			self.viewer = ThresholdedStackVisualizer(initial_threshold=thresh,
# 													 parent_le = self.threshold_le,
# 													 preprocessing=[['gauss',2],["std",4]],
# 													 stack_path=self.parent.current_stack,
# 													 n_channels=len(self.channel_names),
# 													 target_channel=self.target_channel,
# 													 window_title='Set the exclusion threshold',
# 													 )
# 			self.viewer.show()


def color_from_status(status, recently_modified=False):
	if not recently_modified:
		if status == 0:
			return 'tab:blue'
		elif status == 1:
			return 'tab:red'
		elif status == 2:
			return 'yellow'
		else:
			return 'k'
	else:
		if status == 0:
			return 'tab:cyan'
		elif status == 1:
			return 'tab:orange'
		elif status == 2:
			return 'tab:olive'
		else:
			return 'k'

def color_from_state(state, recently_modified=False):
	unique_values = np.unique(state)
	color_map={}
	for value in unique_values:
		color_map[value] = plt.cm.tab10(value)
		if value == 99:
			color_map[value] = 'k'
	# colors = plt.cm.tab10(len(unique_values))
	# color_map = dict(zip(unique_values, colors))
	# print(color_map)
	return color_map




def color_from_class(cclass, recently_modified=False):
	if not recently_modified:
		if cclass == 0:
			return 'tab:red'
		elif cclass == 1:
			return 'tab:blue'
		elif cclass == 2:
			return 'yellow'
		else:
			return 'k'
	else:
		if cclass == 0:
			return 'tab:orange'
		elif cclass == 1:
			return 'tab:cyan'
		elif cclass == 2:
			return 'tab:olive'
		else:
			return 'k'


class ChannelChoice(QWidget):

	def __init__(self, parent_window):
		super().__init__()
		self.parent_window = parent_window
		#self.channel_names = channel_names
		self.setWindowTitle("Choose target channel")
		# Create the QComboBox and add some items
		self.combo_box = QComboBox(self)
		center_window(self)

		channels = parent_window.channel_names

		self.combo_box.addItems(channels)

		self.add_btn = QPushButton("Add")
		self.add_btn.clicked.connect(self.add_current_channel)

		# Create the layout
		layout = QVBoxLayout(self)
		layout.addWidget(self.combo_box)
		layout.addWidget(self.add_btn)

	def add_current_channel(self):
		filtername = self.combo_box.currentText()
		self.parent_window.list_widget.addItems([filtername])
		self.close()

def help_generic(tree):

	output = generic_msg(list(tree.keys())[0])
	while output is not None:
		tree = tree[list(tree.keys())[0]][output]
		if isinstance(tree,dict):
			output = generic_msg(list(tree.keys())[0])
		else:
			# return the final suggestion
			output = None
	return tree

def generic_msg(text):

	msgBox = QMessageBox()
	msgBox.setIcon(QMessageBox.Question)
	msgBox.setText(text)
	msgBox.setWindowTitle("Question")
	msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
	returnValue = msgBox.exec()
	if returnValue == QMessageBox.Yes:
		return "yes"
	elif returnValue == QMessageBox.No:
		return "no"
	else:
		return None