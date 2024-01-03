from PyQt5.QtWidgets import QApplication, QFrame, QSizePolicy, QWidget, QLineEdit, QListWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QHBoxLayout, QCheckBox
from PyQt5.QtCore import QEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.pyplot as plt

def center_window(window):
	
	"""
	Center window in the middle of the screen.
	"""
	
	frameGm = window.frameGeometry()
	screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
	centerPoint = QApplication.desktop().screenGeometry(screen).center()
	frameGm.moveCenter(centerPoint)
	window.move(frameGm.topLeft())

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

	def __init__(self, parent):

		super().__init__()
		self.parent = parent
		self.setWindowTitle("Add feature")
		# Create the QComboBox and add some items
		self.combo_box = QComboBox(self)
		center_window(self)

		self.combo_box.addItems([	"area", 
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
									])

		self.add_btn = QPushButton("Add")
		self.add_btn.clicked.connect(self.add_current_feature)

		# Create the layout
		layout = QVBoxLayout(self)
		layout.addWidget(self.combo_box)
		layout.addWidget(self.add_btn)

	def add_current_feature(self):

		filtername = self.combo_box.currentText()
		self.parent.list_widget.addItems([filtername])
		self.close()

class FilterChoice(QWidget):

	def __init__(self, parent):

		super().__init__()
		self.parent = parent
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
						  #'sauvola_filter': {'window_size': 15, 'k': 0.2}
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
		self.parent.list_widget.addItems([filtername])

		filter_instructions = [filtername.split('_')[0]]
		for a in self.arguments_le:
			arg = a.text()
			arg_num = arg
			if (arg!='') and arg_num.replace('.','').replace(',','').isnumeric():
				num = float(arg)
				if num.is_integer():
					num = int(num)
				filter_instructions.append(num)
			elif arg!='':
				filter_instructions.append(arg)
		
		print(f'You added filter {filter_instructions}.')

		self.parent.items.append(filter_instructions)
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

			if len(args)<2:
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

	def __init__(self, parent):

		super().__init__()
		self.parent = parent
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
		self.parent.list_widget.addItems([filtername])
		self.close()


class GeometryChoice(QWidget):

	def __init__(self, parent):

		super().__init__()
		self.parent = parent
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
			values = [value+'-'+value2]
		else:
			values = [value]
		self.parent.list_widget.addItems(values)
		self.close()


class DistanceChoice(QWidget):

	def __init__(self, parent):

		super().__init__()
		self.parent = parent
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
		self.parent.list_widget.addItems(values)
		self.close()


class ListWidget(QWidget):

	"""
	Generic list widget.
	"""

	def __init__(self, parent, choiceWidget, initial_features, dtype=str):
		
		super().__init__()
		self.parent = parent
		self.initial_features = initial_features
		self.choiceWidget = choiceWidget
		self.dtype = dtype
		self.items = []

		self.setFixedHeight(80)

		# Initialize list widget
		self.list_widget = QListWidget()
		self.list_widget.addItems(initial_features)

		# Set up layout
		main_layout = QVBoxLayout()
		main_layout.addWidget(self.list_widget)
		self.setLayout(main_layout)

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
			if len(self.list_widget.item(x).text().split('-'))==2:
				minn,maxx = self.list_widget.item(x).text().split('-')
				to_add = [self.dtype(minn), self.dtype(maxx)]
				items.append(to_add)
			else:
				items.append(self.dtype(self.list_widget.item(x).text()))
		return items

	def removeSel(self):

		"""
		Remove selected items.
		"""

		listItems=self.list_widget.selectedItems()
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
		center_window(self)
		self.canvas = FigureCanvasQTAgg(self.fig)
		self.canvas.setStyleSheet("background-color: transparent;")
		if interactive:
			self.toolbar = NavigationToolbar2QT(self.canvas)
		self.layout = QVBoxLayout(self)
		self.layout.addWidget(self.canvas)
		if interactive:
			self.layout.addWidget(self.toolbar)
			
	def closeEvent(self, event):
		""" Delete figure on closing window. """
		#self.canvas.ax.cla() # ****
		self.fig.clf() # ****
		plt.close(self.fig)
		super(FigureCanvas, self).closeEvent(event) 

def color_from_status(status, recently_modified=False):
	
	if not recently_modified:
		if status==0:
			return 'tab:blue'
		elif status==1:
			return 'tab:red'
		elif status==2:
			return 'yellow'
		else:
			return 'k'
	else:
		if status==0:
			return 'tab:cyan'
		elif status==1:
			return 'tab:orange'
		elif status==2:
			return 'tab:olive'
		else:
			return 'k'

def color_from_class(cclass, recently_modified=False):

	if not recently_modified:
		if cclass==0:
			return 'tab:red'
		elif cclass==1:
			return 'tab:blue'
		elif cclass==2:
			return 'yellow'
		else:
			return 'k'
	else:
		if cclass==0:
			return 'tab:orange'
		elif cclass==1:
			return 'tab:cyan'
		elif cclass==2:
			return 'tab:olive'
		else:
			return 'k'		