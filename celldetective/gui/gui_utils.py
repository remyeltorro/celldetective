from PyQt5.QtWidgets import QApplication, QFrame, QSizePolicy, QWidget, QLineEdit, QListWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QHBoxLayout, QCheckBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

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
			self.list_widget.takeItem(self.list_widget.row(item))


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
		layout = QVBoxLayout(self)
		layout.addWidget(self.canvas)
		if interactive:
			layout.addWidget(self.toolbar)

def color_from_status(status):
	if status==0:
		return 'tab:blue'
	elif status==1:
		return 'tab:red'
	elif status==2:
		return 'yellow'
	else:
		return 'k'

def color_from_class(cclass):
	if cclass==0:
		return 'tab:red'
	elif cclass==1:
		return 'tab:blue'
	elif cclass==2:
		return 'yellow'
	else:
		return 'k'