from PyQt5.QtWidgets import QApplication, QFrame, QSizePolicy, QWidget, QListWidget, QVBoxLayout, QComboBox, QPushButton
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


class ListWidget(QWidget):

	def __init__(self, parent, choiceWidget, initial_features):
		
		super().__init__()

		self.parent = parent
		self.initial_features = initial_features
		self.choiceWidget = choiceWidget

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
			items.append(self.list_widget.item(x).text())
		return items

	def removeSel(self):

		"""
		Remove selected items.
		"""

		listItems=self.list_widget.selectedItems()
		if not listItems: return
		for item in listItems:
			self.list_widget.takeItem(self.list_widget.row(item))

	# def removeAll(self):

	# 	"""
	# 	Remove all items.
	# 	"""

	# 	listItems=self.list_widget.Items()
	# 	for item in listItems:
	# 		self.list_widget.takeItem(self.list_widget.row(item))


class FigureCanvas(QWidget):
	def __init__(self, fig, title="", interactive=True):
		super().__init__()
		self.fig = fig
		self.setWindowTitle(title)
		center_window(self)
		self.canvas = FigureCanvasQTAgg(self.fig)
		if interactive:
			self.toolbar = NavigationToolbar2QT(self.canvas)
		layout = QVBoxLayout(self)
		layout.addWidget(self.canvas)
		if interactive:
			layout.addWidget(self.toolbar)
