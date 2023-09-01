from PyQt5.QtWidgets import QMainWindow, QTableView, QAction, QMenu, QLineEdit, QHBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import Qt, QAbstractTableModel
import pandas as pd
import matplotlib.pyplot as plt
from celldetective.gui.gui_utils import FigureCanvas, center_window
import numpy as np

class PandasModel(QAbstractTableModel):

	def __init__(self, data):
		QAbstractTableModel.__init__(self)
		self._data = data

	def rowCount(self, parent=None):
		return self._data.shape[0]

	def columnCount(self, parent=None):
		return self._data.shape[1]

	def data(self, index, role=Qt.DisplayRole):
		if index.isValid():
			if role == Qt.DisplayRole:
				return str(self._data.iloc[index.row(), index.column()])
		return None

	def headerData(self, col, orientation, role):
		if orientation == Qt.Horizontal and role == Qt.DisplayRole:
			return self._data.columns[col]
		return None


class QueryWidget(QWidget):

	def __init__(self, parent):

		super().__init__()
		self.parent = parent
		self.setWindowTitle("Filter table")
		# Create the QComboBox and add some items
		center_window(self)

		
		layout = QHBoxLayout(self)
		layout.setContentsMargins(30,30,30,30)
		self.query_le = QLineEdit()
		layout.addWidget(self.query_le, 70)

		self.submit_btn = QPushButton('submit')
		self.submit_btn.clicked.connect(self.filter_table)
		layout.addWidget(self.submit_btn, 30)

	def filter_table(self):
		try:
			tab = self.parent.data.query(self.query_le.text())
			self.subtable = TableUI(tab,self.query_le.text(), plot_mode="plot_track_signals")
			self.subtable.show()
			self.close()
		except Exception as e:
			print(e)
			return None


class TableUI(QMainWindow):
	def __init__(self, data, title, plot_mode="plot_track_signals", *args, **kwargs):

		QMainWindow.__init__(self, *args, **kwargs)

		self.setWindowTitle(title)
		self.setGeometry(100,100,1000,400)
		center_window(self)
		self.title = title
		self.plot_mode = plot_mode
		self.numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

		self._createMenuBar()
		self._createActions()

		self.table_view = QTableView(self)
		self.setCentralWidget(self.table_view)

		# Set the model for the table view
		self.data = data

		self.model = PandasModel(data)
		self.table_view.setModel(self.model)

	def _createActions(self):

			self.save_as = QAction("&Save as...", self)
			#self.save_as.triggered.connect(self.save_as_csv)
			self.save_as.setShortcut("Ctrl+s")
			self.fileMenu.addAction(self.save_as)

			self.plot_action = QAction("&Plot...", self)
			self.plot_action.triggered.connect(self.plot)
			self.plot_action.setShortcut("Ctrl+p")
			self.fileMenu.addAction(self.plot_action)		

			self.groupby_action = QAction("&Group by tracks...", self)
			#self.groupby_action.triggered.connect(self.groupby_track_table)
			self.groupby_action.setShortcut("Ctrl+g")
			self.fileMenu.addAction(self.groupby_action)

			self.groupby_time_action = QAction("&Group by frames...", self)
			self.groupby_time_action.triggered.connect(self.groupby_time_table)
			self.groupby_time_action.setShortcut("Ctrl+t")
			self.fileMenu.addAction(self.groupby_time_action)

			self.query_action = QAction('Query...', self)
			self.query_action.triggered.connect(self.perform_query)
			self.fileMenu.addAction(self.query_action)

	def groupby_time_table(self):

		"""
		
		Perform a time average across each track for all features

		"""

		num_df = self.data.select_dtypes(include=self.numerics)

		timeseries = num_df.groupby("FRAME").mean().copy()
		timeseries["timeline"] = timeseries.index
		self.subtable = TableUI(timeseries,"Group by frames", plot_mode="plot_timeseries")
		self.subtable.show()

	def perform_query(self):

		"""
		
		Perform a time average across each track for all features

		"""
		self.query_widget = QueryWidget(self)
		self.query_widget.show()

		# num_df = self.data.select_dtypes(include=self.numerics)

		# timeseries = num_df.groupby("FRAME").mean().copy()
		# timeseries["timeline"] = timeseries.index
		# self.subtable = TableUI(timeseries,"Group by frames", plot_mode="plot_timeseries")
		# self.subtable.show()


	# def groupby_track_table(self):

	# 	"""
		
	# 	Perform a time average across each track for all features

	# 	"""

	# 	self.subtable = TrajectoryTablePanel(self.data.groupby("TRACK_ID").mean(),"Group by tracks", plot_mode="scatter")
	# 	self.subtable.show()

	def _createMenuBar(self):
		menuBar = self.menuBar()
		self.fileMenu = QMenu("&File", self)
		menuBar.addMenu(self.fileMenu)

	# def save_as_csv(self):
	# 	options = QFileDialog.Options()
	# 	options |= QFileDialog.ReadOnly
	# 	file_name, _ = QFileDialog.getSaveFileName(self, "Save as .csv", "","CSV Files (*.csv);;All Files (*)", options=options)
	# 	if file_name:
	# 		if not file_name.endswith(".csv"):
	# 			file_name += ".csv"
	# 		self.data.to_csv(file_name, index=False)

	# def test_bool(self, array):
	# 	if array.dtype=="bool":
	# 		return np.array(array, dtype=int)
	# 	else:
	# 		return array

	def plot(self):

		if self.plot_mode=="scatter":
	
			x = self.table_view.selectedIndexes()
			col_idx = [l.column() for l in x]
			row_idx = [l.row() for l in x]
			column_names = self.data.columns
			unique_cols = np.unique(col_idx)


			if len(unique_cols)==1:
				print("one column, histogram mode")
				x1 = self.test_bool(self.data.iloc[row_idx, unique_cols[0]])
				fig,ax = plt.subplots(1,1,figsize=(7,5.5))
				ax.hist(x1)
				ax.set_xlabel(column_names[unique_cols[0]])
				plt.tight_layout()
				plt.show(block=False)

			elif len(unique_cols)==2:

				print("two columns, plot mode")
				x1 = self.test_bool(self.data.iloc[row_idx, unique_cols[0]])
				x2 = self.test_bool(self.data.iloc[row_idx, unique_cols[1]])

				self.fig, self.ax = plt.subplots(1,1,figsize=(4,3))
				self.scatter_wdw = FigureCanvas(self.fig, title="scatter")
				self.ax.clear()
				self.ax.scatter(x1,x2)
				self.ax.set_xlabel(column_names[unique_cols[0]])
				self.ax.set_ylabel(column_names[unique_cols[1]])
				plt.tight_layout()
				self.fig.set_facecolor('none')  # or 'None'
				self.fig.canvas.setStyleSheet("background-color: transparent;")
				self.scatter_wdw.canvas.draw()
				self.scatter_wdw.show(block=False)

			else:
				print("please select less columns")

		elif self.plot_mode=="plot_timeseries":
			print("mode plot frames")
			x = self.table_view.selectedIndexes()
			col_idx = np.array([l.column() for l in x])
			row_idx = np.array([l.row() for l in x])
			column_names = self.data.columns
			unique_cols = np.unique(col_idx)

			fig,ax = plt.subplots(1,1,figsize=(7,5.5))
			for k in range(len(unique_cols)):

				row_idx_i = row_idx[np.where(col_idx==unique_cols[k])[0]]
				y = self.data.iloc[row_idx_i, unique_cols[k]]
				ax.plot(self.data["timeline"][row_idx_i], y, label=column_names[unique_cols[k]])

			ax.legend()
			ax.set_xlabel("time [frame]")
			ax.set_ylabel(self.title)
			plt.tight_layout()
			plt.show(block=False)

		elif self.plot_mode=="plot_track_signals":

			print("mode plot track signals")

			x = self.table_view.selectedIndexes()
			col_idx = np.array([l.column() for l in x])
			row_idx = np.array([l.row() for l in x])
			column_names = self.data.columns
			unique_cols = np.unique(col_idx)

			if len(unique_cols)>2:
				fig,ax = plt.subplots(1,1,figsize=(7,5.5))
				for k in range(len(unique_cols)):

					row_idx_i = row_idx[np.where(col_idx==unique_cols[k])[0]]
					y = self.data.iloc[row_idx_i, unique_cols[k]]
					print(unique_cols[k])
					for tid,group in self.data.groupby('TRACK_ID'):
						ax.plot(group["FRAME"], group[column_names[unique_cols[k]]],label=column_names[unique_cols[k]])
					#ax.plot(self.data["FRAME"][row_idx_i], y, label=column_names[unique_cols[k]])
				ax.legend()
				ax.set_xlabel("time [frame]")
				ax.set_ylabel(self.title)
				plt.tight_layout()
				plt.show(block=False)

			if len(unique_cols)==2:

				self.fig, self.ax = plt.subplots(1,1,figsize=(4,3))
				self.scatter_wdw = FigureCanvas(self.fig, title="scatter")
				self.ax.clear()
				for tid,group in self.data.groupby('TRACK_ID'):
					self.ax.plot(group[column_names[unique_cols[0]]], group[column_names[unique_cols[1]]], marker="o")
				self.ax.set_xlabel(column_names[unique_cols[0]])
				self.ax.set_ylabel(column_names[unique_cols[1]])
				plt.tight_layout()
				self.fig.set_facecolor('none')  # or 'None'
				self.fig.canvas.setStyleSheet("background-color: transparent;")
				self.scatter_wdw.canvas.draw()
				self.scatter_wdw.show()

			if len(unique_cols)==1:
				
				self.fig, self.ax = plt.subplots(1,1,figsize=(4,3))
				self.plot_wdw = FigureCanvas(self.fig, title="scatter")
				self.ax.clear()
				
				if 't0' in list(self.data.columns):
					ref_time_col = 't0'
				else:
					ref_time_col = 'FRAME'

				for tid,group in self.data.groupby('TRACK_ID'):
					self.ax.plot(group["FRAME"] - group[ref_time_col].to_numpy()[0], group[column_names[unique_cols[0]]],c="k", alpha = 0.1)
				self.ax.set_xlabel(r"$t - t_0$ [frame]")
				self.ax.set_ylabel(column_names[unique_cols[0]])
				plt.tight_layout()
				self.fig.set_facecolor('none')  # or 'None'
				self.fig.canvas.setStyleSheet("background-color: transparent;")
				self.plot_wdw.canvas.draw()
				self.plot_wdw.show()