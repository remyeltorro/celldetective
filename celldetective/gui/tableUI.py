from PyQt5.QtWidgets import QMainWindow, QTableView, QAction, QMenu,QFileDialog, QLineEdit, QHBoxLayout, QWidget, QPushButton, QVBoxLayout, QComboBox, QLabel, QCheckBox
from PyQt5.QtCore import Qt, QAbstractTableModel
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
plt.rcParams['svg.fonttype'] = 'none'
from celldetective.gui.gui_utils import FigureCanvas, center_window
import numpy as np
import seaborn as sns
import matplotlib.cm as mcm

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
			query_text = self.query_le.text().replace('class','`class`')
			tab = self.parent.data.query(query_text)
			self.subtable = TableUI(tab,query_text, plot_mode="scatter")
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
			self.save_as.triggered.connect(self.save_as_csv)
			self.save_as.setShortcut("Ctrl+s")
			self.fileMenu.addAction(self.save_as)

			self.plot_action = QAction("&Plot...", self)
			self.plot_action.triggered.connect(self.plot)
			self.plot_action.setShortcut("Ctrl+p")
			self.fileMenu.addAction(self.plot_action)		

			self.groupby_action = QAction("&Group by tracks...", self)
			self.groupby_action.triggered.connect(self.set_projection_mode_tracks)
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

	def set_projection_mode_tracks(self):

		self.projectionWidget = QWidget()
		self.projectionWidget.setWindowTitle('Set projection mode')
		
		layout = QVBoxLayout()
		self.projectionWidget.setLayout(layout)
		self.projection_op_cb = QComboBox()
		self.projection_op_cb.addItems(['mean','median','min','max', 'prod', 'sum', 'std', 'var', 'sem', 'skew', 'kurt', 'idmax', 'idmin'])
		hbox = QHBoxLayout()
		hbox.addWidget(QLabel('operation: '), 33)
		hbox.addWidget(self.projection_op_cb, 66)
		layout.addLayout(hbox)

		self.set_projection_btn = QPushButton('set')
		self.set_projection_btn.clicked.connect(self.set_proj_mode)
		layout.addWidget(self.set_projection_btn)

		self.projectionWidget.show()
		center_window(self.projectionWidget)

	def set_1D_plot_params(self):

		self.plot1Dparams = QWidget()
		self.plot1Dparams.setWindowTitle('Set 1D plot parameters')
		
		layout = QVBoxLayout()
		self.plot1Dparams.setLayout(layout)

		layout.addWidget(QLabel('Representations: '))
		self.hist_check = QCheckBox('histogram')
		self.kde_check = QCheckBox('KDE plot')
		self.ecdf_check = QCheckBox('ECDF plot')
		self.swarm_check = QCheckBox('swarm')
		self.violin_check = QCheckBox('violin')
		self.strip_check = QCheckBox('strip')
		self.box_check = QCheckBox('Boxplot')

		layout.addWidget(self.hist_check)
		layout.addWidget(self.kde_check)
		layout.addWidget(self.ecdf_check)
		layout.addWidget(self.swarm_check)
		layout.addWidget(self.violin_check)
		layout.addWidget(self.strip_check)
		layout.addWidget(self.box_check)

		self.hue_cb = QComboBox()
		self.hue_cb.addItems(list(self.data.columns))
		idx = self.hue_cb.findText('well_index')
		self.hue_cb.setCurrentIndex(idx)
		hbox = QHBoxLayout()
		hbox.addWidget(QLabel('hue: '), 33)
		hbox.addWidget(self.hue_cb, 66)
		layout.addLayout(hbox)


		self.cmap_cb = QComboBox()
		self.cmap_cb.addItems(list(plt.colormaps()))
		hbox = QHBoxLayout()
		hbox.addWidget(QLabel('colormap: '), 33)
		hbox.addWidget(self.cmap_cb, 66)
		layout.addLayout(hbox)

		self.plot1d_btn = QPushButton('set')
		self.plot1d_btn.clicked.connect(self.plot1d)
		layout.addWidget(self.plot1d_btn)

		self.plot1Dparams.show()
		center_window(self.plot1Dparams)


	def plot1d(self):

		x = self.table_view.selectedIndexes()
		col_idx = np.array([l.column() for l in x])
		row_idx = np.array([l.row() for l in x])
		column_names = self.data.columns
		unique_cols = np.unique(col_idx)[0]

		self.fig, self.ax = plt.subplots(1,1,figsize=(4,3))
		self.plot1dWindow = FigureCanvas(self.fig, title="scatter")
		self.ax.clear()
		row_idx_i = row_idx[np.where(col_idx==unique_cols)[0]]
		y = self.data.iloc[row_idx_i, unique_cols]
		
		cmap = getattr(mcm, self.cmap_cb.currentText())
		hue_variable = self.hue_cb.currentText()

		colors = [cmap(i / len(self.data[hue_variable].unique())) for i in range(len(self.data[hue_variable].unique()))]
		#for w,well_group in self.data.groupby('well_index'):

		legend=True
		if self.hist_check.isChecked():
			sns.histplot(data=self.data, x=column_names[unique_cols], hue=hue_variable, legend=legend, ax=self.ax, palette=colors, kde=True)
			legend = False
		if self.kde_check.isChecked():
			sns.kdeplot(data=self.data, x=column_names[unique_cols], hue=hue_variable, legend=legend, ax=self.ax, palette=colors)
			legend = False

		if self.ecdf_check.isChecked():
			sns.ecdfplot(data=self.data, x=column_names[unique_cols], hue=hue_variable, legend=legend, ax=self.ax, palette=colors)
			legend = False

		if self.swarm_check.isChecked():
			sns.swarmplot(data=self.data, y=column_names[unique_cols],dodge=True, hue=hue_variable,legend=legend, ax=self.ax, palette=colors)
			legend = False

		if self.violin_check.isChecked():
			sns.violinplot(data=self.data, y=column_names[unique_cols],dodge=True, hue=hue_variable,legend=legend, ax=self.ax, palette=colors)
			legend = False

		if self.box_check.isChecked():
			sns.boxplot(data=self.data, y=column_names[unique_cols],dodge=True, hue=hue_variable,legend=legend, ax=self.ax, fill=False,palette=colors, linewidth=2,)
			legend = False

		if self.strip_check.isChecked():
			sns.stripplot(data=self.data, y=column_names[unique_cols],dodge=True, ax=self.ax, hue=hue_variable, legend=legend, palette=colors)
			legend = False

		plt.tight_layout()
		self.fig.set_facecolor('none')  # or 'None'
		self.fig.canvas.setStyleSheet("background-color: transparent;")
		self.plot1dWindow.canvas.draw()
		self.plot1dWindow.show()


	def set_proj_mode(self):
		self.projection_mode = self.projection_op_cb.currentText()
		#eval(self.projection_mode)
		op = getattr(self.data.groupby(['position', 'TRACK_ID']), self.projection_mode)
		group_table = op(self.data.groupby(['position', 'TRACK_ID']))

		self.static_columns = ['well_index', 'well_name', 'pos_name', 'position', 'well', 'status', 't0', 'class', 'concentration', 'antibody', 'pharmaceutical_agent']
		for c in self.static_columns:
			group_table[c] = self.data.groupby(['position','TRACK_ID'])[c].apply(lambda x: x.unique()[0])

		self.subtable = TableUI(group_table,f"Group by tracks: {self.projection_mode}", plot_mode="static")
		self.subtable.show()

		self.projectionWidget.close()

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

	def save_as_csv(self):
		options = QFileDialog.Options()
		options |= QFileDialog.ReadOnly
		file_name, _ = QFileDialog.getSaveFileName(self, "Save as .csv", "","CSV Files (*.csv);;All Files (*)", options=options)
		if file_name:
			if not file_name.endswith(".csv"):
				file_name += ".csv"
			self.data.to_csv(file_name, index=False)

	# def test_bool(self, array):
	# 	if array.dtype=="bool":
	# 		return np.array(array, dtype=int)
	# 	else:
	# 		return array

	def plot(self):

		if self.plot_mode=="static":
	
			x = self.table_view.selectedIndexes()
			col_idx = [l.column() for l in x]
			row_idx = [l.row() for l in x]
			column_names = self.data.columns
			unique_cols = np.unique(col_idx)

			if len(unique_cols)==1:
				# 1D plot
				# Open widget to set 1D data representations
				self.set_1D_plot_params()



				# x = self.table_view.selectedIndexes()
				# col_idx = np.array([l.column() for l in x])
				# row_idx = np.array([l.row() for l in x])
				# column_names = self.data.columns
				# unique_cols = np.unique(col_idx)[0]

				# self.fig, self.ax = plt.subplots(1,1,figsize=(4,3))
				# self.histogram_window = FigureCanvas(self.fig, title="scatter")
				# self.ax.clear()
				# row_idx_i = row_idx[np.where(col_idx==unique_cols)[0]]
				# y = self.data.iloc[row_idx_i, unique_cols]

				# colors = [viridis(i / len(self.data['well_index'].unique())) for i in range(len(self.data['well_index'].unique()))]
				# #for w,well_group in self.data.groupby('well_index'):
				# sns.boxplot(data=self.data, y=column_names[unique_cols],dodge=True, hue='well_index',legend=False, ax=self.ax, fill=False,palette=colors, linewidth=2,)
				# sns.stripplot(data=self.data, y=column_names[unique_cols],dodge=True, ax=self.ax, hue='well_index', legend=False, palette=colors)
				# # sns.kdeplot(data=self.data, x=column_names[unique_cols], hue='well_index', ax=self.ax, fill=False,common_norm=False, palette=colors, alpha=.5, linewidth=2,)
				# # for k,(w,well_group) in enumerate(self.data.groupby('well_index')):
				# # 	self.ax.hist(well_group[column_names[unique_cols]],label=w, density=True, alpha=0.5, color=colors[k])
				# #self.ax.legend()
				# self.ax.set_xlabel(column_names[unique_cols])
				# plt.tight_layout()
				# self.fig.set_facecolor('none')  # or 'None'
				# self.fig.canvas.setStyleSheet("background-color: transparent;")
				# self.histogram_window.canvas.draw()
				# self.histogram_window.show()


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
			print('we plot here')

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
					for w,well_group in self.data.groupby('well_name'):
						for pos,pos_group in well_group.groupby('pos_name'):
							for tid,group_track in pos_group.groupby('TRACK_ID'):
								ax.plot(group_track["FRAME"], group_track[column_names[unique_cols[k]]],label=column_names[unique_cols[k]])
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


				for w,well_group in self.data.groupby('well_name'):
					for pos,pos_group in well_group.groupby('pos_name'):
						for tid,group_track in pos_group.groupby('TRACK_ID'):
							self.ax.plot(group_track["FRAME"] - group_track[ref_time_col].to_numpy()[0], group_track[column_names[unique_cols[0]]],c="k", alpha = 0.1)
				self.ax.set_xlabel(r"$t - t_0$ [frame]")
				self.ax.set_ylabel(column_names[unique_cols[0]])
				plt.tight_layout()
				self.fig.set_facecolor('none')  # or 'None'
				self.fig.canvas.setStyleSheet("background-color: transparent;")
				self.plot_wdw.canvas.draw()
				self.plot_wdw.show()