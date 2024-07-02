from PyQt5.QtWidgets import QRadioButton, QButtonGroup, QMainWindow, QTableView, QAction, QMenu,QFileDialog, QLineEdit, QHBoxLayout, QWidget, QPushButton, QVBoxLayout, QComboBox, QLabel, QCheckBox, QMessageBox
from PyQt5.QtCore import Qt, QAbstractTableModel
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
plt.rcParams['svg.fonttype'] = 'none'
from celldetective.gui.gui_utils import FigureCanvas, center_window
from celldetective.utils import differentiate_per_track
import numpy as np
import seaborn as sns
import matplotlib.cm as mcm
import os
from celldetective.gui import Styles
from superqt import QColormapComboBox, QLabeledSlider, QSearchableComboBox
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from math import floor

from matplotlib import colormaps

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

	def __init__(self, parent_window):

		super().__init__()
		self.parent_window = parent_window
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
		self.setAttribute(Qt.WA_DeleteOnClose)

	def filter_table(self):
		try:
			query_text = self.query_le.text().replace('class', '`class`')
			tab = self.parent_window.data.query(query_text)
			self.subtable = TableUI(tab, query_text, plot_mode="scatter")
			self.subtable.show()
			self.close()
		except Exception as e:
			print(e)
			return None


class MergeOneHotWidget(QWidget, Styles):

	def __init__(self, parent_window, selected_columns=None):

		super().__init__()
		self.parent_window = parent_window
		self.selected_columns = selected_columns

		self.setWindowTitle("Merge one-hot encoded columns...")
		# Create the QComboBox and add some items
		center_window(self)
		
		self.layout = QVBoxLayout(self)
		self.layout.setContentsMargins(30,30,30,30)

		if self.selected_columns is not None:
			n_cols = len(self.selected_columns)
		else:
			n_cols = 2

		name_hbox = QHBoxLayout()
		name_hbox.addWidget(QLabel('New categorical column: '), 33)
		self.new_col_le = QLineEdit()
		self.new_col_le.setText('categorical_')
		self.new_col_le.textChanged.connect(self.allow_merge)
		name_hbox.addWidget(self.new_col_le, 66)
		self.layout.addLayout(name_hbox)


		self.layout.addWidget(QLabel('Source columns: '))

		self.cbs = [QSearchableComboBox() for i in range(n_cols)]
		self.cbs_layout = QVBoxLayout()

		for i in range(n_cols):
			lay = QHBoxLayout()
			lay.addWidget(QLabel(f'column {i}: '), 33)
			self.cbs[i].addItems(['--']+list(self.parent_window.data.columns))
			if self.selected_columns is not None:
				self.cbs[i].setCurrentText(self.selected_columns[i])
			lay.addWidget(self.cbs[i], 66)
			self.cbs_layout.addLayout(lay)

		self.layout.addLayout(self.cbs_layout)

		hbox = QHBoxLayout()
		self.add_col_btn = QPushButton('Add column')
		self.add_col_btn.clicked.connect(self.add_col)
		self.add_col_btn.setStyleSheet(self.button_add)
		self.add_col_btn.setIcon(icon(MDI6.plus,color="black"))

		hbox.addWidget(QLabel(''), 50)
		hbox.addWidget(self.add_col_btn, 50, alignment=Qt.AlignRight)
		self.layout.addLayout(hbox)

		self.submit_btn = QPushButton('Merge')
		self.submit_btn.setStyleSheet(self.button_style_sheet)
		self.submit_btn.clicked.connect(self.merge_cols)
		self.layout.addWidget(self.submit_btn, 30)

		self.setAttribute(Qt.WA_DeleteOnClose)

	def add_col(self):
		self.cbs.append(QSearchableComboBox())
		self.cbs[-1].addItems(['--']+list(self.parent_window.data.columns))
		lay = QHBoxLayout()
		lay.addWidget(QLabel(f'column {len(self.cbs)-1}: '), 33)
		lay.addWidget(self.cbs[-1], 66)
		self.cbs_layout.addLayout(lay)	

	def merge_cols(self):
		
		self.parent_window.data[self.new_col_le.text()] = self.parent_window.data.loc[:,list(self.selected_columns)].idxmax(axis=1)
		self.parent_window.model = PandasModel(self.parent_window.data)
		self.parent_window.table_view.setModel(self.parent_window.model)
		self.close()

	def allow_merge(self):

		if self.new_col_le.text()=='':
			self.submit_btn.setEnabled(False)
		else:
			self.submit_btn.setEnabled(True)


class DifferentiateColWidget(QWidget, Styles):

	def __init__(self, parent_window, column=None):

		super().__init__()
		self.parent_window = parent_window
		self.column = column

		self.setWindowTitle("d/dt")
		# Create the QComboBox and add some items
		center_window(self)
		
		layout = QVBoxLayout(self)
		layout.setContentsMargins(30,30,30,30)

		self.measurements_cb = QComboBox()
		self.measurements_cb.addItems(list(self.parent_window.data.columns))
		if self.column is not None:
			idx = self.measurements_cb.findText(self.column)
			self.measurements_cb.setCurrentIndex(idx)

		measurement_layout = QHBoxLayout()
		measurement_layout.addWidget(QLabel('measurements: '), 25)
		measurement_layout.addWidget(self.measurements_cb, 75)
		layout.addLayout(measurement_layout)

		self.window_size_slider = QLabeledSlider()
		self.window_size_slider.setRange(1,np.nanmax(self.parent_window.data.FRAME.to_numpy()))
		self.window_size_slider.setValue(3)
		window_layout = QHBoxLayout()
		window_layout.addWidget(QLabel('window size: '), 25)
		window_layout.addWidget(self.window_size_slider, 75)
		layout.addLayout(window_layout)

		self.backward_btn = QRadioButton('backward')
		self.bi_btn = QRadioButton('bi')
		self.bi_btn.click()
		self.forward_btn = QRadioButton('forward')
		self.mode_btn_group = QButtonGroup()
		self.mode_btn_group.addButton(self.backward_btn)
		self.mode_btn_group.addButton(self.bi_btn)
		self.mode_btn_group.addButton(self.forward_btn)

		mode_layout = QHBoxLayout()
		mode_layout.addWidget(QLabel('mode: '),25)
		mode_sublayout = QHBoxLayout()
		mode_sublayout.addWidget(self.backward_btn, 33, alignment=Qt.AlignCenter)
		mode_sublayout.addWidget(self.bi_btn, 33, alignment=Qt.AlignCenter)
		mode_sublayout.addWidget(self.forward_btn, 33, alignment=Qt.AlignCenter)
		mode_layout.addLayout(mode_sublayout, 75)
		layout.addLayout(mode_layout)

		self.submit_btn = QPushButton('Compute')
		self.submit_btn.setStyleSheet(self.button_style_sheet)
		self.submit_btn.clicked.connect(self.compute_derivative_and_add_new_column)
		layout.addWidget(self.submit_btn, 30)

		self.setAttribute(Qt.WA_DeleteOnClose)


	def compute_derivative_and_add_new_column(self):
		
		if self.bi_btn.isChecked():
			mode = 'bi'
		elif self.forward_btn.isChecked():
			mode = 'forward'
		elif self.backward_btn.isChecked():
			mode = 'backward'
		self.parent_window.data = differentiate_per_track(self.parent_window.data,
														  self.measurements_cb.currentText(),
														  window_size=self.window_size_slider.value(),
														  mode=mode)
		self.parent_window.model = PandasModel(self.parent_window.data)
		self.parent_window.table_view.setModel(self.parent_window.model)
		self.close()



class RenameColWidget(QWidget):

	def __init__(self, parent_window, column=None):

		super().__init__()
		self.parent_window = parent_window
		self.column = column
		if self.column is None:
			self.column = ''

		self.setWindowTitle("Rename column")
		# Create the QComboBox and add some items
		center_window(self)
		
		layout = QHBoxLayout(self)
		layout.setContentsMargins(30,30,30,30)
		self.new_col_name = QLineEdit()
		self.new_col_name.setText(self.column)
		layout.addWidget(self.new_col_name, 70)

		self.submit_btn = QPushButton('rename')
		self.submit_btn.clicked.connect(self.rename_col)
		layout.addWidget(self.submit_btn, 30)
		self.setAttribute(Qt.WA_DeleteOnClose)

	def rename_col(self):
		
		old_name = self.column
		new_name = self.new_col_name.text()
		self.parent_window.data = self.parent_window.data.rename(columns={old_name: new_name})
		print(self.parent.data.columns)

		self.parent_window.model = PandasModel(self.parent_window.data)
		self.parent_window.table_view.setModel(self.parent_window.model)
		self.close()


class TableUI(QMainWindow, Styles):
	def __init__(self, data, title, population='targets',plot_mode="plot_track_signals", *args, **kwargs):

		QMainWindow.__init__(self, *args, **kwargs)

		self.setWindowTitle(title)
		self.setGeometry(100,100,1000,400)
		center_window(self)
		self.title = title
		self.plot_mode = plot_mode
		self.population = population
		self.numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

		self._createMenuBar()
		self._createActions()

		self.table_view = QTableView(self)
		self.setCentralWidget(self.table_view)

		# Set the model for the table view
		self.data = data

		self.model = PandasModel(data)
		self.table_view.setModel(self.model)
		self.table_view.resizeColumnsToContents()
		self.setAttribute(Qt.WA_DeleteOnClose)


	def _createActions(self):

			self.save_as = QAction("&Save as...", self)
			self.save_as.triggered.connect(self.save_as_csv)
			self.save_as.setShortcut("Ctrl+s")
			self.fileMenu.addAction(self.save_as)

			self.save_inplace = QAction("&Save inplace...", self)
			self.save_inplace.triggered.connect(self.save_as_csv_inplace_per_pos)
			#self.save_inplace.setShortcut("Ctrl+s")
			self.fileMenu.addAction(self.save_inplace)

			self.plot_action = QAction("&Plot...", self)
			self.plot_action.triggered.connect(self.plot)
			self.plot_action.setShortcut("Ctrl+p")
			self.fileMenu.addAction(self.plot_action)		

			self.plot_inst_action = QAction("&Plot instantaneous...", self)
			self.plot_inst_action.triggered.connect(self.plot_instantaneous)
			self.plot_inst_action.setShortcut("Ctrl+i")
			self.fileMenu.addAction(self.plot_inst_action)	

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

			self.delete_action = QAction('&Delete...', self)
			self.delete_action.triggered.connect(self.delete_columns)
			self.delete_action.setShortcut(Qt.Key_Delete)
			self.editMenu.addAction(self.delete_action)

			self.rename_col_action = QAction('&Rename...', self)
			self.rename_col_action.triggered.connect(self.rename_column)
			#self.rename_col_action.setShortcut(Qt.Key_Delete)
			self.editMenu.addAction(self.rename_col_action)

			self.derivative_action = QAction('&Differentiate...', self)
			self.derivative_action.triggered.connect(self.differenciate_selected_feature)
			self.derivative_action.setShortcut("Ctrl+D")
			self.mathMenu.addAction(self.derivative_action)

			self.onehot_action = QAction('&One hot to categorical...', self)
			self.onehot_action.triggered.connect(self.transform_one_hot_cols_to_categorical)
			#self.onehot_action.setShortcut("Ctrl+D")
			self.mathMenu.addAction(self.onehot_action)		

	def delete_columns(self):

		x = self.table_view.selectedIndexes()
		col_idx = np.unique(np.array([l.column() for l in x]))
		cols = np.array(list(self.data.columns))

		msgBox = QMessageBox()
		msgBox.setIcon(QMessageBox.Question)
		msgBox.setText(f"You are about to delete columns {cols[col_idx]}... Do you want to proceed?")
		msgBox.setWindowTitle("Info")
		msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
		returnValue = msgBox.exec()
		if returnValue == QMessageBox.No:
			return None

		self.data = self.data.drop(list(cols[col_idx]),axis=1)
		self.model = PandasModel(self.data)
		self.table_view.setModel(self.model)

	def rename_column(self):

		x = self.table_view.selectedIndexes()
		col_idx = np.unique(np.array([l.column() for l in x]))

		if len(col_idx) == 0:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Question)
			msgBox.setText(f"Please select a column first.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None
			else:
				return None
			
		cols = np.array(list(self.data.columns))
		selected_col = str(cols[col_idx][0])

		self.renameWidget = RenameColWidget(self, selected_col)
		self.renameWidget.show()

	def save_as_csv_inplace_per_pos(self):

		print("Saving each table in its respective position folder...")
		for pos,pos_group in self.data.groupby('position'):
			pos_group.to_csv(pos+os.sep.join(['output', 'tables', f'trajectories_{self.population}.csv']), index=False)
		print("Done...")



	def differenciate_selected_feature(self):
		
		# check only one col selected and assert is numerical
		# open widget to select window parameters, directionality
		# create new col
		
		x = self.table_view.selectedIndexes()
		col_idx = np.unique(np.array([l.column() for l in x]))
		if col_idx!=0:
			cols = np.array(list(self.data.columns))
			selected_col = str(cols[col_idx][0])
		else:
			selected_col = None

		self.diffWidget = DifferentiateColWidget(self, selected_col)
		self.diffWidget.show()

	def transform_one_hot_cols_to_categorical(self):

		x = self.table_view.selectedIndexes()
		col_idx = np.unique(np.array([l.column() for l in x]))
		if list(col_idx):
			cols = np.array(list(self.data.columns))
			selected_cols = cols[col_idx]
		else:
			selected_cols = None

		self.mergewidget = MergeOneHotWidget(self, selected_columns=selected_cols)
		self.mergewidget.show()


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
		self.projectionWidget.setMinimumWidth(500)
		self.projectionWidget.setWindowTitle('Set projection mode')
		
		layout = QVBoxLayout()
		self.projectionWidget.setLayout(layout)

		self.projection_option = QRadioButton('global operation: ')
		self.projection_option.setToolTip('Collapse the cell track measurements with an operation over each track.')
		self.projection_option.toggled.connect(self.enable_projection_options)
		self.projection_op_cb = QComboBox()
		self.projection_op_cb.addItems(['mean','median','min','max', 'prod', 'sum'])

		projection_layout = QHBoxLayout()
		projection_layout.addWidget(self.projection_option, 33)
		projection_layout.addWidget(self.projection_op_cb, 66)
		layout.addLayout(projection_layout)

		self.event_time_option = QRadioButton('@event time: ')
		self.event_time_option.setToolTip('Pick the measurements at a specific event time.')
		self.event_time_option.toggled.connect(self.enable_projection_options)
		self.event_times_cb = QComboBox()
		cols = np.array(self.data.columns)
		time_cols = np.array([c.startswith('t_') for c in cols])
		time_cols = list(cols[time_cols])
		if 't0' in list(self.data.columns):
			time_cols.append('t0')
		self.event_times_cb.addItems(time_cols)
		self.event_times_cb.setEnabled(False)

		event_time_layout = QHBoxLayout()
		event_time_layout.addWidget(self.event_time_option, 33)
		event_time_layout.addWidget(self.event_times_cb, 66)
		layout.addLayout(event_time_layout)


		self.per_status_option = QRadioButton('per status: ')
		self.per_status_option.setToolTip('Collapse the cell track measurements independently for each of the cell state.')
		self.per_status_option.toggled.connect(self.enable_projection_options)
		self.per_status_cb = QComboBox()
		self.status_operation = QComboBox()
		self.status_operation.setEnabled(False)
		self.status_operation.addItems(['mean','median','min','max', 'prod', 'sum'])

		status_cols = np.array([c.startswith('status_') or c.startswith('group_') for c in cols])
		status_cols = list(cols[status_cols])
		if 'status' in list(self.data.columns):
			status_cols.append('status')
		self.per_status_cb.addItems(status_cols)
		self.per_status_cb.setEnabled(False)

		per_status_layout = QHBoxLayout()
		per_status_layout.addWidget(self.per_status_option, 33)
		per_status_layout.addWidget(self.per_status_cb, 66)
		layout.addLayout(per_status_layout)

		status_operation_layout = QHBoxLayout()
		status_operation_layout.addWidget(QLabel('operation: '), 33, alignment=Qt.AlignRight)
		status_operation_layout.addWidget(self.status_operation, 66)
		layout.addLayout(status_operation_layout)

		self.btn_projection_group = QButtonGroup()
		self.btn_projection_group.addButton(self.projection_option)
		self.btn_projection_group.addButton(self.event_time_option)
		self.btn_projection_group.addButton(self.per_status_option)

		apply_layout = QHBoxLayout()

		self.set_projection_btn = QPushButton('Apply')
		self.set_projection_btn.setStyleSheet(self.button_style_sheet)
		self.set_projection_btn.clicked.connect(self.set_proj_mode)
		apply_layout.addWidget(QLabel(''), 33)
		apply_layout.addWidget(self.set_projection_btn,33)
		apply_layout.addWidget(QLabel(''),33)
		layout.addLayout(apply_layout)

		self.projectionWidget.show()
		center_window(self.projectionWidget)

	def enable_projection_options(self):

		if self.projection_option.isChecked():
			self.projection_op_cb.setEnabled(True)
			self.event_times_cb.setEnabled(False)
			self.per_status_cb.setEnabled(False)
			self.status_operation.setEnabled(False)
		elif self.event_time_option.isChecked():
			self.projection_op_cb.setEnabled(False)
			self.event_times_cb.setEnabled(True)
			self.per_status_cb.setEnabled(False)
			self.status_operation.setEnabled(False)
		elif self.per_status_option.isChecked():
			self.projection_op_cb.setEnabled(False)
			self.event_times_cb.setEnabled(False)
			self.per_status_cb.setEnabled(True)
			self.status_operation.setEnabled(True)

	def set_1D_plot_params(self):

		self.plot1Dparams = QWidget()
		self.plot1Dparams.setWindowTitle('Set 1D plot parameters')
		
		layout = QVBoxLayout()
		self.plot1Dparams.setLayout(layout)

		layout.addWidget(QLabel('Representations: '))
		self.hist_check = QCheckBox('histogram')
		self.kde_check = QCheckBox('KDE plot')
		self.count_check = QCheckBox('Countplot')
		self.ecdf_check = QCheckBox('ECDF plot')
		self.swarm_check = QCheckBox('swarm')
		self.violin_check = QCheckBox('violin')
		self.strip_check = QCheckBox('strip')
		self.box_check = QCheckBox('Boxplot')
		self.boxenplot_check = QCheckBox('Boxenplot')

		layout.addWidget(self.hist_check)
		layout.addWidget(self.kde_check)
		layout.addWidget(self.count_check)
		layout.addWidget(self.ecdf_check)
		layout.addWidget(self.swarm_check)
		layout.addWidget(self.violin_check)
		layout.addWidget(self.strip_check)
		layout.addWidget(self.box_check)
		layout.addWidget(self.boxenplot_check)

		self.x_cb = QSearchableComboBox()
		self.x_cb.addItems(['--']+list(self.data.columns))

		self.y_cb = QSearchableComboBox()
		self.y_cb.addItems(['--']+list(self.data.columns))

		self.hue_cb = QSearchableComboBox()
		self.hue_cb.addItems(['--']+list(self.data.columns))
		idx = self.hue_cb.findText('--')
		self.hue_cb.setCurrentIndex(idx)

		# Set selected column

		try:
			x = self.table_view.selectedIndexes()
			col_idx = np.array([l.column() for l in x])
			row_idx = np.array([l.row() for l in x])
			column_names = self.data.columns
			unique_cols = np.unique(col_idx)[0]
			y = column_names[unique_cols]
			idx = self.y_cb.findText(y)
			self.y_cb.setCurrentIndex(idx)
		except:
			pass

		hbox = QHBoxLayout()
		hbox.addWidget(QLabel('x: '), 33)
		hbox.addWidget(self.x_cb, 66)
		layout.addLayout(hbox)

		hbox = QHBoxLayout()
		hbox.addWidget(QLabel('y: '), 33)
		hbox.addWidget(self.y_cb, 66)
		layout.addLayout(hbox)

		hbox = QHBoxLayout()
		hbox.addWidget(QLabel('hue: '), 33)
		hbox.addWidget(self.hue_cb, 66)
		layout.addLayout(hbox)

		self.cmap_cb = QColormapComboBox()
		for cm in list(colormaps):
			try:
				self.cmap_cb.addColormap(cm)
			except:
				pass
		
		hbox = QHBoxLayout()
		hbox.addWidget(QLabel('colormap: '), 33)
		hbox.addWidget(self.cmap_cb, 66)
		layout.addLayout(hbox)

		self.plot1d_btn = QPushButton('set')
		self.plot1d_btn.setStyleSheet(self.button_style_sheet)
		self.plot1d_btn.clicked.connect(self.plot1d)
		layout.addWidget(self.plot1d_btn)

		self.plot1Dparams.show()
		center_window(self.plot1Dparams)


	def plot1d(self):

		self.x_option = False
		if self.x_cb.currentText()!='--':
			self.x_option = True
			self.x = self.x_cb.currentText()

		self.fig, self.ax = plt.subplots(1,1,figsize=(4,3))
		self.plot1dWindow = FigureCanvas(self.fig, title="scatter")
		self.ax.clear()

		cmap = getattr(mcm, self.cmap_cb.currentText())

		try:
			hue_variable = self.hue_cb.currentText()
			colors = [cmap(i / len(self.data[hue_variable].unique())) for i in range(len(self.data[hue_variable].unique()))]
		except:
			colors = None

		if self.hue_cb.currentText()=='--':
			hue_variable = None

		if self.y_cb.currentText()=='--':
			self.y = None
		else:
			self.y = self.y_cb.currentText()

		if self.x_cb.currentText()=='--':
			self.x = None
		else:
			self.x = self.x_cb.currentText()

		legend=True
		if self.hist_check.isChecked():
			sns.histplot(data=self.data, x=self.x, hue=hue_variable, legend=legend, ax=self.ax, palette=colors, kde=True, common_norm=False, stat='density')
			legend = False
		if self.kde_check.isChecked():
			sns.kdeplot(data=self.data, x=self.x, hue=hue_variable, legend=legend, ax=self.ax, palette=colors, cut=0)
			legend = False
		if self.count_check.isChecked():
			sns.countplot(data=self.data, x=self.x, hue=hue_variable, legend=legend, ax=self.ax, palette=colors)
			legend = False
		if self.ecdf_check.isChecked():
			sns.ecdfplot(data=self.data, x=self.x, hue=hue_variable, legend=legend, ax=self.ax, palette=colors)
			legend = False

		if self.swarm_check.isChecked():
			if self.x_option:
				sns.swarmplot(data=self.data, x=self.x,y=self.y,dodge=True, hue=hue_variable,legend=legend, ax=self.ax, palette=colors)
				legend = False
			else:
				sns.swarmplot(data=self.data, y=self.y,dodge=True, hue=hue_variable,legend=legend, ax=self.ax, palette=colors)
				legend = False

		if self.violin_check.isChecked():
			if self.x_option:
				sns.stripplot(data=self.data,x=self.x, y=self.y,dodge=True, ax=self.ax, hue=hue_variable, legend=legend, palette=colors)
				legend = False
			else:
				sns.violinplot(data=self.data, y=self.y,dodge=True, hue=hue_variable,legend=legend, ax=self.ax, palette=colors, cut=0)
				legend = False

		if self.box_check.isChecked():
			if self.x_option:
				sns.boxplot(data=self.data, x=self.x, y=self.y,dodge=True, hue=hue_variable,legend=legend, ax=self.ax, fill=False,palette=colors, linewidth=2,)
				legend = False
			else:
				sns.boxplot(data=self.data, y=self.y,dodge=True, hue=hue_variable,legend=legend, ax=self.ax, fill=False,palette=colors, linewidth=2,)
				legend = False

		if self.boxenplot_check.isChecked():
			if self.x_option:
				sns.boxenplot(data=self.data, x=self.x, y=self.y,dodge=True, hue=hue_variable,legend=legend, ax=self.ax, fill=False,palette=colors, linewidth=2,)
				legend = False
			else:
				sns.boxenplot(data=self.data, y=self.y,dodge=True, hue=hue_variable,legend=legend, ax=self.ax, fill=False,palette=colors, linewidth=2,)
				legend = False

		if self.strip_check.isChecked():
			if self.x_option:
				sns.stripplot(data=self.data, x = self.x, y=self.y,dodge=True, ax=self.ax, hue=hue_variable, legend=legend, palette=colors)
				legend = False
			else:
				sns.stripplot(data=self.data, y=self.y,dodge=True, ax=self.ax, hue=hue_variable, legend=legend, palette=colors)
				legend = False

		plt.tight_layout()
		self.fig.set_facecolor('none')  # or 'None'
		self.fig.canvas.setStyleSheet("background-color: transparent;")
		self.plot1dWindow.canvas.draw()
		self.plot1dWindow.show()


	def set_proj_mode(self):
		
		self.static_columns = ['well_index', 'well_name', 'pos_name', 'position', 'well', 'status', 't0', 'class','cell_type','concentration', 'antibody', 'pharmaceutical_agent','TRACK_ID','position']

		if self.projection_option.isChecked():

			self.projection_mode = self.projection_op_cb.currentText()
			op = getattr(self.data.groupby(['position', 'TRACK_ID']), self.projection_mode)
			group_table = op(self.data.groupby(['position', 'TRACK_ID']))

			for c in self.static_columns:
				try:
					group_table[c] = self.data.groupby(['position','TRACK_ID'])[c].apply(lambda x: x.unique()[0])
				except Exception as e:
					print(e)
					pass
			
			for col in ['TRACK_ID']:
				first_column = group_table.pop(col) 
				group_table.insert(0, col, first_column)
			group_table.pop('FRAME')


		elif self.event_time_option.isChecked():
			time_of_interest = self.event_times_cb.currentText()
			self.projection_mode = f"measurements at {time_of_interest}"
			new_table = []
			for tid,group in self.data.groupby(['position','TRACK_ID']):
				time = group[time_of_interest].values[0]
				if time==time:
					time = floor(time) # floor for onset
				else:
					continue
				frames = group['FRAME'].values
				values = group.loc[group['FRAME']==time,:].to_numpy()
				if len(values)>0:
					values = dict(zip(list(self.data.columns), values[0]))
					values.update({'TRACK_ID': tid[1]})
					values.update({'position': tid[0]})
					new_table.append(values)
			
			group_table = pd.DataFrame(new_table)
			for col in ['TRACK_ID']:
				first_column = group_table.pop(col) 
				group_table.insert(0, col, first_column)
			
			group_table = group_table.sort_values(by=['position','TRACK_ID','FRAME'],ignore_index=True)
			group_table = group_table.reset_index(drop=True)


		elif self.per_status_option.isChecked():

			status_of_interest = self.per_status_cb.currentText()
			self.projection_mode = f'{self.status_operation.currentText()} per {status_of_interest}'
			self.data = self.data.dropna(subset=status_of_interest,ignore_index=True)
			unique_statuses = np.unique(self.data[status_of_interest].to_numpy())

			df_sections = []
			for s in unique_statuses:
				subtab = self.data.loc[self.data[status_of_interest]==s,:]
				op = getattr(subtab.groupby(['position', 'TRACK_ID']), self.status_operation.currentText())
				subtab_projected = op(subtab.groupby(['position', 'TRACK_ID']))
				frame_duration = subtab.groupby(['position','TRACK_ID']).size().to_numpy()
				for c in self.static_columns:
					try:
						subtab_projected[c] = subtab.groupby(['position', 'TRACK_ID'])[c].apply(lambda x: x.unique()[0])
					except Exception as e:
						print(e)
						pass
				subtab_projected['duration_in_state'] = frame_duration
				df_sections.append(subtab_projected)

			group_table = pd.concat(df_sections,axis=0,ignore_index=True)
			for col in ['duration_in_state',status_of_interest,'TRACK_ID']:
				first_column = group_table.pop(col) 
				group_table.insert(0, col, first_column)
			group_table.pop('FRAME')
			group_table = group_table.sort_values(by=['position','TRACK_ID',status_of_interest],ignore_index=True)
			group_table = group_table.reset_index(drop=True)


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
		self.editMenu = QMenu("&Edit", self)
		menuBar.addMenu(self.editMenu)
		self.mathMenu = QMenu('&Math', self)
		menuBar.addMenu(self.mathMenu)

	def save_as_csv(self):
		options = QFileDialog.Options()
		options |= QFileDialog.ReadOnly
		file_name, _ = QFileDialog.getSaveFileName(self, "Save as .csv", "","CSV Files (*.csv);;All Files (*)", options=options)
		if file_name:
			if not file_name.endswith(".csv"):
				file_name += ".csv"
			self.data.to_csv(file_name, index=False)

	def test_bool(self, array):
		if array.dtype=="bool":
			return np.array(array, dtype=int)
		else:
			return array

	def plot_instantaneous(self):
		
		if self.plot_mode=='plot_track_signals':
			self.plot_mode = 'static'
			self.plot()
			self.plot_mode = 'plot_track_signals'
		elif self.plot_mode=="static":
			self.plot()

	def plot(self):
		if self.plot_mode == "static":
	
			x = self.table_view.selectedIndexes()
			col_idx = [l.column() for l in x]
			row_idx = [l.row() for l in x]
			column_names = self.data.columns
			unique_cols = np.unique(col_idx)

			if len(unique_cols)==1 or len(unique_cols)==0:
				self.set_1D_plot_params()

			if len(unique_cols) == 2:

				print("two columns, plot mode")
				x1 = self.test_bool(self.data.iloc[row_idx, unique_cols[0]])
				x2 = self.test_bool(self.data.iloc[row_idx, unique_cols[1]])

				self.fig, self.ax = plt.subplots(1, 1, figsize=(4,3))
				self.scatter_wdw = FigureCanvas(self.fig, title="scatter")
				self.ax.clear()
				self.ax.scatter(x1,x2)
				self.ax.set_xlabel(column_names[unique_cols[0]])
				self.ax.set_ylabel(column_names[unique_cols[1]])
				plt.tight_layout()
				self.fig.set_facecolor('none')  # or 'None'
				self.fig.canvas.setStyleSheet("background-color: transparent;")
				self.scatter_wdw.canvas.draw()
				self.scatter_wdw.show()

			else:
				print("please select less columns")

		elif self.plot_mode == "plot_timeseries":
			print("mode plot frames")
			x = self.table_view.selectedIndexes()
			col_idx = np.array([l.column() for l in x])
			row_idx = np.array([l.row() for l in x])
			column_names = self.data.columns
			unique_cols = np.unique(col_idx)

			self.fig, self.ax = plt.subplots(1, 1, figsize=(4, 3))
			self.plot_wdw = FigureCanvas(self.fig, title="scatter")
			self.ax.clear()
			for k in range(len(unique_cols)):
				row_idx_i = row_idx[np.where(col_idx == unique_cols[k])[0]]
				y = self.data.iloc[row_idx_i, unique_cols[k]]
				self.ax.plot(self.data["timeline"][row_idx_i], y, label=column_names[unique_cols[k]])

			self.ax.legend()
			self.ax.set_xlabel("time [frame]")
			self.ax.set_ylabel(self.title)
			plt.tight_layout()
			self.fig.set_facecolor('none')  # or 'None'
			self.fig.canvas.setStyleSheet("background-color: transparent;")
			self.plot_wdw.canvas.draw()
			plt.show()

		elif self.plot_mode == "plot_track_signals":

			print("mode plot track signals")
			print('we plot here')

			x = self.table_view.selectedIndexes()
			col_idx = np.array([l.column() for l in x])
			row_idx = np.array([l.row() for l in x])
			column_names = self.data.columns
			unique_cols = np.unique(col_idx)

			if len(unique_cols) > 2:
				fig,ax = plt.subplots(1, 1, figsize=(7, 5.5))
				for k in range(len(unique_cols)):

					row_idx_i = row_idx[np.where(col_idx == unique_cols[k])[0]]
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

			if len(unique_cols) == 2:

				self.fig, self.ax = plt.subplots(1, 1, figsize=(4, 3))
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

			if len(unique_cols) == 1:
				
				self.fig, self.ax = plt.subplots(1, 1, figsize=(4, 3))
				self.plot_wdw = FigureCanvas(self.fig, title="scatter")
				self.ax.clear()
				
				# if 't0' in list(self.data.columns):
				# 	ref_time_col = 't0'
				# else:
				# 	ref_time_col = 'FRAME'

				for w,well_group in self.data.groupby('well_name'):
					for pos,pos_group in well_group.groupby('pos_name'):
						for tid,group_track in pos_group.groupby('TRACK_ID'):
							self.ax.plot(group_track["FRAME"], group_track[column_names[unique_cols[0]]],c="k", alpha = 0.1)
				self.ax.set_xlabel(r"$t$ [frame]")
				self.ax.set_ylabel(column_names[unique_cols[0]])
				plt.tight_layout()
				self.fig.set_facecolor('none')  # or 'None'
				self.fig.canvas.setStyleSheet("background-color: transparent;")
				self.plot_wdw.canvas.draw()
				self.plot_wdw.show()