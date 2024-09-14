from PyQt5.QtWidgets import QMessageBox,QGridLayout, QButtonGroup, \
	QCheckBox, QLineEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton, \
	QRadioButton, QFileDialog
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QDoubleValidator

from celldetective.gui.gui_utils import center_window, FigureCanvas
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from celldetective.signals import mean_signal
import numpy as np
import json
import os
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from glob import glob
from matplotlib.cm import tab10
from celldetective.gui import Styles
import matplotlib.cm as mcm


class GenericSignalPlotWidget(QWidget, Styles):

	def __init__(self, df = None, df_pos_info = None, df_well_info = None, feature_selected = None, parent_window=None, title='plot', *args, **kwargs):
		
		super().__init__()

		self.parent_window = parent_window
		self.setWindowTitle(title)

		self.show_ci = False
		self.legend_visible = True
		self.show_cell_lines = False
		self.scaling_factor = 1
		self.target_class = [0]
		self.float_validator = QDoubleValidator()

		cmap_lbl = self.parent_window.cbs[-1].currentText()
		self.cmap = getattr(mcm, cmap_lbl)

		self.feature_selected = feature_selected
		self.df = df
		self.df_pos_info = df_pos_info
		self.df_well_info = df_well_info

		self.layout = QVBoxLayout()
		self.layout.setSpacing(3)
		self.populate_widget()

		self.ci_btn.click()
		#self.legend_btn.click()

		center_window(self)
		self.setLayout(self.layout)
		self.setAttribute(Qt.WA_DeleteOnClose)

	def populate_widget(self):

		self.plot_options = [QRadioButton() for i in range(3)]
		self.radio_labels = ['well', 'position', 'both']
		radio_hbox = QHBoxLayout()
		radio_hbox.setContentsMargins(15,15,15,0)
		
		self.group_lbl = QLabel('grouping: ')
		radio_hbox.addWidget(self.group_lbl, 25)
		radio_subhbox = QHBoxLayout()
		radio_hbox.addLayout(radio_subhbox, 75)

		self.plot_btn_group = QButtonGroup()
		for i in range(3):
			self.plot_options[i].setText(self.radio_labels[i])
			self.plot_btn_group.addButton(self.plot_options[i])
			radio_subhbox.addWidget(self.plot_options[i], 33, alignment=Qt.AlignCenter)

		if self.parent_window.position_indices is not None:
			if len(self.parent_window.well_indices)>1 and len(self.parent_window.position_indices)==1:
				self.plot_btn_group.buttons()[0].click()
				for i in [1,2]:
					self.plot_options[i].setEnabled(False)
			elif len(self.parent_window.well_indices)>1:
				self.plot_btn_group.buttons()[0].click()
			elif len(self.parent_window.well_indices)==1 and len(self.parent_window.position_indices)==1:
				self.plot_btn_group.buttons()[1].click()
				for i in [0,2]:
					self.plot_options[i].setEnabled(False)
		else:
			if len(self.parent_window.well_indices)>1:
				self.plot_btn_group.buttons()[0].click()
			elif len(self.parent_window.well_indices)==1:
				self.plot_btn_group.buttons()[2].click()

		self.layout.addLayout(radio_hbox)


		plot_buttons_hbox = QHBoxLayout()
		plot_buttons_hbox.setContentsMargins(10,10,5,0)
		plot_buttons_hbox.addWidget(QLabel(''),80, alignment=Qt.AlignLeft)

		self.legend_btn = QPushButton('')
		self.legend_btn.setIcon(icon(MDI6.text_box,color=self.help_color))
		self.legend_btn.setStyleSheet(self.button_select_all)
		self.legend_btn.setToolTip('Show or hide the legend')
		self.legend_btn.setIconSize(QSize(20, 20))
		plot_buttons_hbox.addWidget(self.legend_btn, 5,alignment=Qt.AlignRight)

		self.log_btn = QPushButton('')
		self.log_btn.setIcon(icon(MDI6.math_log,color="black"))
		self.log_btn.setStyleSheet(self.button_select_all)
		self.log_btn.setIconSize(QSize(20, 20))
		self.log_btn.setToolTip('Enable or disable log scale')
		plot_buttons_hbox.addWidget(self.log_btn, 5, alignment=Qt.AlignRight)

		self.ci_btn = QPushButton('')
		self.ci_btn.setIcon(icon(MDI6.arrow_expand_horizontal,color="black"))
		self.ci_btn.setStyleSheet(self.button_select_all)
		self.ci_btn.setIconSize(QSize(20, 20))
		self.ci_btn.setToolTip('Show or hide confidence intervals.')
		plot_buttons_hbox.addWidget(self.ci_btn, 5, alignment=Qt.AlignRight)

		self.cell_lines_btn = QPushButton('')
		self.cell_lines_btn.setIcon(icon(MDI6.view_headline,color="black"))
		self.cell_lines_btn.setStyleSheet(self.button_select_all)
		self.cell_lines_btn.setToolTip('Show or hide individual cell signals.')
		self.cell_lines_btn.setIconSize(QSize(20, 20))
		plot_buttons_hbox.addWidget(self.cell_lines_btn, 5, alignment=Qt.AlignRight)

		self.export_btn = QPushButton('')
		self.export_btn.setIcon(icon(MDI6.content_save,color="black"))
		self.export_btn.setStyleSheet(self.button_select_all)
		self.export_btn.setToolTip('Export figure.')
		self.export_btn.setIconSize(QSize(20, 20))
		plot_buttons_hbox.addWidget(self.export_btn, 5, alignment=Qt.AlignRight)

		self.layout.addLayout(plot_buttons_hbox)

		self.fig, self.ax = plt.subplots(1,1,figsize=(4,3))
		#self.setMinimumHeight(100)
		self.plot_widget = FigureCanvas(self.fig, title="")
		self.plot_widget.setContentsMargins(0,0,0,0)
		# if self.df is not None:
		self.initialize_axis()
		plt.tight_layout()

		self.ax.set_prop_cycle('color',[self.cmap(i) for i in np.linspace(0, 1, len(self.parent_window.well_indices))])

		self.fig.set_facecolor('none')  # or 'None'
		self.fig.canvas.setStyleSheet("background-color: transparent;")
		self.plot_widget.canvas.draw()

		self.layout.addWidget(self.plot_widget)

		self.plot_btn_group.buttonClicked[int].connect(self.plot_signals)
		self.legend_btn.clicked.connect(self.show_hide_legend)
		self.log_btn.clicked.connect(self.switch_to_log)
		self.ci_btn.clicked.connect(self.switch_ci)
		self.cell_lines_btn.clicked.connect(self.switch_cell_lines)
		self.export_btn.clicked.connect(self.save_plot)


		self.class_selection_widget = QWidget()

		self.class_selection_lbl = QLabel('class of interest:')
		class_hbox = QHBoxLayout()
		self.class_selection_widget.setLayout(class_hbox)
		class_hbox.addWidget(self.class_selection_lbl, 25, alignment=Qt.AlignLeft)

		class_subhbox = QHBoxLayout()
		class_hbox.addLayout(class_subhbox, 75)

		self.all_btn = QRadioButton('*')
		self.event_btn = QRadioButton('event')
		self.event_btn.setChecked(True)
		self.no_event_btn = QRadioButton('no event')
		self.class_btn_group = QButtonGroup()
		for btn in [self.all_btn, self.event_btn, self.no_event_btn]:
			self.class_btn_group.addButton(btn)

		self.class_btn_group.buttonClicked[int].connect(self.set_class_to_plot)

		class_subhbox.addWidget(self.all_btn, 33, alignment=Qt.AlignCenter)
		class_subhbox.addWidget(self.event_btn, 33, alignment=Qt.AlignCenter)
		class_subhbox.addWidget(self.no_event_btn, 33, alignment=Qt.AlignCenter)

		self.layout.addWidget(self.class_selection_widget) #Layout(class_hbox)

		# Rescale 
		self.rescale_widget = QWidget()

		scale_hbox = QHBoxLayout()
		self.rescale_widget.setLayout(scale_hbox)

		scale_hbox.addWidget(QLabel('scaling factor: '), 25)
		self.scaling_factor_le = QLineEdit('1')
		self.scaling_factor_le.setValidator(self.float_validator)
		scale_hbox.addWidget(self.scaling_factor_le, 65)

		self.rescale_btn = QPushButton('rescale')
		self.rescale_btn.setStyleSheet(self.button_style_sheet_2)
		self.rescale_btn.clicked.connect(self.rescale_y_axis)
		scale_hbox.addWidget(self.rescale_btn, 10)
		#self.layout.addLayout(scale_hbox)
		self.layout.addWidget(self.rescale_widget)

		self.select_option = [QRadioButton() for i in range(2)]
		self.select_label = ['by name', 'spatially']

		select_hbox = QHBoxLayout()
		select_hbox.addWidget(QLabel('select position: '), 25)

		select_subhbox = QHBoxLayout()
		select_hbox.addLayout(select_subhbox, 75)

		self.select_btn_group = QButtonGroup()
		for i in range(2):
			self.select_option[i].setText(self.select_label[i])
			self.select_btn_group.addButton(self.select_option[i])
			select_subhbox.addWidget(self.select_option[i],33, alignment=Qt.AlignCenter)
		self.select_option[0].setChecked(True)
		self.select_btn_group.buttonClicked[int].connect(self.switch_selection_mode)
		self.layout.addLayout(select_hbox)

		self.look_for_metadata()
		if self.metadata_found:
			self.fig_scatter, self.ax_scatter = plt.subplots(1,1,figsize=(4,3))
			self.position_scatter = FigureCanvas(self.fig_scatter)
			self.load_coordinates()
			self.plot_spatial_location()
			self.ax_scatter.spines['top'].set_visible(False)
			self.ax_scatter.spines['right'].set_visible(False)
			self.ax_scatter.set_aspect('equal')
			self.ax_scatter.set_xticks([])
			self.ax_scatter.set_yticks([])
			plt.tight_layout()

			self.fig_scatter.set_facecolor('none')  # or 'None'
			self.fig_scatter.canvas.setStyleSheet("background-color: transparent;")
			self.layout.addWidget(self.position_scatter)

		self.generate_pos_selection_widget()
		# self.select_btn_group.buttons()[0].click()

	def rescale_y_axis(self):
		new_scale = self.scaling_factor_le.text().replace(',','.')
		if new_scale=='':
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("Please set a valid scaling factor...")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None
		else:
			self.scaling_factor = float(new_scale)
			self.plot_signals(0)

	def save_plot(self):

		fileName, _ = QFileDialog.getSaveFileName(self, 
			"Save Image", self.parent_window.exp_dir+os.sep+'plot.png', "Images (*.png *.xpm *.jpg *.svg)") #, options=options
		if fileName:
			plt.tight_layout()
			plt.savefig(fileName, bbox_inches='tight', dpi=300)

	def switch_selection_mode(self, id):

		for i in range(2):
			if self.select_option[i].isChecked():
				self.selection_mode = self.select_label[i]
		if self.selection_mode=='by name':
			if len(self.metafiles)>0:
				self.position_scatter.hide()
			self.line_choice_widget.show()
		else:
			if len(self.metafiles)>0:
				self.position_scatter.show()
			self.line_choice_widget.hide()

	def set_class_to_plot(self):

		if self.all_btn.isChecked():
			self.target_class=[0,1]
		elif self.event_btn.isChecked():
			self.target_class = [0]
		else:
			self.target_class = [1]

		self.plot_signals(0)

	def generate_pos_selection_widget(self):
		
		self.well_names = self.df['well_name'].unique()
		self.pos_names = self.df_pos_info['pos_name'].unique() #pd.DataFrame(self.ks_estimators_per_position)['position_name'].unique()

		self.usable_well_labels = []
		for name in self.well_names:
			for lbl in self.parent_window.well_labels:
				if name+':' in lbl:
					self.usable_well_labels.append(lbl)

		self.line_choice_widget = QWidget()
		self.line_check_vbox = QGridLayout()
		self.line_choice_widget.setLayout(self.line_check_vbox)

		if len(self.parent_window.well_indices)>1:
			self.well_display_options = [QCheckBox(self.usable_well_labels[i]) for i in range(len(self.usable_well_labels))]
			for i in range(len(self.well_names)):
				self.line_check_vbox.addWidget(self.well_display_options[i], i, 0, 1, 1, alignment=Qt.AlignLeft)
				self.well_display_options[i].setChecked(True)
				self.well_display_options[i].setStyleSheet("font-size: 12px;")
				self.well_display_options[i].toggled.connect(self.select_lines)
		else:
			self.pos_display_options = [QCheckBox(self.pos_names[i]) for i in range(len(self.pos_names))]
			for i in range(len(self.pos_names)):
				self.line_check_vbox.addWidget(self.pos_display_options[i], i%4, i//4, 1, 1, alignment=Qt.AlignCenter)
				self.pos_display_options[i].setChecked(True)
				self.pos_display_options[i].setStyleSheet("font-size: 12px;")
				self.pos_display_options[i].toggled.connect(self.select_lines)

		self.layout.addWidget(self.line_choice_widget)
		#self.layout.addLayout(self.line_check_vbox)

	def look_for_metadata(self):
		self.metadata_found = False
		self.metafiles = glob(self.parent_window.exp_dir+os.sep.join([f'W*','*','movie','*metadata.txt'])) \
					+ glob(self.parent_window.exp_dir+os.sep.join([f'W*','*','*metadata.txt'])) \
					+ glob(self.parent_window.exp_dir+os.sep.join([f'W*','*metadata.txt'])) \
					+ glob(self.parent_window.exp_dir+'*metadata.txt')
		print(f'Found {len(self.metafiles)} metadata files...')
		if len(self.metafiles)>0:
			self.metadata_found = True

	def load_coordinates(self):

		"""
		Read metadata and try to extract position coordinates
		"""

		self.no_meta = False
		try:
			with open(self.metafiles[0], 'r') as f:
				data = json.load(f)
				positions = data['Summary']['InitialPositionList']
		except Exception as e:
			print(f'Trouble loading metadata: error {e}...')
			return None

		for k in range(len(positions)):
			pos_label = positions[k]['Label']
			try:
				coords = positions[k]['DeviceCoordinatesUm']['XYStage']
			except:
				try:
					coords = positions[k]['DeviceCoordinatesUm']['PIXYStage']
				except:
					self.no_meta = True

			if not self.no_meta:
				files = self.df_pos_info['stack_path'].values
				pos_loc = [pos_label in f for f in files]
				self.df_pos_info.loc[pos_loc, 'x'] = coords[0]
				self.df_pos_info.loc[pos_loc, 'y'] = coords[1]
				self.df_pos_info.loc[pos_loc, 'metadata_tag'] = pos_label


	def plot_spatial_location(self):

		try:
			self.sc = self.ax_scatter.scatter(self.df_pos_info["x"].values, self.df_pos_info["y"].values, picker=True, pickradius=1, color=self.select_color(self.df_pos_info["select"].values))
			self.scat_labels = self.df_pos_info['metadata_tag'].values
			self.ax_scatter.invert_xaxis()
			self.annot = self.ax_scatter.annotate("", xy=(0,0), xytext=(10,10),textcoords="offset points",
								bbox=dict(boxstyle="round", fc="w"),
								arrowprops=dict(arrowstyle="->"))
			self.annot.set_visible(False)
			self.fig_scatter.canvas.mpl_connect("motion_notify_event", self.hover)
			self.fig_scatter.canvas.mpl_connect("pick_event", self.unselect_position)
		except Exception as e:
			pass

	def select_color(self, selection):
		colors = [tab10(0) if s else tab10(0.1) for s in selection]
		return colors

	def initialize_axis(self):

		previous_ymin, previous_ymax = self.ax.get_ylim()
		previous_legend = self.legend_visible
		is_log = self.ax.get_yscale()

		self.ax.clear()
		self.ax.plot([],[])

		# Labels
		self.ax.set_xlabel('time [min]')
		self.ax.set_ylabel(self.feature_selected)

		# Spines
		self.ax.spines['top'].set_visible(False)
		self.ax.spines['right'].set_visible(False)

		# Lims
		safe_df = self.df.dropna(subset=self.feature_selected)
		values = safe_df[self.feature_selected].values
		if len(values)>0:
			self.ax.set_ylim(np.percentile(values, 1)*self.scaling_factor, np.percentile(values, 99)*self.scaling_factor)
		self.ax.set_xlim(-(self.df['FRAME'].max()+2)*self.parent_window.FrameToMin,(self.df['FRAME'].max()+2)*self.parent_window.FrameToMin)
		
		if is_log=='log':
			self.ax.set_yscale('log')
		if previous_legend:
			leg = self.ax.get_legend()
			if leg is not None:
				leg.set_visible(True)

	def show_hide_legend(self):
		
		if self.legend_visible:
			leg = self.ax.get_legend()
			leg.set_visible(False)
			self.legend_visible = False
			self.legend_btn.setIcon(icon(MDI6.text_box,color="black"))
		else:
			leg = self.ax.get_legend()
			leg.set_visible(True)
			self.legend_visible = True
			self.legend_btn.setIcon(icon(MDI6.text_box,color=self.help_color))

		self.plot_widget.canvas.draw_idle()

	def switch_to_log(self):

		"""
		Switch threshold histogram to log scale. Auto adjust.
		"""

		if self.ax.get_yscale()=='linear':
			self.ax.set_yscale('log')
			self.log_btn.setIcon(icon(MDI6.math_log,color=self.help_color))
			#self.ax.set_ylim(0.01,1.05)
		else:
			self.ax.set_yscale('linear')
			self.log_btn.setIcon(icon(MDI6.math_log,color="black"))
			#self.ax.set_ylim(0.01,1.05)

		#self.ax.autoscale()
		self.plot_widget.canvas.draw_idle()

	def plot_signals(self, id):

		for i in range(3):
			if self.plot_options[i].isChecked():
				self.plot_mode = self.radio_labels[i]

		if self.target_class==[0,1]:
			mean_signal = 'mean_all'
			std_signal = 'std_all'
			matrix = 'matrix_all'
		elif self.target_class==[0]:
			mean_signal = 'mean_event'
			std_signal = 'std_event'
			matrix = 'matrix_event'
		else:
			mean_signal = 'mean_no_event'
			std_signal = 'std_no_event'
			matrix = 'matrix_no_event'


		colors = np.array([self.cmap(i / len(self.df_pos_info)) for i in range(len(self.df_pos_info))])
		well_color = [self.cmap(i / len(self.df_well_info)) for i in range(len(self.df_well_info))]

		if self.plot_mode=='position':
			self.initialize_axis()
			lines = self.df_pos_info.loc[self.df_pos_info['select'],'signal'].values
			pos_labels = self.df_pos_info.loc[self.df_pos_info['select'],'pos_name'].values
			pos_indices = self.df_pos_info.loc[self.df_pos_info['select'],'pos_index'].values
			well_index = self.df_pos_info.loc[self.df_pos_info['select'],'well_index'].values
			for i in range(len(lines)):
				if len(self.parent_window.well_indices)<=1:
					self.plot_line(lines[i], colors[pos_indices[i]], pos_labels[i], mean_signal, std_signal=std_signal, ci_option=self.show_ci, cell_lines_option=self.show_cell_lines, matrix=matrix)
				else:
					self.plot_line(lines[i], well_color[well_index[i]], pos_labels[i], mean_signal, std_signal=std_signal, ci_option=self.show_ci, cell_lines_option=self.show_cell_lines, matrix=matrix)
			if self.legend_visible:
				self.ax.legend(ncols=3,fontsize='x-small')

		elif self.plot_mode=='well':
			self.initialize_axis()
			lines = self.df_well_info.loc[self.df_well_info['select'],'signal'].values	
			well_index = self.df_well_info.loc[self.df_well_info['select'],'well_index'].values
			well_labels = self.df_well_info.loc[self.df_well_info['select'],'well_name'].values
			for i in range(len(lines)):	
				if len(self.parent_window.well_indices)<=1:
					self.plot_line(lines[i], 'k', well_labels[i], mean_signal, std_signal=std_signal, ci_option=self.show_ci, cell_lines_option=self.show_cell_lines,matrix=matrix)
				else:
					self.plot_line(lines[i], well_color[well_index[i]], well_labels[i], mean_signal, std_signal=std_signal, ci_option=self.show_ci, cell_lines_option=self.show_cell_lines,matrix=matrix)
			if self.legend_visible:
				self.ax.legend(ncols=2, fontsize='x-small')

		elif self.plot_mode=='both':
			
			self.initialize_axis()
			lines_pos = self.df_pos_info.loc[self.df_pos_info['select'],'signal'].values
			lines_well = self.df_well_info.loc[self.df_well_info['select'],'signal'].values	

			pos_indices = self.df_pos_info.loc[self.df_pos_info['select'],'pos_index'].values
			well_index_pos = self.df_pos_info.loc[self.df_pos_info['select'],'well_index'].values
			well_index = self.df_well_info.loc[self.df_well_info['select'],'well_index'].values
			well_labels = self.df_well_info.loc[self.df_well_info['select'],'well_name'].values
			pos_labels = self.df_pos_info.loc[self.df_pos_info['select'],'pos_name'].values

			for i in range(len(lines_pos)):
				if len(self.parent_window.well_indices)<=1:
					self.plot_line(lines_pos[i], colors[pos_indices[i]], pos_labels[i], mean_signal, std_signal=std_signal, ci_option=self.show_ci, cell_lines_option=self.show_cell_lines,matrix=matrix)
				else:
					self.plot_line(lines_pos[i], well_color[well_index_pos[i]], None, mean_signal, std_signal=std_signal, ci_option=False)

			for i in range(len(lines_well)):
				if len(self.parent_window.well_indices)<=1:
					self.plot_line(lines_well[i], 'k', 'pool', mean_signal, std_signal=std_signal, ci_option=False)
				else:
					self.plot_line(lines_well[i], well_color[well_index[i]], well_labels[i], mean_signal, std_signal=std_signal, ci_option=False)
			if self.legend_visible:
				self.ax.legend(ncols=3,fontsize='x-small')

		self.plot_widget.canvas.draw()

	def plot_line(self, line, color, label, mean_signal, ci_option=True, cell_lines_option=False, alpha_ci=0.5, alpha_cell_lines=0.5, std_signal=None, matrix=None):
		
		# Plot a signal
		if line==line:
			self.ax.plot(line['timeline']*self.parent_window.FrameToMin, line[mean_signal]*self.scaling_factor, color=color, label=label)

			if ci_option and std_signal is not None:

				self.ax.fill_between(line['timeline']*self.parent_window.FrameToMin,
								[a-b for a,b in zip(line[mean_signal]*self.scaling_factor, line[std_signal]*self.scaling_factor)],
								[a+b for a,b in zip(line[mean_signal]*self.scaling_factor, line[std_signal]*self.scaling_factor)],
								color=color,
								alpha=alpha_ci,
								)
			if cell_lines_option and matrix is not None:
				# Show individual cell signals
				mat = line[matrix]
				for i in range(mat.shape[0]):
					self.ax.plot(line['timeline']*self.parent_window.FrameToMin, mat[i,:]*self.scaling_factor, color=color, alpha=alpha_cell_lines)


	def switch_ci(self):

		# Show the confidence interval / STD

		if self.show_ci:
			self.ci_btn.setIcon(icon(MDI6.arrow_expand_horizontal,color="black"))
		else:
			self.ci_btn.setIcon(icon(MDI6.arrow_expand_horizontal,color=self.help_color))
		self.show_ci = not self.show_ci
		self.plot_signals(0)

	def switch_cell_lines(self):

		# Show individual cell signals

		if self.show_cell_lines:
			self.cell_lines_btn.setIcon(icon(MDI6.view_headline,color="black"))
		else:
			self.cell_lines_btn.setIcon(icon(MDI6.view_headline,color=self.help_color))
		self.show_cell_lines = not self.show_cell_lines
		self.plot_signals(0)

	def select_lines(self):
		
		if len(self.parent_window.well_indices)>1:
			for i in range(len(self.well_display_options)):
				self.df_well_info.loc[self.df_well_info['well_index']==i,'select'] = self.well_display_options[i].isChecked()
				self.df_pos_info.loc[self.df_pos_info['well_index']==i,'select'] = self.well_display_options[i].isChecked()
		else:
			for i in range(len(self.pos_display_options)):
				self.df_pos_info.loc[self.df_pos_info['pos_index']==i,'select'] = self.pos_display_options[i].isChecked()

		if len(self.metafiles)>0:
			self.sc.set_color(self.select_color(self.df_pos_info["select"].values))
			self.position_scatter.canvas.draw_idle()
		self.plot_signals(0)

class SurvivalPlotWidget(GenericSignalPlotWidget):

	def __init__(self, *args, **kwargs):

		super(SurvivalPlotWidget, self).__init__(*args, **kwargs)
		self.cell_lines_btn.hide()
		self.class_selection_widget.hide()
		self.rescale_widget.hide()

	def switch_to_log(self):

		"""
		Switch threshold histogram to log scale. Auto adjust.
		"""

		if self.ax.get_yscale()=='linear':
			ymin,_ = self.ax.get_ylim()
			self.ax.set_ylim(max(ymin,0.01),1.05)
			self.ax.set_yscale('log')
			self.log_btn.setIcon(icon(MDI6.math_log,color=self.help_color))
		else:
			self.ax.set_yscale('linear')
			self.log_btn.setIcon(icon(MDI6.math_log,color="black"))
			#self.ax.set_ylim(0.01,1.05)

		#self.ax.autoscale()
		self.plot_widget.canvas.draw_idle()

	def initialize_axis(self):

		previous_legend = self.legend_visible
		is_log = self.ax.get_yscale()

		self.ax.clear()
		self.ax.plot([],[])

		# Labels
		self.ax.set_xlabel('time [min]')
		self.ax.set_ylabel('survival')

		# Spines
		self.ax.spines['top'].set_visible(False)
		self.ax.spines['right'].set_visible(False)

		# Lims
		self.ax.set_xlim(0*self.parent_window.FrameToMin,(self.df['FRAME'].max()+2)*self.parent_window.FrameToMin)
		if is_log=='log':
			ymin = 0.1
		else:
			ymin = 0.0
		self.ax.set_ylim(ymin,1.05)
	
		if is_log=='log':
			self.ax.set_yscale('log')
		if previous_legend:
			leg = self.ax.get_legend()
			if leg is not None:
				leg.set_visible(True)

	def plot_signals(self, id):

		for i in range(3):
			if self.plot_options[i].isChecked():
				self.plot_mode = self.radio_labels[i]

		colors = np.array([self.cmap(i / len(self.df_pos_info)) for i in range(len(self.df_pos_info))])
		well_color = [self.cmap(i / len(self.df_well_info)) for i in range(len(self.df_well_info))]

		if self.plot_mode=='position':
			self.initialize_axis()
			lines = self.df_pos_info.loc[self.df_pos_info['select'],'survival_fit'].values
			pos_labels = self.df_pos_info.loc[self.df_pos_info['select'],'pos_name'].values
			pos_indices = self.df_pos_info.loc[self.df_pos_info['select'],'pos_index'].values
			well_index = self.df_pos_info.loc[self.df_pos_info['select'],'well_index'].values
			for i in range(len(lines)):
				if len(self.parent_window.well_indices)<=1:
					self.plot_line(lines[i], colors[pos_indices[i]], pos_labels[i], ci_option=self.show_ci)
				else:
					self.plot_line(lines[i], well_color[well_index[i]], pos_labels[i], ci_option=self.show_ci)
			if self.legend_visible:
				self.ax.legend(ncols=3,fontsize='x-small')

		elif self.plot_mode=='well':
			self.initialize_axis()
			lines = self.df_well_info.loc[self.df_well_info['select'],'survival_fit'].values	
			well_index = self.df_well_info.loc[self.df_well_info['select'],'well_index'].values
			well_labels = self.df_well_info.loc[self.df_well_info['select'],'well_name'].values
			for i in range(len(lines)):	
				if len(self.parent_window.well_indices)<=1:
					self.plot_line(lines[i], 'k', well_labels[i], ci_option=self.show_ci, legend=True)
				else:
					self.plot_line(lines[i], well_color[well_index[i]], well_labels[i], ci_option=self.show_ci, legend=True)
			if self.legend_visible:
				self.ax.legend(ncols=2, fontsize='x-small')

		elif self.plot_mode=='both':
			
			self.initialize_axis()
			lines_pos = self.df_pos_info.loc[self.df_pos_info['select'],'survival_fit'].values
			lines_well = self.df_well_info.loc[self.df_well_info['select'],'survival_fit'].values	

			pos_indices = self.df_pos_info.loc[self.df_pos_info['select'],'pos_index'].values
			well_index_pos = self.df_pos_info.loc[self.df_pos_info['select'],'well_index'].values
			well_index = self.df_well_info.loc[self.df_well_info['select'],'well_index'].values
			well_labels = self.df_well_info.loc[self.df_well_info['select'],'well_name'].values
			pos_labels = self.df_pos_info.loc[self.df_pos_info['select'],'pos_name'].values

			for i in range(len(lines_pos)):
				if len(self.parent_window.well_indices)<=1:
					self.plot_line(lines_pos[i], colors[pos_indices[i]], pos_labels[i], ci_option=self.show_ci, legend=True)
				else:
					self.plot_line(lines_pos[i], well_color[well_index_pos[i]], None, ci_option=False)

			for i in range(len(lines_well)):
				if len(self.parent_window.well_indices)<=1:
					self.plot_line(lines_well[i], 'k', 'pool', ci_option=False, legend=True)
				else:
					self.plot_line(lines_well[i], well_color[well_index[i]], well_labels[i], ci_option=False, legend=True)
			if self.legend_visible:
				self.ax.legend(ncols=3,fontsize='x-small')

		self.plot_widget.canvas.draw()

	def plot_line(self, line, color, label, ci_option=True, legend=None, alpha_ci=0.5):
		
		# Plot a signal
		if line==line:
			line.plot_survival_function(ci_show=ci_option, ax=self.ax, legend=legend, color=color, label=label, xlabel='timeline [min]')