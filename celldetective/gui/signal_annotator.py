from PyQt5.QtWidgets import QMainWindow, QComboBox, QLabel, QRadioButton, QLineEdit, QApplication, QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QKeySequence
from celldetective.gui.gui_utils import center_window, QHSeperationLine, FilterChoice
from superqt import QLabeledDoubleSlider
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
import pandas as pd

class SignalAnnotator(QMainWindow):
	
	"""
	UI to set tracking parameters for bTrack.

	"""

	def __init__(self, parent=None):
		
		super().__init__()
		self.parent = parent
		self.setWindowTitle("Signal annotator")
		self.mode = self.parent.mode
		self.pos = self.parent.parent.pos
		self.exp_dir = self.parent.exp_dir
		self.n_signals = 3
		self.soft_path = get_software_location()
		self.selection = []
		if self.mode=="targets":
			self.instructions_path = self.exp_dir + "configs/signal_annotator_config_targets.json"
			self.trajectories_path = self.pos+'output/tables/trajectories_targets.csv'
		elif self.mode=="effectors":
			self.instructions_path = self.exp_dir + "configs/signal_annotator_config_effectors.json"
			self.trajectories_path = self.pos+'output/tables/trajectories_effectors.csv'

		self.screen_height = self.parent.parent.parent.screen_height
		self.screen_width = self.parent.parent.parent.screen_width

		center_window(self)

		self.locate_stack()
		self.load_annotator_config()
		self.locate_tracks()
		self.prepare_stack()

		self.generate_signal_choices()
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

		options_hbox.addWidget(self.event_btn, 33)
		options_hbox.addWidget(self.no_event_btn, 33)
		options_hbox.addWidget(self.else_btn, 33)
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
										self.time_of_interest_le]
		self.hide_annotation_buttons()
		#### End of annotation buttons


		# Cell signals
		self.left_panel.addWidget(self.cell_fcanvas)

		signal_choice_vbox = QVBoxLayout()
		signal_choice_vbox.setContentsMargins(30,30,30,50)
		for i in range(len(self.signal_choice_cb)):
			
			hlayout = QHBoxLayout()
			hlayout.addWidget(self.signal_choice_label[i], 20)
			hlayout.addWidget(self.signal_choice_cb[i], 80)
			signal_choice_vbox.addLayout(hlayout)

		self.left_panel.addLayout(signal_choice_vbox)

		self.save_btn = QPushButton('Save')
		self.save_btn.setStyleSheet(self.parent.parent.parent.button_style_sheet)
		#self.submit_btn.clicked.connect(self.write_instructions)
		self.left_panel.addWidget(self.save_btn)

		# Animation
		self.right_panel.addWidget(self.fcanvas)

		#self.populate_left_panel()
		#grid.addLayout(self.left_side, 0, 0, 1, 1)

		main_layout.addLayout(self.left_panel, 35)
		main_layout.addLayout(self.right_panel, 65)
		self.button_widget.adjustSize()

		self.setCentralWidget(self.button_widget)
		self.show()

		QApplication.processEvents()

	def cancel_selection(self):

		self.hide_annotation_buttons()
		self.correct_btn.setEnabled(False)
		self.correct_btn.setText('correct')
		self.cancel_btn.setEnabled(False)
		self.selection.pop(0)

		for k,(t,idx) in enumerate(zip(self.loc_t,self.loc_idx)):
			self.colors[t][idx,0] = self.previous_color[k][0]
			self.colors[t][idx,1] = self.previous_color[k][1]

	def hide_annotation_buttons(self):
		
		for a in self.annotation_btns_to_hide:
			a.hide()
		for b in [self.event_btn, self.no_event_btn, self.else_btn]:
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

		cclass = self.df_tracks.loc[self.df_tracks['TRACK_ID']==self.track_of_interest, 'class'].to_numpy()[0]
		t0 = self.df_tracks.loc[self.df_tracks['TRACK_ID']==self.track_of_interest, 't0'].to_numpy()[0]

		if cclass==0:
			self.event_btn.setChecked(True)
			self.time_of_interest_le.setText(str(t0))
		elif cclass==1:
			self.no_event_btn.setChecked(True)
		elif cclass==2:
			self.else_btn.setChecked(True)

		self.enable_time_of_interest()
		self.correct_btn.setText('submit')


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

	def locate_tracks(self):

		"""
		Locate the tracks.
		"""

		if not os.path.exists(self.trajectories_path):

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
			self.df_tracks = pd.read_csv(self.trajectories_path)
			self.df_tracks = self.df_tracks.sort_values(by=['TRACK_ID', 'FRAME'])
			if not 'status' in self.df_tracks.columns:
				if 't0' in self.df_tracks.columns and 'class' in self.df_tracks.columns:
					self.make_status_column()
				else:
					self.df_tracks['status'] = 0
					self.df_tracks['status_color'] = color_from_status(0)
					self.df_tracks['class_color'] = color_from_class(1)
			if not 'class' in self.df_tracks.columns:
				self.df_tracks['class'] = 1
			if not 't0' in self.df_tracks.columns:
				self.df_tracks['t0'] = -1

			self.df_tracks = self.df_tracks.dropna(subset=['POSITION_X', 'POSITION_Y'])
			self.df_tracks['x_anim'] = self.df_tracks['POSITION_X'] * self.fraction
			self.df_tracks['y_anim'] = self.df_tracks['POSITION_Y'] * self.fraction
			self.df_tracks['x_anim'] = self.df_tracks['x_anim'].astype(int)
			self.df_tracks['y_anim'] = self.df_tracks['y_anim'].astype(int)

			self.extract_scatter_from_trajectories()
			self.track_of_interest = self.df_tracks['TRACK_ID'].min()

			self.loc_t = []
			self.loc_idx = []
			for t in range(len(self.tracks)):
				indices = np.where(self.tracks[t]==self.track_of_interest)[0]
				if len(indices)>0:
					self.loc_t.append(t)
					self.loc_idx.append(indices[0])

			#self.loc_t, self.loc_idx = np.where(self.tracks==self.track_of_interest)


	def make_status_column(self):

		for tid, group in self.df_tracks.groupby('TRACK_ID'):
			
			indices = group.index
			t0 = group['t0'].to_numpy()[0]
			cclass = group['class'].to_numpy()[0]
			timeline = group['FRAME'].to_numpy()
			status = np.zeros_like(timeline)
			if t0 > 0:
				status[timeline>=t0] = 1.
			status_color = [color_from_status(s) for s in status]
			class_color = [color_from_class(cclass) for i in range(len(status))]

			self.df_tracks.loc[indices, 'status'] = status
			self.df_tracks.loc[indices, 'status_color'] = status_color
			self.df_tracks.loc[indices, 'class_color'] = class_color

		print(self.df_tracks)

	def generate_signal_choices(self):
		
		self.signal_choice_cb = [QComboBox() for i in range(self.n_signals)]
		self.signal_choice_label = [QLabel(f'signal {i+1}: ') for i in range(self.n_signals)]

		signals = list(self.df_tracks.columns)
		print(signals)
		to_remove = ['TRACK_ID', 'FRAME','x_anim','y_anim','t', 'state', 'generation', 'root', 'parent', 'class_id', 'class', 't0', 'POSITION_X', 'POSITION_Y']
		for c in to_remove:
			if c in signals:
				signals.remove(c)

		for i in range(len(self.signal_choice_cb)):
			self.signal_choice_cb[i].addItems(['--']+signals)
			self.signal_choice_cb[i].setCurrentIndex(i+1)
			self.signal_choice_cb[i].currentIndexChanged.connect(self.plot_signals)

	def plot_signals(self):
		
		yvalues = []
		for i in range(len(self.signal_choice_cb)):
			
			signal_choice = self.signal_choice_cb[i].currentText()
			self.lines[i].set_label(signal_choice)

			if signal_choice=="--":
				self.lines[i].set_xdata([])
				self.lines[i].set_ydata([])
			else:
				print(f'plot signal {signal_choice} for cell {self.track_of_interest}')
				xdata = self.df_tracks.loc[self.df_tracks['TRACK_ID']==self.track_of_interest, 'FRAME'].to_numpy()
				ydata = self.df_tracks.loc[self.df_tracks['TRACK_ID']==self.track_of_interest, signal_choice].to_numpy()
				yvalues.extend(ydata)
				self.lines[i].set_xdata(xdata)
				self.lines[i].set_ydata(ydata)
		
		self.configure_ylims()

		min_val,max_val = self.cell_ax.get_ylim()
		t0 = self.df_tracks.loc[self.df_tracks['TRACK_ID']==self.track_of_interest, 't0'].to_numpy()[0]
		self.line_dt.set_xdata([t0, t0])
		self.line_dt.set_ydata([min_val,max_val])

		self.cell_ax.legend()
		self.cell_fcanvas.canvas.draw()

	def extract_scatter_from_trajectories(self):

		self.positions = []
		self.colors = []
		self.tracks = []

		for t in np.arange(self.len_movie):

			self.positions.append(self.df_tracks.loc[self.df_tracks['FRAME']==t,['x_anim', 'y_anim']].to_numpy())
			self.colors.append(self.df_tracks.loc[self.df_tracks['FRAME']==t,['class_color', 'status_color']].to_numpy())
			self.tracks.append(self.df_tracks.loc[self.df_tracks['FRAME']==t, 'TRACK_ID'].to_numpy())


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
		else:
			self.rgb_mode = False
			self.percentile_mode = True
			self.target_channels = [[self.channel_names[0], 0.01, 99.99]]
			self.fraction = 0.25

	def prepare_stack(self):

		self.img_num_channels = _get_img_num_per_channel(self.channels, self.len_movie, self.nbr_channels)
		self.stack = []
		for ch in tqdm(self.target_channels, desc="channel"):
			target_ch_name = ch[0]
			if self.percentile_mode:
				normalize_kwargs = {"percentiles": (ch[1], ch[2]), "values": None}
			else:
				normalize_kwargs = {"values": (ch[1], ch[2]), "percentiles": None}
			chan = []
			indices = self.img_num_channels[self.channels[np.where(self.channel_names==target_ch_name)][0]]
			for t in tqdm(range(len(indices)),desc='frame'):
				
				f = load_frames(indices[t], self.stack_path, scale=self.fraction, normalize_input=True, normalize_kwargs=normalize_kwargs)
				chan.append(f[:,:,0])

			self.stack.append(chan)

		self.stack = np.array(self.stack)
		if self.rgb_mode:
			self.stack = np.moveaxis(self.stack, 0, -1)
		else:
			self.stack = self.stack[0]

		print(f'Load stack of shape: {self.stack.shape}.')

	
	def closeEvent(self, event):

		self.stop()
		result = QMessageBox.question(self,
					  "Confirm Exit...",
					  "Are you sure you want to exit ?",
					  QMessageBox.Yes| QMessageBox.No,
					  )
		del self.stack
		gc.collect()

	def looped_animation(self):

		"""
		Load an image. 

		"""

		self.speed = 1
		self.framedata = 0

		self.fig, self.ax = plt.subplots(tight_layout=True)
		self.fcanvas = FigureCanvas(self.fig, interactive=True)
		self.ax.clear()

		self.im = self.ax.imshow(self.stack[0], cmap='gray')
		self.status_scatter = self.ax.scatter(self.positions[0][:,0], self.positions[0][:,1], marker="x", c=self.colors[0][:,1], s=50, picker=True, pickradius=100)
		self.class_scatter = self.ax.scatter(self.positions[0][:,0], self.positions[0][:,1], marker='o', facecolors='none',edgecolors=self.colors[0][:,0], s=200)
		
		self.ax.set_xticks([])
		self.ax.set_yticks([])
		self.ax.set_aspect('equal')

		self.fig.set_facecolor('none')  # or 'None'
		self.fig.canvas.setStyleSheet("background-color: black;")

		self.anim = FuncAnimation(
							   self.fig, 
							   self.draw_frame, 
							   frames = self.len_movie, # better would be to cast np.arange(len(movie)) in case frame column is incomplete
							   interval = self.speed, # in ms
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

		self.lines = [self.cell_ax.plot([np.linspace(0,self.len_movie-1,self.len_movie)],[np.zeros((self.len_movie))])[0] for i in range(len(self.signal_choice_cb))]
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
		if len(ind)>1:
			# More than one point in vicinity
			datax,datay = [self.positions[self.framedata][i,0] for i in ind],[self.positions[self.framedata][i,1] for i in ind]
			msx, msy = event.mouseevent.xdata, event.mouseevent.ydata
			dist = np.sqrt((np.array(datax)-msx)**2+(np.array(datay)-msy)**2)
			ind = [ind[np.argmin(dist)]]
		

		if ind and (len(self.selection))==0:
			ind = ind[0]
			self.selection.append(ind)
			self.correct_btn.setEnabled(True)
			self.cancel_btn.setEnabled(True)

			self.track_of_interest = self.tracks[self.framedata][ind]
			print(f'You selected track {self.track_of_interest}.')
			self.give_cell_information()
			self.plot_signals()

			self.loc_t = []
			self.loc_idx = []
			for t in range(len(self.tracks)):
				indices = np.where(self.tracks[t]==self.track_of_interest)[0]
				if len(indices)>0:
					self.loc_t.append(t)
					self.loc_idx.append(indices[0])


			self.previous_color = []
			for t,idx in zip(self.loc_t,self.loc_idx):
				self.previous_color.append(self.colors[t][idx].copy())
				self.colors[t][idx] = 'lime'

		elif ind and len(self.selection)==1:
			self.cancel_btn.click()
		else:
			pass
			
	def configure_ylims(self):

		min_values = []
		max_values = []
		for i in range(len(self.signal_choice_cb)):
			signal = self.signal_choice_cb[i].currentText()
			if signal=='--':
				continue
			else:
				maxx = np.nanpercentile(self.df_tracks.loc[:,signal].to_numpy().flatten(),99)
				minn = np.nanpercentile(self.df_tracks.loc[:,signal].to_numpy().flatten(),1)
				min_values.append(minn)
				max_values.append(maxx)

		if len(min_values)>0:
			self.cell_ax.set_ylim(np.amin(min_values), np.amax(max_values))

	def draw_frame(self, framedata):
		
		"""
		Update plot elements at each timestep of the loop.
		"""

		self.framedata = framedata
		self.im.set_array(self.stack[self.framedata])
		self.status_scatter.set_offsets(self.positions[self.framedata])
		self.status_scatter.set_color(self.colors[self.framedata][:,1])

		self.class_scatter.set_offsets(self.positions[self.framedata])
		self.class_scatter.set_edgecolor(self.colors[self.framedata][:,0])

		return (self.im,self.status_scatter,self.class_scatter,)


	def stop(self):
		# # On stop we disconnect all of our events.
		self.anim.event_source.stop()

	def start(self):
		'''
		Starts interactive animation. Adds the draw frame command to the GUI
		handler, calls show to start the event loop.
		'''
		self.anim.event_source.start()

	def give_cell_information(self):

		cell_selected = f"cell: {self.track_of_interest}\n"
		cell_class = f"class: {df_tracks.loc[df_tracks['TRACK_ID']==self.track_of_interest, 'class'].to_numpy()[0]}\n"
		cell_time = f"time of interest: {df_tracks.loc[df_tracks['TRACK_ID']==self.track_of_interest, 't0'].to_numpy()[0]}\n"
		self.cell_info.setText(cell_selected+cell_class+cell_time)


