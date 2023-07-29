from PyQt5.QtWidgets import QMainWindow, QComboBox, QLabel, QRadioButton, QLineEdit, QApplication, QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtCore import Qt, QSize
from celldetective.gui.gui_utils import center_window, QHSeperationLine
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
		self.soft_path = get_software_location()
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

		self.populate_widget()
		self.looped_animation()

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
		self.right_panel = QVBoxLayout()

		self.submit_btn = QPushButton('Save')
		self.submit_btn.setStyleSheet(self.parent.parent.parent.button_style_sheet)
		#self.submit_btn.clicked.connect(self.write_instructions)
		self.left_panel.addWidget(self.submit_btn)

		#self.populate_left_panel()
		#grid.addLayout(self.left_side, 0, 0, 1, 1)

		main_layout.addLayout(self.left_panel, 25)
		main_layout.addLayout(self.right_panel, 75)
		self.button_widget.adjustSize()

		self.setCentralWidget(self.button_widget)
		self.show()

		QApplication.processEvents()

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


	def extract_scatter_from_trajectories(self):

		self.positions = []
		self.colors = []
		# for k,group in self.df_tracks.groupby('FRAME'):
		# 	self.positions.append(group[['x_anim', 'y_anim']].to_numpy())
		# 	self.colors.append(group[['class_color', 'status_color']].to_numpy())
		for t in np.arange(0,self.df_tracks['FRAME'].max()):
			self.positions.append(self.df_tracks.loc[self.df_tracks['FRAME']==t,['x_anim', 'y_anim']].to_numpy())
			self.colors.append(self.df_tracks.loc[self.df_tracks['FRAME']==t,['class_color', 'status_color']].to_numpy())			


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
		self.iter_val = 0

		self.fig, self.ax = plt.subplots()
		self.fcanvas = FigureCanvas(self.fig, interactive=True)
		self.ax.clear()

		self.im = self.ax.imshow(self.stack[0], cmap='gray')
		self.status_scatter = self.ax.scatter(self.positions[0][:,0], self.positions[0][:,1], marker="x", c=self.colors[0][:,1], s=50, picker=True, pickradius=5)
		self.class_scatter = self.ax.scatter(self.positions[0][:,0], self.positions[0][:,1], marker='o', facecolors='none',edgecolors=self.colors[0][:,0], s=200)
		
		self.ax.set_xticks([])
		self.ax.set_yticks([])
		self.ax.set_aspect('equal')

		self.fig.set_facecolor('none')  # or 'None'
		self.fig.canvas.setStyleSheet("background-color: black;")

		self.anim = FuncAnimation(
							   self.fig, 
							   self.draw_frame, 
							   frames = self.len_movie,
							   interval = self.speed, # in ms
							   blit=True,
							   )

		self.fig.canvas.mpl_connect('pick_event', self.on_scatter_pick)
		self.fcanvas.canvas.draw()

		self.right_panel.addWidget(self.fcanvas)

	def on_scatter_pick(self, event):
		
		ind = event.ind
		print(ind)
		#print('onpick3 scatter:', ind, x[ind], y[ind])		


	def draw_frame(self, framedata):
		
		"""
		Update plot elements at each timestep of the loop.
		"""

		self.iter_val = framedata
		self.im.set_array(self.stack[framedata])
		self.status_scatter.set_offsets(self.positions[framedata])
		self.status_scatter.set_color(self.colors[framedata][:,1])

		self.class_scatter.set_offsets(self.positions[framedata])
		self.class_scatter.set_edgecolor(self.colors[framedata][:,0])

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

