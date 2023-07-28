#!/usr/bin/python

import sys

import os

from adccfactory.core.utils import *
from adccfactory.core.plot import *
from adccfactory.core.visual import *
from adccfactory.core.core import *
from adccfactory.core.parse import *
from matplotlib.patches import Rectangle
import sys
import random
import time
from screeninfo import get_monitors
import shutil

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import numpy as np
import pandas as pd

from glob import glob
from natsort import natsorted
from tqdm import tqdm
from tifffile import imread

from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QApplication, QAction, QMenu, QWidget, QGridLayout, QLabel, QPushButton, QRadioButton, QLineEdit
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtGui import QIcon, QImage, QDoubleValidator, QKeySequence
from datetime import datetime
from pathlib import Path, PurePath
from adccfactory.gui.styles import Styles
from adccfactory.gui.gui_utils import QIcon_from_svg
#from superqt.fonticon import icon
#from fonticon_mdi6 import MDI6
for m in get_monitors():
	res_w = int(m.width*0.75)
	res_h = int(m.height*0.75)


table_path = str(sys.argv[1]) #"/media/limozin/HDD/Storage_Bea_stuff/JanExp/W1/100/"
pos = os.path.split(table_path)[0]
#print(pos,table_path)

# MOVIE PARAMETERS
pos_folder = Path(pos).parent.parent
expfolder = pos_folder.parent.parent

#print(parent1, expfolder)
config = PurePath(expfolder,Path("config.ini"))
#print("config path = ",config)

PxToUm = float(ConfigSectionMap(config,"MovieSettings")["pxtoum"])
FrameToMin = float(ConfigSectionMap(config,"MovieSettings")["frametomin"])
len_movie = int(ConfigSectionMap(config,"MovieSettings")["len_movie"])
shape_x = int(ConfigSectionMap(config,"MovieSettings")["shape_x"])
shape_y = int(ConfigSectionMap(config,"MovieSettings")["shape_y"])
movie_prefix = ConfigSectionMap(config,"MovieSettings")["movie_prefix"]

# blue_channel = int(ConfigSectionMap(config,"MovieSettings")["live_nuclei_channel"])
# red_channel = int(ConfigSectionMap(config,"MovieSettings")["dead_nuclei_channel"])
# nbr_channels = 3

# try:
# 	green_channel = int(ConfigSectionMap(config,"MovieSettings")["effector_fluo_channel"])
# 	nbr_channels = 4
# except:
# 	green_channel = None
# 	print("No green channel detected...")

fraction = int(ConfigSectionMap(config,"Display")["fraction"])
time_dilation = int(ConfigSectionMap(config,"BinningParameters")["time_dilation"])
len_movie = len_movie//time_dilation

# DISPLAY PARAMETERS
try:
	blue_percentiles = [float(s) for s in ConfigSectionMap(config,"Display")["blue_percentiles"].split(",")]
except:
	blue_percentiles = None

try:
	red_percentiles = [float(s) for s in ConfigSectionMap(config,"Display")["red_percentiles"].split(",")]
except:
	red_percentiles = None

try:
	stack_path = glob(str(pos_folder)+f"/movie/{movie_prefix}*.tif")[0]
except IndexError as e:
	print(e)
	os.abort()

df0 = pd.read_csv(table_path)
df = df0.copy()


movie_name = os.path.split(stack_path)[-1]

# Get available channels
channel_names, channel_indices = extract_experiment_channels(config)
nbr_channels = len(channel_names)
channel_names = np.array(channel_names)
channel_indices = np.array(channel_indices)

if "live_nuclei_channel" in channel_names:
	blue_channel = channel_indices[np.where(channel_names=="live_nuclei_channel")[0]][0]
else:
	blue_channel = None

if "dead_nuclei_channel" in channel_names:
	red_channel = channel_indices[np.where(channel_names=="dead_nuclei_channel")[0]][0]
else:
	red_channel = None

if "effector_fluo_channel" in channel_names:
	green_channel = channel_indices[np.where(channel_names=="effector_fluo_channel")[0]][0]
elif "fluo_channel_1" in channel_names:
	green_channel = channel_indices[np.where(channel_names=="fluo_channel_1")[0]][0]
else:
	green_channel = None

if blue_channel is not None:
	blue_indices = np.arange(len_movie*nbr_channels)[blue_channel::nbr_channels]
if red_channel is not None:
	red_indices = np.arange(len_movie*nbr_channels)[red_channel::nbr_channels]
if green_channel is not None:
	green_indices = np.arange(len_movie*nbr_channels)[green_channel::nbr_channels]

# Load stack (efficient way)
stack = np.zeros((3, len_movie, int(shape_x/fraction), int(shape_y/fraction)), dtype=np.float32)
if red_channel is not None:
	stack[0] = np.array([zoom(load_frame_i(i, stack_path), [1/fraction, 1/fraction], order=0) for i in tqdm(red_indices,desc="red channel")])
if green_channel is not None:
	stack[1] = np.array([zoom(load_frame_i(i, stack_path), [1/fraction, 1/fraction], order=0) for i in tqdm(green_indices,desc="green channel")])
if blue_channel is not None:
	stack[2] = np.array([zoom(load_frame_i(i, stack_path), [1/fraction, 1/fraction], order=0) for i in tqdm(blue_indices,desc="blue channel")])

stack = np.moveaxis(stack, 0, -1)
print(f"RGB stack reduced: {stack.shape}...")

stack = normalize_rgb(stack, blue_percentiles=blue_percentiles,red_percentiles=red_percentiles)

xscats = []
yscats = []
cscats0 = []
class0 = []

df.loc[df['CLASS_COLOR'].isna(), 'CLASS_COLOR'] = "yellow"
df.loc[df['STATUS_COLOR'].isna(), 'STATUS_COLOR'] = "yellow"

for k in range(len(stack)):
	dft = df[(df["T"]==k)]
	cscats0.append(dft.STATUS_COLOR.to_numpy())
	class0.append(dft.CLASS_COLOR.to_numpy())
	xscats.append(dft.X.to_numpy()//fraction)
	yscats.append(dft.Y.to_numpy()//fraction)

cscats = np.copy(cscats0)
cclass = np.copy(class0)

x_scat_inside = []
y_scat_inside = []



# class RetrainWindow(QMainWindow):
# 	def __init__(self):
# 		super().__init__()

# 		w = QWidget()
# 		grid = QGridLayout(w)
# 		self.setWindowTitle(f"Retrain model")

# 		grid.addWidget(QLabel("Dataset folder:"), 0, 0, 1, 3)
# 		self.dataFolder = QLineEdit()
# 		self.dataFolder.setAlignment(Qt.AlignLeft)	
# 		self.dataFolder.setEnabled(True)
# 		self.dataFolder.setText(f"{home_dir}ADCCFactory_2.0/src/datasets/cell_signals")
# 		grid.addWidget(self.dataFolder, 1, 0, 1, 2)

# 		self.browse_button = QPushButton("Browse...")
# 		self.browse_button.clicked.connect(self.browse_dataset_folder)
# 		grid.addWidget(self.browse_button, 1, 2, 1, 1)

# 		grid.addWidget(QLabel("New model name:"), 2, 0, 1, 3)
# 		self.ModelName = QLineEdit()
# 		self.ModelName.setAlignment(Qt.AlignLeft)	
# 		self.ModelName.setEnabled(True)
# 		self.ModelName.setText(f"Untitled_model_{datetime.today().strftime('%Y-%m-%d')}")
# 		grid.addWidget(self.ModelName, 3, 0, 1, 2)

# 		grid.addWidget(QLabel("Number of epochs:"), 4, 0, 1, 3)
# 		self.NbrEpochs = QLineEdit()
# 		self.NbrEpochs.setAlignment(Qt.AlignLeft)	
# 		self.NbrEpochs.setEnabled(True)
# 		self.NbrEpochs.setText(f"100")
# 		grid.addWidget(self.NbrEpochs, 5, 0, 1, 2)

# 		self.confirm_button = QPushButton("Submit")
# 		self.confirm_button.clicked.connect(self.retrain)
# 		grid.addWidget(self.confirm_button, 6, 1, 1, 1)

# 		self.setCentralWidget(w)

# 	def browse_dataset_folder(self):
# 		self.newDataFolder = str(QFileDialog.getExistingDirectory(self, 'Select directory'))
# 		self.dataFolder.setText(self.newDataFolder)

# 	def retrain(self):
# 		dataset_dir = self.dataFolder.text()
# 		output_dir = f"{home_dir}ADCCFactory_2.0/src/models/combined/"
# 		model_name = self.ModelName.text()
# 		model_signal_length = 128
# 		nbr_epochs = int(self.NbrEpochs.text())

# 		to_freeze = [self.dataFolder, self.browse_button, self.ModelName, self.confirm_button, self.NbrEpochs]
# 		for q in to_freeze:
# 			q.setEnabled(False)
# 			q.repaint()

# 		train_cell_model(dataset_dir, output_dir, model_name, model_signal_length, nbr_epochs=nbr_epochs)

# 		for q in to_freeze:
# 			q.setEnabled(True)
# 			q.repaint()

class CustomFigCanvas(FigureCanvas, FuncAnimation):

	def __init__(self):

		# The data
		self.len_movie = len_movie
		self.shape_x = shape_x//fraction
		self.shape_y = shape_y//fraction
		self.title_label = QLabel()
		self.iter_val = 0
		self.speed = 1
		self.exclusion_mode = False

		self.rect_origin = None
		self.rect_h = None
		self.rect_w = None

		#print(self.speed)
		
		# The window
		self.fig = Figure(tight_layout=True)
		self.ax = self.fig.add_subplot(111)

		# ax settings
		self.im = self.ax.imshow(stack[0]) #, aspect='auto'
		self.scat = self.ax.scatter(xscats[0], yscats[0], c=cscats[0], marker='x', facecolors='r',s=50)
		self.scat2 = self.ax.scatter(xscats[0],yscats[0],facecolors='none',edgecolors=cclass[0],s=200)
		#self.titi = self.ax.text(0.5,1.01, "", fontsize=30, bbox={'facecolor':'w', 'alpha':1, 'pad':5},transform=self.ax.transAxes, ha="center")
		self.fig.patch.set_facecolor('black')		
		
		for ax in [self.ax]:
			ax.set_aspect("equal")
			ax.set_xticks([])
			ax.set_yticks([])

		#self.fig.tight_layout()
		self._canvas = FigureCanvas.__init__(self, self.fig)
		#TimedAnimation.__init__(self, self.fig, interval = 100, blit = True)
		self._toolbar = NavigationToolbar(self,self._canvas) 

		#self.patches = [self.im] + [self.scat] + [self.scat2]

		self.anim = FuncAnimation(
							   self.fig, 
							   self._draw_frame, 
							   frames = len_movie,
							   interval = self.speed, # in ms
							   blit=True,
							   )

		#print(dir(self.anim.event_source))
		#print(dir(self.fig.suptitle()))


	def new_frame_seq(self):
		# Use the generating function to generate a new frame sequence
		return self._iter_gen()

	def activate_exclusion_mode(self):
		self.exclusion_mode = True
		self.scat_exclusion = self.ax.scatter(x_scat_inside[0],y_scat_inside[0],marker="o",facecolors='yellow',s=500,zorder=10)

	def disable_exclusion_mode(self):
		self.exclusion_mode = False

	def _draw_frame(self, framedata):

		self.iter_val = framedata
		self.title_label.setText(f"Frame: {str(framedata).zfill(3)}")
		self.im.set_array(stack[framedata])
		self.scat.set_offsets(np.swapaxes([xscats[framedata],yscats[framedata]],0,1))
		self.scat.set_color(cscats[framedata])
		
		self.scat2.set_offsets(np.swapaxes([xscats[framedata],yscats[framedata]],0,1))
		self.scat2.set_edgecolor(cclass[framedata])

		if self.exclusion_mode:
			self.scat_exclusion.set_offsets(np.swapaxes([x_scat_inside[framedata],y_scat_inside[framedata]],0,1))
			return(self.im,self.scat,self.scat2,self.scat_exclusion)
		else:
			return(self.im,self.scat,self.scat2)


	def set_last_frame(self, framedata):
		
		self.anim._drawn_artists = self._draw_frame(len(stack)-1)
		self.anim._drawn_artists = sorted(self.anim._drawn_artists, key=lambda x: x.get_zorder())
		for a in self.anim._drawn_artists:
			a.set_visible(True)

		self.fig.canvas.draw()
		self.anim.event_source.stop()


	def _init_draw(self):
		self.im.set_array(stack[0])
		self.scat.set_offsets(np.swapaxes([xscats[0],yscats[0]],0,1))
		self.scat.set_color(cscats[0])
		
		self.scat2.set_offsets(np.swapaxes([xscats[0],yscats[0]],0,1))
		self.scat2.set_edgecolor(cclass[0])

		#self.titi = self.ax.set_title(f"Time: 0 s",fontsize=20)


	def start(self):
		'''
		Starts interactive animation. Adds the draw frame command to the GUI
		handler, calls show to start the event loop.
		'''
		self.anim.event_source.start()


	def stop(self):
		# # On stop we disconnect all of our events.
		self.anim.event_source.stop()

class MplCanvas(FigureCanvas):

	def __init__(self, parent=None, width=5, height=4, dpi=100):

		self.fig, self.ax = plt.subplots()
		super(MplCanvas, self).__init__(self.fig)

		#self.fig = Figure()
		self.parent = parent

		self.fig.set_facecolor('none')  # or 'None'
		self.fig.canvas.setStyleSheet("background-color: transparent;")

		self.signal1 = "BLUE_INTENSITY"
		self.signal2 = "RED_INTENSITY"

		#self.ax = self.fig.add_subplot(111)

		spacing = 0.5 # This can be your user specified spacing. 
		minorLocator = MultipleLocator(1)
		self.ax.xaxis.set_minor_locator(minorLocator)
		self.ax.xaxis.set_major_locator(MultipleLocator(5))
		self.ax.grid(which = 'major')
		self.ax.set_xlabel("Frame")
		self.ax.set_ylabel("Signals")
		self.line_blue, = self.ax.plot(np.linspace(0,len_movie-1,len_movie),np.zeros((len_movie)),c="tab:blue",label=self.signal1)
		self.line_red, = self.ax.plot(np.linspace(0,len_movie-1,len_movie),np.zeros((len_movie)),c="tab:red",label=self.signal2)
		self.configure_ylims()

		self.line_dt, = self.ax.plot([-1,-1],[0,self.max_signal],c="tab:purple",linestyle="--")
		self.ax.set_xlim(0,len_movie)
		self.ax.legend()


		#self.fig.tight_layout()

	def get_signal_1(self):
		self.signal1 = self.parent.signal_1_option.currentText()

	def get_signal_2(self):
		self.signal2 = self.parent.signal_2_option.currentText()
		if self.signal2=="--":
			self.signal2 = None

	def configure_ylims(self):
		if self.signal2 is not None:
			self.max_signal = np.nanpercentile(np.array([df[self.signal1].to_numpy(),df[self.signal2].to_numpy()]).flatten(),98)
			self.min_signal = np.nanpercentile(np.array([df[self.signal1].to_numpy(),df[self.signal2].to_numpy()]).flatten(),1)
		else:
			self.max_signal = np.nanpercentile(df[self.signal1].to_numpy(),98)
			self.min_signal = np.nanpercentile(df[self.signal1].to_numpy(),10)	
		self.ax.set_ylim(self.min_signal,self.max_signal)
		self.draw()

	def switch_axes(self):

		self.get_signal_1()
		self.get_signal_2()
		self.configure_ylims()
		self.plot_cell()

	def plot_cell(self):

		self.new_track = self.parent.track_selected
		self.line_blue.set_ydata(df.loc[df["TID"]==self.new_track, self.signal1])
		self.line_blue.set_xdata(df.loc[df["TID"]==self.new_track, "T"])
		self.line_blue.set_label(self.signal1)

		if self.signal2 is not None:
			self.line_red.set_ydata(df.loc[df["TID"]==self.new_track, self.signal2])
			self.line_red.set_xdata(df.loc[df["TID"]==self.new_track, "T"])
			self.line_red.set_label(self.signal2)
		else:
			self.line_red.set_ydata([])
			self.line_red.set_xdata([])
			self.line_red.set_label("")

		t0 = df.loc[df["TID"]==self.new_track, "T0"].to_numpy()[0]
		self.line_dt.set_xdata([t0,t0])
		self.line_dt.set_ydata([0,self.max_signal])

		self.ax.legend()

		#self.line_dt.set_xdata([t0,t0])
		self.draw()

class Window(QMainWindow):
	def __init__(self):
		super().__init__()

		self.setWindowTitle(f"User check: {movie_name}")

		icon = QImage('icon.png')

		self.setWindowIcon(QIcon("icon.png"))
		self.Styles = Styles()
		self.init_styles()
		
		self.center_coords = []
		self.selected_x = 0
		self.selected_y = 0
		self.tc_index = 0
		self.track_selected = 0
		
		self.new_cell_class = -1
		self.new_cell_death_time = -1
		self.new_cell_color = "tab:cyan"

		self._createActions()
		self._createMenuBar()

		self.setGeometry(0,0,res_w,res_h)

		default_time = df[(df["T"]==0)]["T0"].to_numpy()[self.tc_index]
		track_selected = df[(df["T"]==0)]["TID"].to_numpy()[self.tc_index]
		classification_label = df[(df["TID"]==track_selected)]["CLASS"].to_numpy()[0]
		mean_x = np.mean(df[(df["TID"]==track_selected)]["X"].to_numpy())
		mean_y = np.mean(df[(df["TID"]==track_selected)]["Y"].to_numpy())

		w = QWidget()
		self.grid = QGridLayout(w)
		self.cell_info = QLabel("\n \n \n")
		self.cell_info.setText(f"Cell selected: track {track_selected}\nClassification = {classification_label}\nEstimated death time = {round(default_time,2)}")
		self.cell_info.setFixedSize(QSize(300, 50))
		self.grid.addWidget(self.cell_info,0,0,1,4,alignment=Qt.AlignVCenter)
		
		frameGm = self.frameGeometry()
		screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
		centerPoint = QApplication.desktop().screenGeometry(screen).center()
		frameGm.moveCenter(centerPoint)
		self.move(frameGm.topLeft())

		cell_action_grid = QHBoxLayout()
		self.change_btn = QPushButton("Correct")
		self.change_btn.setStyleSheet(self.button_style_sheet)
		self.change_btn.setIcon(QIcon_from_svg(abs_path+f"/icons/change.svg", color='white'))
		self.change_btn.clicked.connect(self.buttonPress_change_class)
		cell_action_grid.addWidget(self.change_btn)	

		self.cancel_btn = QPushButton("Cancel")
		self.cancel_btn.setStyleSheet(self.button_style_sheet_2)
		self.cancel_btn.clicked.connect(self.cancel_cell_selection)
		cell_action_grid.addWidget(self.cancel_btn)

		cell_action_grid.setContentsMargins(60,30,60,30)

		self.grid.addLayout(cell_action_grid, 3, 0, 1, 4)
		
		grid_color_btn = QHBoxLayout()

		self.red_btn = QRadioButton("lysis")
		self.red_btn.setStyleSheet("""
			QRadioButton {
				color: #d62728; 
				font-weight: bold; 
				border : 4px solid #d62728;
				border-radius : 15px;
				padding: 10px;
						}
		""")
		self.red_btn.toggled.connect(self.buttonPress_red)
		self.dt_label = QLabel("Death time: ")
		
		self.blue_btn = QRadioButton("no lysis")
		self.blue_btn.setStyleSheet("""
			QRadioButton {
				color: #1f77b4; 
				font-weight: bold; 
				border : 4px solid #1f77b4;
				border-radius : 15px;
				padding: 10px;
						}
		""")
		self.blue_btn.setChecked(True)
		self.blue_btn.toggled.connect(self.buttonPress_blue)
		
		self.yellow_btn = QRadioButton("left-censored")
		self.yellow_btn.setStyleSheet("""
			QRadioButton {
				color: #bcbd22; 
				font-weight: bold; 
				border : 4px solid #bcbd22;
				border-radius : 15px;
				padding: 10px;
						}
		""")
		self.yellow_btn.toggled.connect(self.buttonPress_yellow)

		self.delete_btn = QRadioButton("bad")
		self.delete_btn.setStyleSheet("color: black; font-weight: bold;")
		self.delete_btn.toggled.connect(self.buttonPress_delete)
		
		grid_color_btn.addWidget(self.red_btn, 33)
		self.grid.addWidget(self.dt_label,2,0,1,1)
		grid_color_btn.addWidget(self.blue_btn,33)
		grid_color_btn.addWidget(self.yellow_btn,33)
		grid_color_btn.setContentsMargins(0,30,80,0)

		self.grid.addLayout(grid_color_btn, 1, 1, 1, 3)
		self.grid.addWidget(self.delete_btn,2,3,1,1)
		
		self.red_btn.hide()
		self.blue_btn.hide()
		self.yellow_btn.hide()
		self.delete_btn.hide()
		self.dt_label.hide()
		#self.change_btn.hide()
		self.change_btn.setEnabled(False)
		self.cancel_btn.setEnabled(False)

		cell_plot_grid = QHBoxLayout()
		cell_plot_grid.setContentsMargins(40,0,40,0)
		self.cell_plot = MplCanvas(self)
		cell_plot_grid.addWidget(self.cell_plot)
		self.grid.addLayout(cell_plot_grid, 5,0,1,4)

		# Set up signal 1 option

		signal_box = QHBoxLayout()
		signal_box.setContentsMargins(30,30,30,5)

		self.signal_1_option = QComboBox()
		self.signal_1_option.addItems(list(df.columns))
		# index = self.signal_1_option.findText(self.intensity_features[0], Qt.MatchFixedString)
		# if index >= 0:
		# 	self.signal_1_option.setCurrentIndex(index)
		self.signal_1_option.currentTextChanged.connect(self.set_signal_1)
		signal_box.addWidget(QLabel("Signal 1: "), 10,alignment=Qt.AlignRight)
		signal_box.addWidget(self.signal_1_option,  60, alignment=Qt.AlignLeft)

		self.grid.addLayout(signal_box, 6,1,1,1)


		signal_box_2 = QHBoxLayout()
		signal_box_2.setContentsMargins(30,5,30,30)

		# Set up signal 2 option
		self.signal_2_option = QComboBox()
		self.signal_2_option.addItems(["--"]+list(df.columns))
		# index = self.signal_1_option.findText(self.intensity_features[0], Qt.MatchFixedString)
		# if index >= 0:
		# 	self.signal_1_option.setCurrentIndex(index)
		self.signal_2_option.currentTextChanged.connect(self.set_signal_2)
		signal_box_2.addWidget(QLabel("Signal 2: "), 10, alignment=Qt.AlignRight)
		signal_box_2.addWidget(self.signal_2_option,  60, alignment=Qt.AlignLeft)

		self.grid.addLayout(signal_box_2, 7,1,1,1)

		grid_bottom_buttons = QHBoxLayout()
		grid_bottom_buttons.setContentsMargins(60,30,60,30)

		self.exclude_btn = QPushButton("Exclude ROI")
		self.exclude_btn.setStyleSheet(self.button_style_sheet_2)
		grid_bottom_buttons.addWidget(self.exclude_btn)
		self.exclude_btn.clicked.connect(self.switch_to_exclude_mode)

		self.save_btn = QPushButton("Save")
		self.save_btn.setStyleSheet(self.button_style_sheet)
		self.save_btn.setIcon(QIcon_from_svg(abs_path+f"/icons/save.svg", color='white'))
		grid_bottom_buttons.addWidget(self.save_btn)
		self.save_btn.clicked.connect(self.save_csv)
		self.grid.addLayout(grid_bottom_buttons, 10,0,1,4)

	
		self.timer = QTimer()
		self.timer.setInterval(100)
		self.timer.timeout.connect(self.update_cell_plot)
		self.timer.start()

		self.e1 = QLineEdit()
		self.e1.setValidator(QDoubleValidator().setDecimals(2))
		self.e1.setFixedWidth(100)
		self.e1.setMaxLength(5)
		self.e1.setAlignment(Qt.AlignLeft)	
		self.grid.addWidget(self.e1,2,1,1,1)
		self.e1.setEnabled(False)
		self.e1.hide()
		
		self.myFigCanvas = CustomFigCanvas()
		self.grid.addWidget(self.myFigCanvas,1,4,10,12)
		

		self.grid.addWidget(self.myFigCanvas.title_label,0,10,1,1, alignment = Qt.AlignRight)

		self.grid.addWidget(self.myFigCanvas.toolbar,11,4)
		self.cid = self.myFigCanvas.mpl_connect('button_press_event', self.onclick)

		self.stop_btn = QPushButton("stop")
		self.stop_btn.clicked.connect(self.stop_anim)
		self.stop_btn.setFixedSize(QSize(80, 40))
		self.grid.addWidget(self.stop_btn,0,4,1,1,alignment=Qt.AlignLeft)

		self.start_btn = QPushButton("start")
		self.start_btn.clicked.connect(self.start_anim)
		self.start_btn.setFixedSize(QSize(80, 40))
		self.grid.addWidget(self.start_btn,0,4,1,1,alignment=Qt.AlignLeft)
		self.start_btn.hide()

		self.set_last_btn = QPushButton("last frame")
		#self.set_last_btn.setIcon(icon(MDI6.page_last,color="black"))
		#self.set_last_btn.setStyleSheet(self.button_select_all)

		self.set_last_btn.clicked.connect(self.set_last_frame)
		self.set_last_btn.setShortcut(QKeySequence("l"))
		self.set_last_btn.setFixedSize(QSize(100, 40))
		self.grid.addWidget(self.set_last_btn,0,5,1,1,alignment=Qt.AlignLeft)
		
		self.setCentralWidget(w)
		self.show()

	def set_signal_1(self):
		self.cell_plot.switch_axes()

	def set_signal_2(self):
		self.cell_plot.switch_axes()

	def init_styles(self):

		self.qtab_style = self.Styles.qtab_style
		self.button_style_sheet = self.Styles.button_style_sheet
		self.button_style_sheet_2 = self.Styles.button_style_sheet_2
		self.button_style_sheet_2_not_done = self.Styles.button_style_sheet_2_not_done
		self.button_style_sheet_3 = self.Styles.button_style_sheet_3
		self.button_select_all = self.Styles.button_select_all


	def switch_to_exclude_mode(self):
		self.myFigCanvas.mpl_disconnect(self.cid)
		self.myFigCanvas.press = None

		self.cidpress = self.myFigCanvas.mpl_connect('button_press_event', self.rect_on_press)
		self.cidrelease = self.myFigCanvas.mpl_connect('button_release_event', self.rect_on_release)
		self.cidmotion = self.myFigCanvas.mpl_connect('motion_notify_event', self.rect_on_motion)


	def rect_on_press(self,event):
		self.myFigCanvas.rect_origin = (event.xdata,event.ydata)
		print(self.myFigCanvas.rect_origin)
		return(self.myFigCanvas.rect_origin)

	def rect_on_release(self,event):
		global x_scat_inside
		global y_scat_inside

		self.myFigCanvas.bottom_right = (event.xdata,event.ydata)
		try:
			self.myFigCanvas.rect_h = np.abs(self.myFigCanvas.rect_origin[0] - event.xdata)
			self.myFigCanvas.rect_w = np.abs(self.myFigCanvas.rect_origin[1] - event.ydata)
		except:
			print("Please select a rectangle within the image.")
			return(None)

		xscats_temp = np.array(np.copy(xscats))
		yscats_temp = np.array(np.copy(yscats))

		indices_inside = np.where((xscats_temp[0] > self.myFigCanvas.rect_origin[0])*
			(xscats_temp[0] < event.xdata)*
			(yscats_temp[0] > self.myFigCanvas.rect_origin[1])*
			(yscats_temp[0] < event.ydata))[0]

		if len(indices_inside)>0:
			x_scat_inside = xscats_temp[:,indices_inside]
			y_scat_inside = yscats_temp[:,indices_inside]

			self.myFigCanvas.activate_exclusion_mode()

			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setText("Do you want to apply the selection?")
			msgBox.setWindowTitle("Info")
			msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.No | QMessageBox.Cancel)

			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Cancel:
				self.myFigCanvas.disable_exclusion_mode()
				self.myFigCanvas.mpl_disconnect(self.cidpress)
				self.myFigCanvas.mpl_disconnect(self.cidrelease)
				self.myFigCanvas.mpl_disconnect(self.cidmotion)
				self.cid = self.myFigCanvas.mpl_connect('button_press_event', self.onclick)

			elif returnValue == QMessageBox.Ok:
				self.myFigCanvas.disable_exclusion_mode()

				#Write cells as yellow
				for cell in indices_inside:

					self.track_selected = df[(df["T"]==0)]["TID"].to_numpy()[cell]

					for k in range(0,len(cscats)):
						cscats[k][cell] = "yellow"
						cclass[k][cell] = "yellow"

					df.loc[df.TID==self.track_selected,"STATUS"] = 2
					df.loc[df.TID==self.track_selected,"STATUS_COLOR"] = "yellow"
								
					df.loc[df.TID==self.track_selected,"CLASS"] = 2
					df.loc[df.TID==self.track_selected,"T0"] = -1
					df.loc[df.TID==self.track_selected, "CLASS_COLOR"] = "yellow"


				# Disconnect exclusion mode at the end
				self.myFigCanvas.mpl_disconnect(self.cidpress)
				self.myFigCanvas.mpl_disconnect(self.cidrelease)
				self.myFigCanvas.mpl_disconnect(self.cidmotion)
				self.cid = self.myFigCanvas.mpl_connect('button_press_event', self.onclick)				
			elif returnValue == QMessageBox.No:
				self.myFigCanvas.disable_exclusion_mode()

		return(self.myFigCanvas.bottom_right)

	def rect_on_motion(self,event):
		pass


	def set_last_frame(self):
		#self.set_last_btn.hide()
		self.set_last_btn.setEnabled(False)
		self.set_last_btn.disconnect()

		self.myFigCanvas.set_last_frame(49)
		#self.cell_plot.draw()
		self.stop_btn.hide()
		self.start_btn.show()
		self.stop_btn.clicked.connect(self.start_anim)

		self.start_btn.setShortcut(QKeySequence("l"))


	def update_cell_plot(self):
		global df
		global stack

		# Drop off the first y element, append a new one.
		self.track_selected = df[(df["T"]==0)]["TID"].to_numpy()[self.tc_index]
		df_at_track = df[(df["TID"]==self.track_selected)]
		time_array = df_at_track["T"].to_numpy()
		cclass = df_at_track.CLASS.to_numpy()[0]
		t0 = df_at_track.T0.to_numpy()[0]
		blue_signal = df_at_track["BLUE_INTENSITY"].to_numpy()
		red_signal = df_at_track["RED_INTENSITY"].to_numpy()
		xpos = df_at_track.X.to_numpy()
		ypos = df_at_track.Y.to_numpy()

		self.cell_plot.plot_cell()

		# self.cell_plot.line_blue.set_ydata(blue_signal)
		# self.cell_plot.line_red.set_ydata(red_signal)
		# self.cell_plot.line_dt.set_xdata([t0,t0])
		# self.cell_plot.draw()

	def stop_anim(self):
		self.stop_btn.hide()
		self.start_btn.show()
		self.myFigCanvas.anim.pause()
		self.stop_btn.clicked.connect(self.start_anim)

	def start_anim(self):

		self.start_btn.setShortcut(QKeySequence(""))

		self.set_last_btn.setEnabled(True)
		self.set_last_btn.clicked.connect(self.set_last_frame)

		self.start_btn.hide()
		self.stop_btn.show()
		self.myFigCanvas.start()
		self.stop_btn.clicked.connect(self.stop_anim)
	
	def find_closest(self,blob,xclick0,yclick0):
		distance_all = []
		distance = np.sqrt((blob[0,:]-xclick0)**2+(blob[1,:]-yclick0)**2)
		distance_all.append(distance)
		index = np.argmin(distance_all)
		x,y = blob[:,index]
		return(x,y,index)
	
	def cancel_cell_selection(self):
		self.center_coords.pop(0)
		#self.center_coords.remove((self.selected_x,self.selected_y,self.tc_index))
		#self.cell_info.setText("")
		cscats[:,self.tc_index] = self.previous_color #cscats0[k][self.tc_index]
		#self.cancel_btn.hide()
		#self.change_btn.hide()
		self.cancel_btn.setEnabled(False)
		self.change_btn.setEnabled(False)


	def buttonPress_change_class(self):
		self.change_btn.disconnect()
		self.change_btn.setText("Submit")
		self.change_btn.clicked.connect(self.submit_action)
		self.change_btn.setShortcut(QKeySequence("Enter"))
		self.change_btn.setShortcut(QKeySequence("Return"))

		self.red_btn.show()
		self.blue_btn.show()
		self.yellow_btn.show()
		self.delete_btn.show()

		self.dt_label.show()
		self.e1.show()

		self.default_time = df[(df["T"]==0)]["T0"].to_numpy()[self.tc_index]
		
		self.e1.setText(str(self.default_time))
	
	def buttonPress_red(self):
	
		self.e1.setEnabled(True)
		
		self.default_time = df[(df["T"]==0)]["T0"].to_numpy()[self.tc_index]
		self.new_cell_class = 0
		self.new_cell_death_time = self.default_time
		self.new_cell_color = "tab:orange"
		

	def buttonPress_blue(self):
	
		self.e1.setEnabled(False)
		
		self.new_cell_class = 1
		self.new_cell_death_time = -1
		self.new_cell_color = "tab:cyan"


	def buttonPress_yellow(self):
	
		self.e1.setEnabled(False)
		
		self.new_cell_class = 2
		self.new_cell_death_time = -1
		self.new_cell_color = "y"

	def buttonPress_delete(self):
	
		self.e1.setEnabled(False)
		self.new_cell_class = -1
		self.new_cell_death_time = -1
		self.new_cell_color = "k"


	def save_csv(self):
		global df
		path=str(pos_folder)+"/output/tables/lysis_table_checked.csv"
		if os.path.exists(path):
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("A checked version has been found.\nDo you want to rewrite it?")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Cancel:
				return(None)			
			elif returnValue == QMessageBox.Ok:
				os.remove(path)
				#df = df[df["CLASS"]!=-1]
				df.drop(df[df.CLASS==-1].index, inplace=True)
				df.to_csv(path, index=False)
				print(f"Visual table saved in {path}")
		else:
			df.to_csv(path, index=False)
			print(f"Visual table saved in {path}")
	

	def submit_action(self):
	
		#Reinitialize change class button
		self.change_btn.disconnect()
		self.change_btn.clicked.connect(self.buttonPress_change_class)
		self.change_btn.setText("Change class")
		
		if self.red_btn.isChecked():
		
			try:
				self.new_cell_death_time = float(self.e1.text())
			except ValueError:
				self.new_cell_death_time = -1.0
			self.e1.setText("")
			if self.new_cell_death_time<0:
				self.new_cell_death_time=0
			elif self.new_cell_death_time>len(cscats):
				self.new_cell_death_time=len(cscats)
			for k in range(0,len(cclass)):
				cclass[k][self.tc_index] = self.new_cell_color			
			for k in range(0,int(self.new_cell_death_time)):
				cscats[k][self.tc_index] = "tab:cyan"
			for k in range(int(self.new_cell_death_time),len(cscats)):
				cscats[k][self.tc_index] = "tab:orange"
				
			c_array = np.array(cscats[:,self.tc_index])
			bin_status = [0 if c=="r" else 1 for c in c_array]
			df.loc[df.TID==self.track_selected,"STATUS"] = bin_status
			
			df_at_track = df[(df["TID"]==self.track_selected)]
			track_indices = df_at_track.index
			
			for j in range(len_movie):
				status,status_color = get_status_color(0,np.linspace(0,len_movie-1,len_movie)[j],self.new_cell_death_time,len_movie)
				df.loc[track_indices[j],"STATUS"] = status
				df.loc[track_indices[j],"STATUS_COLOR"] = status_color
			
		else:
			if self.blue_btn.isChecked():
				self.new_cell_class = 1
			elif self.yellow_btn.isChecked():
				self.new_cell_class = 2
			elif self.delete_btn.isChecked():
				self.new_cell_class = -1

			for k in range(0,len(cscats)):
				cscats[k][self.tc_index] = self.new_cell_color
				cclass[k][self.tc_index] = self.new_cell_color

			df.loc[df.TID==self.track_selected,"STATUS"] = 1
			df.loc[df.TID==self.track_selected,"STATUS_COLOR"] = get_class_color(self.new_cell_class)
						
		
		#self.track_selected = df[(df["T"]==0)]["TID"].to_numpy()[self.tc_index]
		print(self.track_selected, self.new_cell_class, self.new_cell_death_time)
		df.loc[df.TID==self.track_selected,"CLASS"] = self.new_cell_class
		df.loc[df.TID==self.track_selected,"T0"] = self.new_cell_death_time
		df.loc[df.TID==self.track_selected, "CLASS_COLOR"] = get_class_color(self.new_cell_class)
		
		#print(get_class_color(self.new_cell_class),self.new_cell_color)
		
		self.center_coords.remove(self.center_coords[-1])
		#self.cell_info.setText("")

		self.red_btn.hide()
		self.blue_btn.hide()
		self.yellow_btn.hide()
		self.delete_btn.hide()

		self.change_btn.setEnabled(False)
		self.cancel_btn.setEnabled(False)

		#self.change_btn.hide()
		#self.cancel_btn.hide()
		self.e1.hide()
		self.dt_label.hide()
	
	def function_abort(self):
		self.save_message = QMessageBox()
		self.save_message.setText("Do you want to save your modifications?")
		self.save_message.setStandardButtons(QMessageBox.Cancel | QMessageBox.Discard | QMessageBox.Ok)
		returnValue = self.save_message.exec()
		if returnValue == QMessageBox.Ok:
			self.save_csv()
			os.abort()
		elif returnValue == QMessageBox.Discard:
			os.abort()
		else:
			pass
	
	def file_save(self):
		global df
		pathsave_custom = QFileDialog.getSaveFileName(self, "Select file name", pos, "CSV files (*.csv)")[0]
		if pathsave_custom.endswith(".csv"):
			df = remove_unnamed_col(df)
			df.drop(df[df.CLASS==-1].index, inplace=True)
			df.to_csv(pathsave_custom)

	def export_training_v2(self):

		#df.drop(df[df.CLASS==-1].index, inplace=True)
		tracks = np.unique(df["TID"].to_numpy())
		training_set = []
		cols = df.columns

		for track in tracks:
			# Add all signals at given track
			signals = {}
			for c in cols:
				signals.update({c: df.loc[df["TID"]==track, c].to_numpy()})
			# live_nuc_signal = df.loc[df["TID"]==track, "BLUE_INTENSITY"].to_numpy()
			# dead_nuc_signal = df.loc[df["TID"]==track, "RED_INTENSITY"].to_numpy()
			lysis_time = df.loc[df["TID"]==track, "T0"].to_numpy()[0]
			cclass = df.loc[df["TID"]==track, "CLASS"].to_numpy()[0]
			signals.update({"lysis_time": lysis_time,"class": cclass})
			# Here auto add all available channels
			training_set.append(signals)

		pathsave_ = QFileDialog.getSaveFileName(self, "Select file name", pos, "NPY files (*.npy)")[0]
		if not pathsave_.endswith(".npy"):
			pathsave_ += ".npy"
		try:
			np.save(pathsave_,training_set)
		except Exception as e:
			print(f"Error {e}...")


	def export_training(self):
		#global df
		df.drop(df[df.CLASS==-1].index, inplace=True)
		unique_tracks = np.unique(df["TID"].to_numpy())
		inputs = []
		for tid in unique_tracks:
			dft = df[(df["TID"]==tid)]
			signal_b = dft["BLUE_INTENSITY"].to_numpy()
			signal_r = dft["RED_INTENSITY"].to_numpy()
			x0track = dft["T0"].to_numpy()[0]
			class_track = dft["CLASS"].to_numpy()[0]
			inputs.append([class_track,signal_b,signal_r,tid,x0track])
		inputs = np.array(inputs,dtype=object)
		pathsave_ = QFileDialog.getSaveFileName(self, "Select file name", pos, "NPY files (*.npy)")[0]
		if pathsave_.endswith(".npy"):
			np.save(pathsave_,inputs)

	# def retrain_model(self):
	# 	self.retrain = RetrainWindow()
	# 	self.retrain.show()

	
	def _createActions(self):
			# Creating action using the first constructor
			
			self.saveFile = QAction(self)
			self.saveFile.setText("&Save As...")
			self.saveFile.triggered.connect(self.file_save)
			self.saveFile.setShortcut("Ctrl+S")
			
			self.exitAction = QAction(self)
			self.exitAction.setText("&Exit")
			self.exitAction.triggered.connect(self.function_abort)

			self.exportAIset = QAction(self)
			self.exportAIset.setText("&Export training set...")
			self.exportAIset.triggered.connect(self.export_training_v2)

			# self.retrainAI = QAction(self)
			# self.retrainAI.setText("&Retrain AI model...")
			# self.retrainAI.triggered.connect(self.retrain_model)
			
			#self.helpContentAction = QAction(QIcon("/home/limozin/Downloads/test.svg"), "&Open...", self)
		
	def _createMenuBar(self):
		menuBar = self.menuBar()
		# Creating menus using a QMenu object
		fileMenu = QMenu("&File", self)
		menuBar.addMenu(fileMenu)
		fileMenu.addAction(self.saveFile)
		fileMenu.addAction(self.exportAIset)
		#fileMenu.addAction(self.retrainAI)
		fileMenu.addAction(self.exitAction)

		#helpMenu = QMenu("&Help",self)
		#menuBar.addMenu(helpMenu)
		#helpMenu.addAction(self.helpContentAction)
		
	def onclick(self,event):
		"""
		This on-click function highlights in lime color the cell that has been selected and extracts its track ID.
		"""	
		if event.dblclick:
			global ix, iy
			global death_t
			global cscats
			
			ix, iy = event.xdata, event.ydata
			instant = self.myFigCanvas.iter_val
			self.selected_x,self.selected_y,temp_tc_index = self.find_closest(np.array([xscats[0],yscats[0]]),ix,iy)
			#print(self.center_coords)
			check_if_second_selection = len(self.center_coords)==1 #(self.selected_x,self.selected_y,temp_tc_index) in self.center_coords

			if len(self.center_coords)==0:
				self.cancel_btn.setShortcut(QKeySequence("Esc"))
				self.tc_index = temp_tc_index
				self.center_coords.append((self.selected_x, self.selected_y,self.tc_index))
				#print(cscats[:][self.tc_index])
				
				self.previous_color = np.copy(cscats[:,self.tc_index]) #cscats[0][self.tc_index]
				
				cscats[:,self.tc_index] = 'lime'
				#print(self.previous_color,cscats[:,self.tc_index])
				default_time = df[(df["T"]==0)]["T0"].to_numpy()[self.tc_index]
				self.track_selected = df[(df["T"]==0)]["TID"].to_numpy()[self.tc_index]
				classification_label = df[(df["TID"]==self.track_selected)]["CLASS"].to_numpy()[0]
				mean_x = np.mean(df[(df["TID"]==self.track_selected)]["X"].to_numpy())
				mean_y = np.mean(df[(df["TID"]==self.track_selected)]["Y"].to_numpy())

				self.cell_info.setText(f"Cell selected: track {self.track_selected}\nClassification = {classification_label}\nEstimated death time = {round(default_time,2)}")
				self.cell_plot.tc_index = self.tc_index
				
				self.update_cell_plot()
				self.show()

				self.change_btn.setEnabled(True)
				self.cancel_btn.setEnabled(True)

			elif (check_if_second_selection==True)*(self.center_coords[0][2]==temp_tc_index):
				self.cancel_btn.setShortcut(QKeySequence())

				cscats[:,self.tc_index] = self.previous_color #cscats0[k][self.tc_index]

				self.center_coords.pop(0)
				self.change_btn.setEnabled(False)
				self.cancel_btn.setEnabled(False)
			else:
				pass 

App = QApplication(sys.argv)
App.setStyle("Fusion")
window = Window()

sys.exit(App.exec())
