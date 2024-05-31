import numpy as np
from celldetective.io import auto_load_number_of_frames, load_frames
from celldetective.filters import *
from celldetective.segmentation import filter_image
from celldetective.measure import contour_of_instance_segmentation
from celldetective.utils import _get_img_num_per_channel
from tifffile import imread
import matplotlib.pyplot as plt 
from stardist import fill_label_holes
from pathlib import Path
from natsort import natsorted
from glob import glob
import os

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QComboBox, QLineEdit, QListWidget
from PyQt5.QtCore import Qt, QSize
from celldetective.gui.gui_utils import FigureCanvas, QuickSliderLayout, center_window
from celldetective.gui import Styles
from superqt import QLabeledDoubleSlider, QLabeledSlider, QLabeledDoubleRangeSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from matplotlib_scalebar.scalebar import ScaleBar
import gc


class StackVisualizer(QWidget, Styles):

	"""
	A widget around an imshow and accompanying sliders.
	"""
	def __init__(self, stack=None, stack_path=None, frame_slider=True, contrast_slider=True, channel_cb=False, channel_names=None, n_channels=1, target_channel=0, window_title='View', PxToUm=None, background_color='transparent',imshow_kwargs={}):
		super().__init__()

		#self.setWindowTitle(window_title)
		self.window_title = window_title

		self.stack = stack
		self.stack_path = stack_path
		self.create_frame_slider = frame_slider
		self.background_color = background_color
		self.create_contrast_slider = contrast_slider
		self.create_channel_cb = channel_cb
		self.n_channels = n_channels
		self.channel_names = channel_names
		self.target_channel = target_channel
		self.imshow_kwargs = imshow_kwargs
		self.PxToUm = PxToUm
		self.init_contrast = False

		self.load_stack() # need to get stack, frame etc
		self.generate_figure_canvas()
		if self.create_channel_cb:
			self.generate_channel_cb()
		if self.create_contrast_slider:
			self.generate_contrast_slider()
		if self.create_frame_slider:
			self.generate_frame_slider()

		center_window(self)
		self.setAttribute(Qt.WA_DeleteOnClose)

	def show(self):
		self.canvas.show()

	def load_stack(self):

		if self.stack is not None:

			if isinstance(self.stack, list):
				self.stack = np.array(self.stack)

			if self.stack.ndim==3:
				print('No channel axis found...')
				self.stack = self.stack[:,:,:,np.newaxis]
				self.target_channel = 0
			
			self.mode = 'direct'
			self.stack_length = len(self.stack)
			self.mid_time = self.stack_length // 2
			self.init_frame = self.stack[self.mid_time,:,:,self.target_channel]
			self.last_frame = self.stack[-1,:,:,self.target_channel]
		else:
			self.mode = 'virtual'
			assert isinstance(self.stack_path, str)
			assert self.stack_path.endswith('.tif')
			self.locate_image_virtual()

	def locate_image_virtual(self):

		self.stack_length = auto_load_number_of_frames(self.stack_path)
		if self.stack_length is None:
			stack = imread(self.stack_path)
			self.stack_length = len(stack)
			del stack
			gc.collect()

		self.mid_time = self.stack_length // 2
		self.img_num_per_channel = _get_img_num_per_channel(np.arange(self.n_channels), self.stack_length, self.n_channels)

		self.init_frame = load_frames(self.img_num_per_channel[self.target_channel, self.mid_time], 
									  self.stack_path,
									  normalize_input=False).astype(float)[:,:,0]
		self.last_frame = load_frames(self.img_num_per_channel[self.target_channel, self.stack_length-1], 
									  self.stack_path,
									  normalize_input=False).astype(float)[:,:,0]

	def generate_figure_canvas(self):

		self.fig, self.ax = plt.subplots(tight_layout=True) #figsize=(5, 5)
		self.canvas = FigureCanvas(self.fig, title=self.window_title, interactive=True)
		self.ax.clear()
		self.im = self.ax.imshow(self.init_frame, cmap='gray', interpolation='none', **self.imshow_kwargs)
		if self.PxToUm is not None:
			scalebar = ScaleBar(self.PxToUm,
								"um",
								length_fraction=0.25,
								location='upper right',
								border_pad=0.4,
								box_alpha=0.95,
								color='white',
								box_color='black',
								)
			if self.PxToUm==1:
				scalebar = ScaleBar(1,
								"px",
								dimension="pixel-length",
								length_fraction=0.25,
								location='upper right',
								border_pad=0.4,
								box_alpha=0.95,
								color='white',
								box_color='black',
								)
			self.ax.add_artist(scalebar)
		self.ax.set_xticks([])
		self.ax.set_yticks([])
		self.fig.set_facecolor('none')  # or 'None'
		self.fig.canvas.setStyleSheet(f"background-color: {self.background_color};")
		self.canvas.canvas.draw()

	def generate_channel_cb(self):

		assert self.channel_names is not None
		assert len(self.channel_names)==self.n_channels

		channel_layout = QHBoxLayout()
		channel_layout.setContentsMargins(15,0,15,0)
		channel_layout.addWidget(QLabel('Channel: '), 25)

		self.channels_cb = QComboBox()
		self.channels_cb.addItems(self.channel_names)
		self.channels_cb.currentIndexChanged.connect(self.set_channel_index)
		channel_layout.addWidget(self.channels_cb, 75)
		self.canvas.layout.addLayout(channel_layout)

	def generate_contrast_slider(self):
		
		self.contrast_slider = QLabeledDoubleRangeSlider()
		contrast_layout = QuickSliderLayout(
											label='Contrast: ',
											slider=self.contrast_slider,
											slider_initial_value=[np.nanpercentile(self.init_frame, 1),np.nanpercentile(self.init_frame, 99.99)],
											slider_range=(np.nanmin(self.init_frame),np.nanmax(self.init_frame)),
											decimal_option=True,
											precision=1.0E-05,
											)
		contrast_layout.setContentsMargins(15,0,15,0)
		self.im.set_clim(vmin=np.nanpercentile(self.init_frame, 1),vmax=np.nanpercentile(self.init_frame, 99.99))
		self.contrast_slider.valueChanged.connect(self.change_contrast)
		self.canvas.layout.addLayout(contrast_layout)



	def generate_frame_slider(self):
	
		self.frame_slider = QLabeledSlider()
		frame_layout = QuickSliderLayout(
										label='Frame: ',
										slider=self.frame_slider,
										slider_initial_value=int(self.mid_time),
										slider_range=(0,self.stack_length-1),
										decimal_option=False,
										)
		frame_layout.setContentsMargins(15,0,15,0)
		self.frame_slider.valueChanged.connect(self.change_frame)
		self.canvas.layout.addLayout(frame_layout)

	def set_target_channel(self, value):
		
		self.target_channel = value
		self.change_frame(self.frame_slider.value())

	def change_contrast(self, value):

		vmin = value[0]
		vmax = value[1]
		self.im.set_clim(vmin=vmin, vmax=vmax)
		self.fig.canvas.draw_idle()

	def set_channel_index(self, value):

		self.target_channel = value
		self.init_contrast = True
		if self.mode == 'direct':
			self.last_frame = self.stack[-1,:,:,self.target_channel]
		elif self.mode == 'virtual':
			self.last_frame = load_frames(self.img_num_per_channel[self.target_channel, self.stack_length-1], 
										  self.stack_path,
										  normalize_input=False).astype(float)[:,:,0]
		self.change_frame(self.frame_slider.value())
		self.init_contrast = False

	def change_frame(self, value):
		
		if self.mode=='virtual':

			self.init_frame = load_frames(self.img_num_per_channel[self.target_channel, value], 
								self.stack_path,
								normalize_input=False
								).astype(float)[:,:,0]
		elif self.mode=='direct':
			self.init_frame = self.stack[value,:,:,self.target_channel].copy()
		
		self.im.set_data(self.init_frame)
		
		if self.init_contrast:
			self.im.autoscale()
			I_min, I_max = self.im.get_clim()
			self.contrast_slider.setRange(np.nanmin([self.init_frame,self.last_frame]),np.nanmax([self.init_frame,self.last_frame]))
			self.contrast_slider.setValue((I_min,I_max))

		if self.create_contrast_slider:
			self.change_contrast(self.contrast_slider.value())

	
	def closeEvent(self, event):
		self.canvas.close()


class ThresholdedStackVisualizer(StackVisualizer):

	"""
	A widget around an imshow and accompanying sliders.
	"""
	def __init__(self, preprocessing=None, parent_le=None, initial_threshold=5, initial_mask_alpha=0.5, *args, **kwargs):
		
		super().__init__(*args, **kwargs)
		self.preprocessing = preprocessing
		self.thresh = initial_threshold
		self.mask_alpha = initial_mask_alpha
		self.parent_le = parent_le
		self.compute_mask(self.thresh)
		self.generate_mask_imshow()
		self.generate_threshold_slider()
		self.generate_opacity_slider()
		if isinstance(self.parent_le, QLineEdit):
			self.generate_apply_btn()

	def generate_apply_btn(self):
		
		apply_hbox = QHBoxLayout()
		self.apply_threshold_btn = QPushButton('Apply')
		self.apply_threshold_btn.clicked.connect(self.set_threshold_in_parent_le)
		self.apply_threshold_btn.setStyleSheet(self.button_style_sheet)
		apply_hbox.addWidget(QLabel(''),33)
		apply_hbox.addWidget(self.apply_threshold_btn, 33)
		apply_hbox.addWidget(QLabel(''),33)		
		self.canvas.layout.addLayout(apply_hbox)

	def set_threshold_in_parent_le(self):
		self.parent_le.set_threshold(self.threshold_slider.value())
		self.close()

	def generate_mask_imshow(self):
		
		self.im_mask = self.ax.imshow(np.ma.masked_where(self.mask==0, self.mask), alpha=self.mask_alpha, interpolation='none')
		self.canvas.canvas.draw()

	def generate_threshold_slider(self):

		self.threshold_slider = QLabeledDoubleSlider()
		thresh_layout = QuickSliderLayout(label='Threshold: ',
										slider=self.threshold_slider,
										slider_initial_value=self.thresh,
										slider_range=(0,30),
										decimal_option=True,
										precision=1.0E-05,
										)
		thresh_layout.setContentsMargins(15,0,15,0)
		self.threshold_slider.valueChanged.connect(self.change_threshold)
		self.canvas.layout.addLayout(thresh_layout)

	def generate_opacity_slider(self):

		self.opacity_slider = QLabeledDoubleSlider()
		opacity_layout = QuickSliderLayout(label='Opacity: ',
										slider=self.opacity_slider,
										slider_initial_value=0.5,
										slider_range=(0,1),
										decimal_option=True,
										precision=1.0E-03
										)
		opacity_layout.setContentsMargins(15,0,15,0)
		self.opacity_slider.valueChanged.connect(self.change_mask_opacity)
		self.canvas.layout.addLayout(opacity_layout)

	def change_mask_opacity(self, value):

		self.mask_alpha = value
		self.im_mask.set_alpha(self.mask_alpha)
		self.canvas.canvas.draw_idle()

	def change_threshold(self, value):
		
		self.thresh = value
		self.compute_mask(self.thresh)
		mask = np.ma.masked_where(self.mask == 0, self.mask)
		self.im_mask.set_data(mask)
		self.canvas.canvas.draw_idle()

	def change_frame(self, value):
		
		super().change_frame(value)
		self.change_threshold(self.threshold_slider.value())

	def compute_mask(self, threshold_value):

		self.preprocess_image()
		self.mask = self.processed_image > threshold_value
		self.mask = fill_label_holes(self.mask).astype(int)

	def preprocess_image(self):
		
		if self.preprocessing is not None:

			assert isinstance(self.preprocessing, list)
			self.processed_image = filter_image(self.init_frame.copy(),filters=self.preprocessing)


class CellEdgeVisualizer(StackVisualizer):

	"""
	A widget around an imshow and accompanying sliders.
	"""
	def __init__(self, cell_type="effectors", edge_range=(-30,30), invert=False, parent_list_widget=None, parent_le=None, labels=None, initial_edge=5, initial_mask_alpha=0.5, *args, **kwargs):
		
		super().__init__(*args, **kwargs)
		self.edge_size = initial_edge
		self.mask_alpha = initial_mask_alpha
		self.cell_type = cell_type
		self.labels = labels
		self.edge_range = edge_range
		self.invert = invert
		self.parent_list_widget = parent_list_widget
		self.parent_le = parent_le

		self.load_labels()
		self.generate_label_imshow()
		self.generate_edge_slider()
		self.generate_opacity_slider()
		if isinstance(self.parent_list_widget, QListWidget):
			self.generate_add_to_list_btn()
		if isinstance(self.parent_le, QLineEdit):
			self.generate_add_to_le_btn()

	def load_labels(self):

		if self.labels is not None:

			if isinstance(self.labels, list):
				self.labels = np.array(self.labels)

			assert self.labels.ndim==3,'Wrong dimensions for the provided labels, expect TXY'
			assert len(self.labels)==self.stack_length

			self.mode = 'direct'
			self.init_label = self.labels[self.mid_time,:,:]
		else:
			self.mode = 'virtual'
			assert isinstance(self.stack_path, str)
			assert self.stack_path.endswith('.tif')
			self.locate_labels_virtual()
		
		self.compute_edge_labels()

	def locate_labels_virtual(self):

		labels_path = str(Path(self.stack_path).parent.parent) + os.sep + f'labels_{self.cell_type}' + os.sep
		self.mask_paths = natsorted(glob(labels_path + '*.tif'))
		
		if len(self.mask_paths) == 0:

			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Critical)
			msgBox.setText("No labels were found for the selected cells. Abort.")
			msgBox.setWindowTitle("Critical")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			self.close()

		self.init_label = imread(self.mask_paths[self.frame_slider.value()])

	def generate_add_to_list_btn(self):
		
		add_hbox = QHBoxLayout()
		self.add_measurement_btn = QPushButton('Add measurement')
		self.add_measurement_btn.clicked.connect(self.set_measurement_in_parent_list)
		self.add_measurement_btn.setIcon(icon(MDI6.plus,color="white"))
		self.add_measurement_btn.setIconSize(QSize(20, 20))
		self.add_measurement_btn.setStyleSheet(self.button_style_sheet)
		add_hbox.addWidget(QLabel(''),33)
		add_hbox.addWidget(self.add_measurement_btn, 33)
		add_hbox.addWidget(QLabel(''),33)		
		self.canvas.layout.addLayout(add_hbox)

	def generate_add_to_le_btn(self):
		
		add_hbox = QHBoxLayout()
		self.set_measurement_btn = QPushButton('Set')
		self.set_measurement_btn.clicked.connect(self.set_measurement_in_parent_le)
		self.set_measurement_btn.setStyleSheet(self.button_style_sheet)
		add_hbox.addWidget(QLabel(''),33)
		add_hbox.addWidget(self.set_measurement_btn, 33)
		add_hbox.addWidget(QLabel(''),33)		
		self.canvas.layout.addLayout(add_hbox)

	def set_measurement_in_parent_le(self):
		
		self.parent_le.setText(str(int(self.edge_slider.value())))
		self.close()

	def set_measurement_in_parent_list(self):
		
		self.parent_list_widget.addItems([str(self.edge_slider.value())])
		self.close()

	def generate_label_imshow(self):
		
		self.im_mask = self.ax.imshow(np.ma.masked_where(self.edge_labels==0, self.edge_labels), alpha=self.mask_alpha, interpolation='none', cmap="viridis")
		self.canvas.canvas.draw()

	def generate_edge_slider(self):

		self.edge_slider = QLabeledSlider()
		edge_layout = QuickSliderLayout(label='Edge: ',
										slider=self.edge_slider,
										slider_initial_value=self.edge_size,
										slider_range=self.edge_range,
										decimal_option=False,
										)
		edge_layout.setContentsMargins(15,0,15,0)
		self.edge_slider.valueChanged.connect(self.change_edge_size)
		self.canvas.layout.addLayout(edge_layout)

	def generate_opacity_slider(self):

		self.opacity_slider = QLabeledDoubleSlider()
		opacity_layout = QuickSliderLayout(label='Opacity: ',
										slider=self.opacity_slider,
										slider_initial_value=0.5,
										slider_range=(0,1),
										decimal_option=True,
										precision=1.0E-03
										)
		opacity_layout.setContentsMargins(15,0,15,0)
		self.opacity_slider.valueChanged.connect(self.change_mask_opacity)
		self.canvas.layout.addLayout(opacity_layout)

	def change_mask_opacity(self, value):

		self.mask_alpha = value
		self.im_mask.set_alpha(self.mask_alpha)
		self.canvas.canvas.draw_idle()

	def change_edge_size(self, value):
		
		self.edge_size = value
		self.compute_edge_labels()
		mask = np.ma.masked_where(self.edge_labels == 0, self.edge_labels)
		self.im_mask.set_data(mask)
		self.canvas.canvas.draw_idle()

	def change_frame(self, value):
		
		super().change_frame(value)

		if self.mode=='virtual':
			self.init_label = imread(self.mask_paths[value])
		elif self.mode=='direct':
			self.init_label = self.labels[value,:,:]
		
		self.compute_edge_labels()
		mask = np.ma.masked_where(self.edge_labels == 0, self.edge_labels)
		self.im_mask.set_data(mask)

	def compute_edge_labels(self):
		
		if self.invert:
			edge_size = - self.edge_size
		else:
			edge_size = self.edge_size

		self.edge_labels = contour_of_instance_segmentation(self.init_label, edge_size)

class CellSizeViewer(StackVisualizer):

	"""
	A widget around an imshow and accompanying sliders.
	"""
	def __init__(self, initial_diameter=40, diameter_slider_range=(0,200), parent_le=None, parent_list_widget=None, *args, **kwargs):
		
		super().__init__(*args, **kwargs)
		self.diameter = initial_diameter
		self.parent_le = parent_le
		self.diameter_slider_range = diameter_slider_range
		self.parent_list_widget = parent_list_widget
		self.generate_circle()
		self.generate_diameter_slider()

		if isinstance(self.parent_le, QLineEdit):
			self.generate_set_btn()
		if isinstance(self.parent_list_widget, QListWidget):
			self.generate_add_to_list_btn()

	def generate_circle(self):

		self.circ = plt.Circle((self.init_frame.shape[0]//2,self.init_frame.shape[1]//2), self.diameter//2, ec="tab:red",fill=False)
		self.ax.add_patch(self.circ)

		self.ax.callbacks.connect('xlim_changed',self.on_xlims_or_ylims_change)
		self.ax.callbacks.connect('ylim_changed', self.on_xlims_or_ylims_change)

	def generate_add_to_list_btn(self):
		
		add_hbox = QHBoxLayout()
		self.add_measurement_btn = QPushButton('Add measurement')
		self.add_measurement_btn.clicked.connect(self.set_measurement_in_parent_list)
		self.add_measurement_btn.setIcon(icon(MDI6.plus,color="white"))
		self.add_measurement_btn.setIconSize(QSize(20, 20))
		self.add_measurement_btn.setStyleSheet(self.button_style_sheet)
		add_hbox.addWidget(QLabel(''),33)
		add_hbox.addWidget(self.add_measurement_btn, 33)
		add_hbox.addWidget(QLabel(''),33)		
		self.canvas.layout.addLayout(add_hbox)

	def set_measurement_in_parent_list(self):
		
		self.parent_list_widget.addItems([str(self.diameter_slider.value()//2)])
		self.close()

	def on_xlims_or_ylims_change(self, event_ax):

		xmin,xmax = event_ax.get_xlim()
		ymin,ymax = event_ax.get_ylim()
		self.circ.center = np.mean([xmin,xmax]), np.mean([ymin,ymax])

	def generate_set_btn(self):
		
		apply_hbox = QHBoxLayout()
		self.apply_threshold_btn = QPushButton('Set')
		self.apply_threshold_btn.clicked.connect(self.set_threshold_in_parent_le)
		self.apply_threshold_btn.setStyleSheet(self.button_style_sheet)
		apply_hbox.addWidget(QLabel(''),33)
		apply_hbox.addWidget(self.apply_threshold_btn, 33)
		apply_hbox.addWidget(QLabel(''),33)		
		self.canvas.layout.addLayout(apply_hbox)

	def set_threshold_in_parent_le(self):
		self.parent_le.set_threshold(self.diameter_slider.value())
		self.close()

	def generate_diameter_slider(self):

		self.diameter_slider = QLabeledDoubleSlider()
		diameter_layout = QuickSliderLayout(label='Diameter: ',
										slider=self.diameter_slider,
										slider_initial_value=self.diameter,
										slider_range=self.diameter_slider_range,
										decimal_option=True,
										precision=1.0E-05,
										)
		diameter_layout.setContentsMargins(15,0,15,0)
		self.diameter_slider.valueChanged.connect(self.change_diameter)
		self.canvas.layout.addLayout(diameter_layout)

	def change_diameter(self, value):
		
		self.diameter = value
		self.circ.set_radius(self.diameter//2)
		self.canvas.canvas.draw_idle()