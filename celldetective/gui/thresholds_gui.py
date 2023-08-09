from PyQt5.QtWidgets import QMainWindow, QMessageBox, QLabel, QWidget,QFileDialog, QHBoxLayout, QGridLayout, QLineEdit, QScrollArea, QVBoxLayout, QComboBox, QPushButton, QApplication, QPushButton
from celldetective.gui.gui_utils import center_window, FigureCanvas, ListWidget, FilterChoice, color_from_class
from celldetective.utils import get_software_location, extract_experiment_channels, rename_intensity_column
from celldetective.io import auto_load_number_of_frames, load_frames
from celldetective.segmentation import threshold_image, identify_markers_from_binary, apply_watershed, segment_frame_from_thresholds
from scipy.ndimage import binary_fill_holes
from PyQt5.QtCore import Qt, QSize
from glob import glob
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import numpy as np
import matplotlib.pyplot as plt
from superqt import QLabeledSlider, QLabeledDoubleRangeSlider
from celldetective.segmentation import filter_image
import pandas as pd
from skimage.measure import regionprops_table
import json

class ThresholdConfigWizard(QMainWindow):
	
	"""
	UI to create a threshold pipeline for segmentation.

	"""

	def __init__(self, parent=None):
		
		super().__init__()
		self.parent = parent

		self.screen_height = self.parent.parent.parent.parent.screen_height
		self.screen_width = self.parent.parent.parent.parent.screen_width
		self.setMinimumWidth(int(0.8*self.screen_width))
		self.setMinimumHeight(int(0.8*self.screen_height))
		self.setWindowTitle("Threshold configuration wizard")
		center_window(self)

		self.mode = self.parent.mode
		self.pos = self.parent.parent.parent.pos
		self.exp_dir = self.parent.parent.exp_dir
		self.soft_path = get_software_location()
		self.footprint = 30
		self.min_dist = 30
		self.cell_properties = ['centroid','area', 'perimeter', 'eccentricity','intensity_mean']

		if self.mode=="targets":
			self.config_out_name = "threshold_targets.json"
		elif self.mode=="effectors":
			self.config_out_name = "threshold_effectors.json"

		self.locate_stack()
		self.threshold_slider = QLabeledDoubleRangeSlider()
		self.initialize_histogram()
		self.show_image()
		self.initalize_props_scatter()
		self.prep_cell_properties()
		self.populate_widget()

		self.setAttribute(Qt.WA_DeleteOnClose)

	def populate_widget(self):

		"""
		Create the multibox design.

		"""
		self.button_widget = QWidget()
		main_layout = QHBoxLayout()
		self.button_widget.setLayout(main_layout)
		
		main_layout.setContentsMargins(30,30,30,30)

		self.scroll_area = QScrollArea()
		self.scroll_container = QWidget()
		self.scroll_area.setWidgetResizable(True)
		self.scroll_area.setWidget(self.scroll_container)

		self.left_panel = QVBoxLayout(self.scroll_container)
		self.left_panel.setContentsMargins(30,30,30,30)
		self.left_panel.setSpacing(10)
		self.populate_left_panel()

		# Right panel
		self.right_panel = QVBoxLayout()
		self.populate_right_panel()

		# threhsold options
		#self.left_panel.addWidget(self.cell_fcanvas)

		# Animation
		#self.right_panel.addWidget(self.fcanvas)

		#self.populate_left_panel()
		#grid.addLayout(self.left_side, 0, 0, 1, 1)

		# self.scroll_area.setAlignment(Qt.AlignCenter)
		# self.scroll_area.setLayout(self.left_panel)
		# self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		# self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		# self.scroll_area.setWidgetResizable(True)

		main_layout.addWidget(self.scroll_area, 35)
		main_layout.addLayout(self.right_panel, 65)
		self.button_widget.adjustSize()

		self.setCentralWidget(self.button_widget)
		self.show()

		QApplication.processEvents()

	def populate_left_panel(self):

		self.filters_qlist = ListWidget(self, FilterChoice, [])

		grid_preprocess = QGridLayout()
		grid_preprocess.setContentsMargins(20,20,20,20)

		filter_list_option_grid = QHBoxLayout()
		section_preprocess = QLabel("Preprocessing")
		section_preprocess.setStyleSheet("font-weight: bold;")
		filter_list_option_grid.addWidget(section_preprocess,90,alignment=Qt.AlignLeft)

		self.delete_filter = QPushButton("")
		self.delete_filter.setStyleSheet(self.parent.parent.parent.parent.button_select_all)
		self.delete_filter.setIcon(icon(MDI6.trash_can,color="black"))
		self.delete_filter.setToolTip("Remove filter")
		self.delete_filter.setIconSize(QSize(20, 20))
		self.delete_filter.clicked.connect(self.filters_qlist.removeSel)

		self.add_filter = QPushButton("")
		self.add_filter.setStyleSheet(self.parent.parent.parent.parent.button_select_all)
		self.add_filter.setIcon(icon(MDI6.filter_plus,color="black"))
		self.add_filter.setToolTip("Add filter")
		self.add_filter.setIconSize(QSize(20, 20))
		self.add_filter.clicked.connect(self.filters_qlist.addItem)

		#filter_list_option_grid.addWidget(QLabel(""),90)			
		filter_list_option_grid.addWidget(self.delete_filter,5)
		filter_list_option_grid.addWidget(self.add_filter,5)

		grid_preprocess.addLayout(filter_list_option_grid, 0, 0, 1, 3)
		grid_preprocess.addWidget(self.filters_qlist, 1, 0, 1, 3)

		self.apply_filters_btn = QPushButton("Apply")
		self.apply_filters_btn.setIcon(icon(MDI6.filter_cog_outline,color="white"))
		self.apply_filters_btn.setIconSize(QSize(20, 20))
		self.apply_filters_btn.setStyleSheet(self.parent.parent.parent.parent.button_style_sheet)
		self.apply_filters_btn.clicked.connect(self.preprocess_image)
		grid_preprocess.addWidget(self.apply_filters_btn, 2, 0, 1, 3)

		self.left_panel.addLayout(grid_preprocess)

		###################
		# THRESHOLD SECTION
		###################

		grid_threshold = QGridLayout()
		grid_threshold.setContentsMargins(20,20,20,20)
		idx=0

		threshold_title_grid = QHBoxLayout()
		section_threshold = QLabel("Threshold")
		section_threshold.setStyleSheet("font-weight: bold;")
		threshold_title_grid.addWidget(section_threshold,90,alignment=Qt.AlignCenter)

		self.ylog_check = QPushButton("")
		self.ylog_check.setIcon(icon(MDI6.math_log,color="black"))
		self.ylog_check.setStyleSheet(self.parent.parent.parent.parent.button_select_all)
		self.ylog_check.clicked.connect(self.switch_to_log)
		threshold_title_grid.addWidget(self.ylog_check, 5)
		
		self.equalize_option_btn = QPushButton("")
		self.equalize_option_btn.setIcon(icon(MDI6.equalizer,color="black"))
		self.equalize_option_btn.setIconSize(QSize(20,20))
		self.equalize_option_btn.setStyleSheet(self.parent.parent.parent.parent.button_select_all)
		self.equalize_option_btn.setToolTip("Enable histogram matching")
		self.equalize_option_btn.clicked.connect(self.activate_histogram_equalizer)
		self.equalize_option = False
		threshold_title_grid.addWidget(self.equalize_option_btn, 5)
		
		grid_threshold.addLayout(threshold_title_grid, idx, 0,1,2)


		idx+=1

		# Slider to set vmin & vmax
		self.threshold_slider.setSingleStep(0.00001)
		self.threshold_slider.setTickInterval(0.00001)
		self.threshold_slider.setOrientation(1)
		self.threshold_slider.setRange(np.amin(self.img), np.amax(self.img))
		self.threshold_slider.setValue([np.percentile(self.img.flatten(), 90), np.amax(self.img)])
		self.threshold_slider.valueChanged.connect(self.threshold_changed)

		#self.initialize_histogram()
		grid_threshold.addWidget(self.canvas_hist, idx,0,1,3)

		idx+=1

	#self.threshold_contrast_range.valueChanged.connect(self.set_clim_thresh)

		grid_threshold.addWidget(self.threshold_slider,idx,1,1,1)
		self.canvas_hist.setMinimumHeight(self.screen_height//6)
		self.left_panel.addLayout(grid_threshold)

		self.generate_marker_contents()
		self.generate_props_contents()

		#################
		# FINAL SAVE BTN#
		#################

		self.save_btn = QPushButton('Save')
		self.save_btn.setStyleSheet(self.parent.parent.parent.parent.button_style_sheet)
		self.save_btn.clicked.connect(self.write_instructions)
		self.left_panel.addWidget(self.save_btn)

		self.properties_box_widgets = [self.propscanvas, *self.features_cb, 
									   self.property_query_le, self.submit_query_btn, self.save_btn]
		for p in self.properties_box_widgets:
			p.setEnabled(False)

	def generate_marker_contents(self):

		marker_box = QVBoxLayout()
		marker_box.setContentsMargins(30,30,30,30)

		marker_lbl = QLabel('Markers')
		marker_lbl.setStyleSheet("font-weight: bold;")
		marker_box.addWidget(marker_lbl, alignment=Qt.AlignCenter)

		hbox_footprint = QHBoxLayout()
		hbox_footprint.addWidget(QLabel('Footprint: '), 20)
		self.footprint_slider = QLabeledSlider()
		self.footprint_slider.setSingleStep(1)
		self.footprint_slider.setOrientation(1)
		self.footprint_slider.setRange(1,self.binary.shape[0]//4)
		self.footprint_slider.setValue(self.footprint)
		self.footprint_slider.valueChanged.connect(self.set_footprint)	
		hbox_footprint.addWidget(self.footprint_slider, 80)	
		marker_box.addLayout(hbox_footprint)

		hbox_distance = QHBoxLayout()
		hbox_distance.addWidget(QLabel('Min distance: '), 20)
		self.min_dist_slider = QLabeledSlider()
		self.min_dist_slider.setSingleStep(1)
		self.min_dist_slider.setOrientation(1)
		self.min_dist_slider.setRange(0,self.binary.shape[0]//4)
		self.min_dist_slider.setValue(self.min_dist)
		self.min_dist_slider.valueChanged.connect(self.set_min_dist)
		hbox_distance.addWidget(self.min_dist_slider, 80)	
		marker_box.addLayout(hbox_distance)


		hbox_marker_btns = QHBoxLayout()

		self.markers_btn = QPushButton("Run")
		self.markers_btn.clicked.connect(self.detect_markers)
		self.markers_btn.setStyleSheet(self.parent.parent.parent.parent.button_style_sheet)		
		hbox_marker_btns.addWidget(self.markers_btn)

		self.watershed_btn = QPushButton("Watershed")
		self.watershed_btn.setIcon(icon(MDI6.waves_arrow_up,color="white"))
		self.watershed_btn.setIconSize(QSize(20,20))
		self.watershed_btn.clicked.connect(self.apply_watershed_to_selection)
		self.watershed_btn.setStyleSheet(self.parent.parent.parent.parent.button_style_sheet)
		self.watershed_btn.setEnabled(False)
		hbox_marker_btns.addWidget(self.watershed_btn)
		marker_box.addLayout(hbox_marker_btns)

		self.left_panel.addLayout(marker_box)

	def generate_props_contents(self):


		properties_box = QVBoxLayout()
		properties_box.setContentsMargins(30,30,30,30)
		
		properties_lbl = QLabel('Filter on properties')
		properties_lbl.setStyleSheet('font-weight: bold;')
		properties_box.addWidget(properties_lbl, alignment=Qt.AlignCenter)

		properties_box.addWidget(self.propscanvas)

		self.features_cb = [QComboBox() for i in range(2)]
		for i in range(2):
			hbox_feat = QHBoxLayout()
			hbox_feat.addWidget(QLabel(f'feature {i}: '), 20)
			hbox_feat.addWidget(self.features_cb[i], 80)
			properties_box.addLayout(hbox_feat)

		hbox_classify = QHBoxLayout()
		hbox_classify.addWidget(QLabel('classify: '), 10)
		self.property_query_le = QLineEdit()
		self.property_query_le.setPlaceholderText('eliminate points using a query such as: area > 100 or eccentricity > 0.95')
		hbox_classify.addWidget(self.property_query_le, 70)
		self.submit_query_btn = QPushButton('Submit...')
		self.submit_query_btn.clicked.connect(self.apply_property_query)
		hbox_classify.addWidget(self.submit_query_btn, 20)
		properties_box.addLayout(hbox_classify)

		self.left_panel.addLayout(properties_box)

	def populate_right_panel(self):

		self.right_panel.addWidget(self.fcanvas,70)

		channel_hbox = QHBoxLayout()
		channel_hbox.setContentsMargins(150,30,150,5)
		self.channels_cb = QComboBox()
		self.channels_cb.addItems(self.channel_names)
		self.channels_cb.currentTextChanged.connect(self.reload_frame)
		channel_hbox.addWidget(QLabel('channel: '), 10)
		channel_hbox.addWidget(self.channels_cb,90)
		self.right_panel.addLayout(channel_hbox)

		frame_hbox = QHBoxLayout()
		frame_hbox.setContentsMargins(150,5,150,5)
		self.frame_slider = QLabeledSlider()
		self.frame_slider.setSingleStep(1)
		self.frame_slider.setOrientation(1)
		self.frame_slider.setRange(0,self.len_movie)
		self.frame_slider.setValue(0)
		self.frame_slider.valueChanged.connect(self.reload_frame)	
		frame_hbox.addWidget(QLabel('frame: '), 10)
		frame_hbox.addWidget(self.frame_slider, 90)	
		self.right_panel.addLayout(frame_hbox)

		contrast_hbox = QHBoxLayout()
		contrast_hbox.setContentsMargins(150,5,150,5)
		self.contrast_slider = QLabeledDoubleRangeSlider()
		self.contrast_slider.setSingleStep(0.00001)
		self.contrast_slider.setTickInterval(0.00001)		
		self.contrast_slider.setOrientation(1)
		self.contrast_slider.setRange(np.amin(self.img),np.amax(self.img))
		self.contrast_slider.setValue([np.percentile(self.img.flatten(), 1), np.percentile(self.img.flatten(), 99.99)])
		self.contrast_slider.valueChanged.connect(self.contrast_slider_action)
		contrast_hbox.addWidget(QLabel('contrast: '))
		contrast_hbox.addWidget(self.contrast_slider,90)
		self.right_panel.addLayout(contrast_hbox)


	def locate_stack(self):
		
		"""
		Locate the target movie.

		"""
		print(self.pos)
		movies = glob(self.pos + f"movie/{self.parent.parent.parent.movie_prefix}*.tif")

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
			self.len_movie = self.parent.parent.parent.len_movie
			len_movie_auto = auto_load_number_of_frames(self.stack_path)
			if len_movie_auto is not None:
				self.len_movie = len_movie_auto
			exp_config = self.exp_dir +"config.ini"
			self.channel_names, self.channels = extract_experiment_channels(exp_config)
			self.channel_names = np.array(self.channel_names)
			self.channels = np.array(self.channels)
			self.nbr_channels = len(self.channels)
			self.current_channel = 0
			self.img = load_frames(0, self.stack_path, normalize_input=False)
			print(f'{self.stack_path} successfully located.')

	def show_image(self):

		"""
		Load an image. 

		"""

		self.fig, self.ax = plt.subplots(tight_layout=True)
		self.fcanvas = FigureCanvas(self.fig, interactive=True)
		self.ax.clear()

		self.im = self.ax.imshow(self.img, cmap='gray')

		self.binary = threshold_image(self.img, self.threshold_slider.value()[0], self.threshold_slider.value()[1], foreground_value=255., fill_holes=False)
		self.thresholded_image = np.ma.masked_where(self.binary==0.,self.binary)
		self.image_thresholded = self.ax.imshow(self.thresholded_image, cmap="viridis",alpha=0.5, interpolation='none')

		self.ax.set_xticks([])
		self.ax.set_yticks([])
		self.ax.set_aspect('equal')

		self.fig.set_facecolor('none')  # or 'None'
		self.fig.canvas.setStyleSheet("background-color: black;")
		self.scat_markers = self.ax.scatter([],[],color="tab:red")

		self.fcanvas.canvas.draw()

	def initalize_props_scatter(self):

		"""
		Define properties scatter.
		"""

		self.fig_props, self.ax_props = plt.subplots(tight_layout=True)
		self.propscanvas = FigureCanvas(self.fig_props, interactive=True)
		self.fig_props.set_facecolor('none')
		self.fig_props.canvas.setStyleSheet("background-color: transparent;")
		self.scat_props = self.ax_props.scatter([],[], color='k', alpha=0.75)
		self.propscanvas.canvas.draw_idle()
		self.propscanvas.canvas.setMinimumHeight(self.screen_height//5)



	def initialize_histogram(self):
		
		self.fig_hist, self.ax_hist = plt.subplots(tight_layout=True)
		self.canvas_hist = FigureCanvas(self.fig_hist, interactive=False)
		self.fig_hist.set_facecolor('none')
		self.fig_hist.canvas.setStyleSheet("background-color: transparent;")

		#self.ax_hist.clear()
		#self.ax_hist.cla()
		self.ax_hist.patch.set_facecolor('none')
		self.hist_y, x, _ = self.ax_hist.hist(self.img.flatten(),density=True,bins=300,color="k")
		#self.ax_hist.set_xlim(np.amin(self.img),np.amax(self.img))
		self.ax_hist.set_xlabel('intensity [a.u.]')
		self.ax_hist.spines['top'].set_visible(False)
		self.ax_hist.spines['right'].set_visible(False)
		#self.ax_hist.set_yticks([])
		self.ax_hist.set_xlim(np.amin(self.img),np.amax(self.img))
		self.ax_hist.set_ylim(0, self.hist_y.max())

		self.threshold_slider.setRange(np.amin(self.img), np.amax(self.img))
		self.threshold_slider.setValue([np.nanpercentile(self.img.flatten(), 90), np.amax(self.img)])
		self.add_hist_threshold()

		self.canvas_hist.canvas.draw_idle()	
		self.canvas_hist.canvas.setMinimumHeight(self.screen_height//8)

	def update_histogram(self):
		
		"""
		Redraw the histogram after an update on the image. 
		Move the threshold slider accordingly.

		"""

		self.ax_hist.clear()
		self.ax_hist.patch.set_facecolor('none')
		self.hist_y, x, _ = self.ax_hist.hist(self.img.flatten(),density=True,bins=300,color="k")
		self.ax_hist.set_xlabel('intensity [a.u.]')
		self.ax_hist.spines['top'].set_visible(False)
		self.ax_hist.spines['right'].set_visible(False)
		#self.ax_hist.set_yticks([])
		self.ax_hist.set_xlim(np.amin(self.img),np.amax(self.img))
		self.ax_hist.set_ylim(0, self.hist_y.max())
		self.add_hist_threshold()
		self.canvas_hist.canvas.draw()
		
		self.threshold_slider.setRange(np.amin(self.img), np.amax(self.img))
		self.threshold_slider.setValue([np.nanpercentile(self.img.flatten(), 90), np.amax(self.img)])
		self.threshold_changed(self.threshold_slider.value())

	def add_hist_threshold(self):

		ymin,ymax = self.ax_hist.get_ylim()
		self.min_intensity_line, = self.ax_hist.plot([self.threshold_slider.value()[0],self.threshold_slider.value()[0]],[0,ymax],c="tab:purple")
		self.max_intensity_line, = self.ax_hist.plot([self.threshold_slider.value()[1],self.threshold_slider.value()[1]],[0,ymax],c="tab:purple")
		#self.canvas_hist.canvas.draw_idle()

	def reload_frame(self):

		"""
		Load the frame from the current channel and time choice. Show imshow, update histogram.
		"""
		
		self.clear_post_threshold_options()

		self.current_channel = self.channels_cb.currentIndex()
		t = int(self.frame_slider.value())
		idx = t*self.nbr_channels + self.current_channel
		self.img = load_frames(idx, self.stack_path, normalize_input=False)
		self.refresh_imshow()
		self.update_histogram()
		#self.redo_histogram()

	# def redo_histogram(self):
	# 	self.ax_hist.clear()
	# 	self.canvas_hist.canvas.draw()


	def contrast_slider_action(self):

		"""
		Recontrast the imshow as the contrast slider is moved.
		"""

		self.vmin = self.contrast_slider.value()[0]
		self.vmax = self.contrast_slider.value()[1]
		self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
		self.fcanvas.canvas.draw_idle()

	def refresh_imshow(self):

		"""
		
		Update the imshow based on the current frame selection.

		"""

		self.vmin = np.nanpercentile(self.img.flatten(), 1)
		self.vmax = np.nanpercentile(self.img.flatten(), 99.)

		self.contrast_slider.disconnect()
		self.contrast_slider.setRange(np.amin(self.img), np.amax(self.img))
		self.contrast_slider.setValue([self.vmin, self.vmax])
		self.contrast_slider.valueChanged.connect(self.contrast_slider_action)

		self.im.set_data(self.img)
		self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
		self.fcanvas.canvas.draw_idle()

		#self.initialize_histogram()

	def preprocess_image(self):

		"""
		Reload the frame, apply the filters, update imshow and histogram.

		"""

		self.reload_frame()
		filters = self.filters_qlist.items
		self.img = filter_image(self.img, filters)
		self.refresh_imshow()
		self.update_histogram()


	def threshold_changed(self, value):

		"""
		Move the threshold values on histogram, when slider is moved.
		"""

		self.clear_post_threshold_options()

		self.thresh_min = value[0]
		self.thresh_max = value[1]
		ymin,ymax = self.ax_hist.get_ylim()
		self.min_intensity_line.set_data([self.thresh_min, self.thresh_min],[0,ymax])
		self.max_intensity_line.set_data([self.thresh_max, self.thresh_max], [0,ymax])
		self.canvas_hist.canvas.draw_idle()
		# update imshow threshold
		self.update_threshold()

	def switch_to_log(self):

		"""
		Switch threshold histogram to log scale. Auto adjust.
		"""

		if self.ax_hist.get_yscale()=='linear':
			self.ax_hist.set_yscale('log')
		else:
			self.ax_hist.set_yscale('linear')

		#self.ax_hist.autoscale()
		self.ax_hist.set_ylim(0, self.hist_y.max())
		self.canvas_hist.canvas.draw_idle()		

	def update_threshold(self):

		"""
		
		Threshold and binarize the image based on the min/max threshold values
		and display on imshow.

		"""

		self.binary = threshold_image(self.img, self.threshold_slider.value()[0], self.threshold_slider.value()[1], foreground_value=255., fill_holes=False)
		self.thresholded_image = np.ma.masked_where(self.binary==0.,self.binary)
		self.image_thresholded.set_data(self.thresholded_image)
		self.fcanvas.canvas.draw_idle()

	def set_footprint(self):
		self.footprint = self.footprint_slider.value()
		#print(f"Setting footprint to {self.footprint}")

	def set_min_dist(self):
		self.min_dist = self.min_dist_slider.value()
		#print(f"Setting min distance to {self.min_dist}")

	def detect_markers(self):
		
		self.clear_post_threshold_options()

		if self.binary.ndim==3:
			self.binary = np.squeeze(self.binary)
		self.binary = binary_fill_holes(self.binary)
		self.coords, self.edt_map = identify_markers_from_binary(self.binary, self.min_dist, footprint_size=self.footprint, footprint=None, return_edt=True)
		if len(self.coords)>0:
			self.scat_markers.set_offsets(self.coords[:,[1,0]])
			self.scat_markers.set_visible(True)
			self.fcanvas.canvas.draw()
			self.scat_props.set_visible(True)
			self.watershed_btn.setEnabled(True)
		else:
			self.watershed_btn.setEnabled(False)

	def apply_watershed_to_selection(self):

		self.labels = apply_watershed(self.binary, self.coords, self.edt_map)

		self.current_channel = self.channels_cb.currentIndex()
		t = int(self.frame_slider.value())
		idx = t*self.nbr_channels + self.current_channel
		self.img = load_frames(idx, self.stack_path, normalize_input=False)
		self.refresh_imshow()

		self.image_thresholded.set_cmap('tab20c')
		self.image_thresholded.set_data(np.ma.masked_where(self.labels==0.,self.labels))
		self.image_thresholded.autoscale()
		self.fcanvas.canvas.draw_idle()

		self.compute_features()
		for p in self.properties_box_widgets:
			p.setEnabled(True)

		for i in range(2):
			self.features_cb[i].currentTextChanged.connect(self.update_props_scatter)

	def compute_features(self):

		# Run regionprops to have properties for filtering
		intensity_image_idx = [self.nbr_channels*self.frame_slider.value()]
		for i in range(self.nbr_channels-1):
			intensity_image_idx += [intensity_image_idx[-1]+1]


		# Load channels at time t
		multichannel = load_frames(intensity_image_idx, self.stack_path, normalize_input=False)
		self.props = pd.DataFrame(regionprops_table(self.labels, intensity_image=multichannel, properties=self.cell_properties))
		self.props = rename_intensity_column(self.props, self.channel_names)
		for i in range(2):
			self.features_cb[i].clear()
			self.features_cb[i].addItems(list(self.props.columns))
			self.features_cb[i].setCurrentIndex(i)
		self.props["class"] = 1
		
		self.update_props_scatter()

	def update_props_scatter(self):

		self.scat_props.set_offsets(self.props[[self.features_cb[1].currentText(),self.features_cb[0].currentText()]].to_numpy())
		self.scat_props.set_facecolor([color_from_class(c) for c in self.props['class'].to_numpy()])
		self.ax_props.set_xlabel(self.features_cb[1].currentText())
		self.ax_props.set_ylabel(self.features_cb[0].currentText())
		
		self.scat_markers.set_offsets(self.props[['centroid-1','centroid-0']].to_numpy())
		self.scat_markers.set_color(['k']*len(self.props))
		self.scat_markers.set_facecolor([color_from_class(c) for c in self.props['class'].to_numpy()])
		
		self.ax_props.set_xlim(0.75*self.props[self.features_cb[1].currentText()].min(),1.05*self.props[self.features_cb[1].currentText()].max())
		self.ax_props.set_ylim(0.75*self.props[self.features_cb[0].currentText()].min(),1.05*self.props[self.features_cb[0].currentText()].max())
		self.propscanvas.canvas.draw_idle()
		self.fcanvas.canvas.draw_idle()

	def prep_cell_properties(self):

		self.cell_properties_options = list(np.copy(self.cell_properties))
		self.cell_properties_options.remove("centroid")
		for k in range(self.nbr_channels):
			self.cell_properties_options.append(f'intensity_mean-{k}')
		self.cell_properties_options.remove('intensity_mean')

	def apply_property_query(self):
		query = self.property_query_le.text()
		self.props['class'] = 1

		if query=='':
			print('empty query')
		else:
			try:
				self.selection = self.props.query(query).index
				print(self.selection)
				self.props.loc[self.selection,'class'] = 0
			except Exception as e:
				print(e)
				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Warning)
				msgBox.setText("The query could not be understood. No filtering was applied.")
				msgBox.setWindowTitle("Warning")
				msgBox.setStandardButtons(QMessageBox.Ok)
				returnValue = msgBox.exec()
				if returnValue == QMessageBox.Yes:
					return None

		self.update_props_scatter()

	def clear_post_threshold_options(self):
		
		self.watershed_btn.setEnabled(False)

		for p in self.properties_box_widgets:
			p.setEnabled(False)


		for i in range(2):
			try:
				self.features_cb[i].disconnect()
			except:
				pass
			self.features_cb[i].clear()

		self.property_query_le.setText('')

		self.binary = threshold_image(self.img, self.threshold_slider.value()[0], self.threshold_slider.value()[1], foreground_value=255., fill_holes=False)
		self.thresholded_image = np.ma.masked_where(self.binary==0.,self.binary)

		self.scat_markers.set_color('tab:red')
		self.scat_markers.set_visible(False)
		self.image_thresholded.set_data(self.thresholded_image)
		self.image_thresholded.set_cmap('viridis')
		self.image_thresholded.autoscale()

	def write_instructions(self):
		
		instructions = {
						"target_channel": self.channels_cb.currentText(), #for now index but would be more universal to use name
						"thresholds": self.threshold_slider.value(),
						"filters": self.filters_qlist.items,
						"marker_min_distance": self.min_dist,
						"marker_footprint_size": self.footprint,
						"feature_queries": [self.property_query_le.text()],
						"equalize_reference": [self.equalize_option, self.frame_slider.value()],
						}

		print('The following instructions will be written: ', instructions)
		self.instruction_file = QFileDialog.getSaveFileName(self, "Save File", self.exp_dir+f'configs/threshold_config_{self.mode}.json', '.json')[0]
		if os.path.exists(self.instruction_file) and self.instruction_file!='':
			json_object = json.dumps(instructions, indent=4)
			with open(self.instruction_file, "w") as outfile:
				outfile.write(json_object)
			print("Configuration successfully written in ",self.instruction_file)

			self.parent.filename = self.instruction_file
			self.parent.file_label.setText(self.instruction_file)

			self.close()

	def activate_histogram_equalizer(self):

		if not self.equalize_option:
			self.equalize_option = True
			self.equalize_option_btn.setIcon(icon(MDI6.equalizer,color="#1f77b4"))
			self.equalize_option_btn.setIconSize(QSize(20,20))
		else:
			self.equalize_option = False
			self.equalize_option_btn.setIcon(icon(MDI6.equalizer,color="black"))
			self.equalize_option_btn.setIconSize(QSize(20,20))


