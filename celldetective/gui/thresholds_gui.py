from PyQt5.QtWidgets import QMainWindow, QLabel, QWidget, QHBoxLayout, QGridLayout, QScrollArea, QVBoxLayout, QComboBox, QPushButton, QApplication, QPushButton
from celldetective.gui.gui_utils import center_window, FigureCanvas, ListWidget, FilterChoice
from celldetective.utils import get_software_location, extract_experiment_channels
from celldetective.io import auto_load_number_of_frames, load_frames
from PyQt5.QtCore import Qt, QSize
from glob import glob
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import numpy as np
import matplotlib.pyplot as plt
from superqt import QLabeledSlider, QLabeledDoubleRangeSlider
from celldetective.segmentation import filter_image

class ThresholdConfigWizard(QMainWindow):
	
	"""
	UI to create a threshold pipeline for segmentation.

	"""

	def __init__(self, parent=None):
		
		super().__init__()
		self.parent = parent
		self.setWindowTitle("Threshold configuration wizard")
		self.mode = self.parent.mode
		self.pos = self.parent.parent.parent.pos
		self.exp_dir = self.parent.parent.exp_dir
		self.soft_path = get_software_location()
		if self.mode=="targets":
			self.config_out_name = "threshold_targets.json"
		elif self.mode=="effectors":
			self.config_out_name = "threshold_effectors.json"

		self.screen_height = self.parent.parent.parent.parent.screen_height
		self.screen_width = self.parent.parent.parent.parent.screen_width

		self.locate_stack()
		self.show_image()
		self.populate_widget()

		self.setMinimumWidth(int(0.8*self.screen_width))
		self.setMinimumHeight(int(0.8*self.screen_height))

		center_window(self)
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
		self.left_panel = QVBoxLayout()
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

		self.scroll_area.setAlignment(Qt.AlignCenter)
		self.scroll_area.setLayout(self.left_panel)
		self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		self.scroll_area.setWidgetResizable(True)

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
		#self.ylog_check.setIcon(QIcon_from_svg("/home/limozin/Documents/GitHub/ADCCFactory/adccfactory/icons/log.svg", color='black'))
		self.ylog_check.setStyleSheet(self.parent.parent.parent.parent.button_select_all)
		#self.ylog_check.clicked.connect(self.switch_to_log)
		self.log_hist = False
		threshold_title_grid.addWidget(self.ylog_check, 5)
		
		self.equalize_option_btn = QPushButton("")
		self.equalize_option_btn.setIcon(icon(MDI6.equalizer,color="black"))
		self.equalize_option_btn.setIconSize(QSize(20,20))
		self.equalize_option_btn.setStyleSheet(self.parent.parent.parent.parent.button_select_all)
		self.equalize_option_btn.setToolTip("Enable histogram matching")
		#self.equalize_option_btn.clicked.connect(self.activate_histogram_equalizer)
		self.equalize_option = False
		threshold_title_grid.addWidget(self.equalize_option_btn, 5)
		
		grid_threshold.addLayout(threshold_title_grid, idx, 0,1,2)


		idx+=1

		# Slider to set vmin & vmax
		self.threshold_contrast_range = QLabeledDoubleRangeSlider()
		self.threshold_contrast_range.setSingleStep(0.00001)
		self.threshold_contrast_range.setTickInterval(0.00001)
		self.threshold_contrast_range.setOrientation(1)
		self.threshold_contrast_range.setRange(np.amin(self.img), np.amax(self.img))
		self.threshold_contrast_range.setValue([np.percentile(self.img.flatten(), 90), np.amax(self.img)])
		#self.threshold_contrast_range.valueChanged.connect(self.threshold_changed)
		self.make_histogram()


		grid_threshold.addWidget(self.canvas_hist, idx,0,1,3)

		idx+=1

	#self.threshold_contrast_range.valueChanged.connect(self.set_clim_thresh)

		grid_threshold.addWidget(self.threshold_contrast_range,idx,1,1,1)

		self.set_max_intensity = QPushButton("")
		self.set_max_intensity.setIcon(icon(MDI6.page_last,color="black"))
		self.set_max_intensity.setIconSize(QSize(20,20))
		self.set_max_intensity.setStyleSheet(self.parent.parent.parent.parent.button_select_all)
		#self.set_max_intensity.clicked.connect(self.set_upper_threshold_to_max)
		grid_threshold.addWidget(self.set_max_intensity, idx,2,1,1,alignment=Qt.AlignRight)

		self.set_min_intensity = QPushButton("")
		self.set_min_intensity.setIcon(icon(MDI6.page_first,color="black"))
		self.set_min_intensity.setIconSize(QSize(20,20))
		self.set_min_intensity.setStyleSheet(self.parent.parent.parent.parent.button_select_all)
		#self.set_min_intensity.clicked.connect(self.set_lower_threshold_to_min)
		grid_threshold.addWidget(self.set_min_intensity, idx,0,1,1,alignment=Qt.AlignRight)

		self.canvas_hist.setMinimumHeight(self.screen_height//8)
		self.left_panel.addLayout(grid_threshold)

		#################
		# FINAL SAVE BTN#
		#################

		self.save_btn = QPushButton('Save')
		self.save_btn.setStyleSheet(self.parent.parent.parent.parent.button_style_sheet)
		#self.submit_btn.clicked.connect(self.write_instructions)
		self.left_panel.addWidget(self.save_btn)


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

		self.ax.set_xticks([])
		self.ax.set_yticks([])
		self.ax.set_aspect('equal')

		self.fig.set_facecolor('none')  # or 'None'
		self.fig.canvas.setStyleSheet("background-color: black;")

		self.fcanvas.canvas.draw()

	def make_histogram(self):

		self.fig_hist, self.ax_hist = plt.subplots(tight_layout=True)
		self.canvas_hist = FigureCanvas(self.fig_hist, interactive=False)
		self.ax_hist.clear()
		self.ax_hist.patch.set_facecolor('none')
		self.ax_hist.hist(self.img.flatten(),density=True,bins=300,color="k")
		self.fig_hist.set_facecolor('none')
		self.fig_hist.canvas.setStyleSheet("background-color: transparent;")

		self.ax_hist.set_xlabel('intensity [a.u.]')
		self.ax_hist.spines['top'].set_visible(False)
		self.ax_hist.spines['right'].set_visible(False)
		self.ax_hist.set_yticks([])
		
		ymin,ymax = self.ax_hist.get_ylim()
		self.min_intensity_line, = self.ax_hist.plot([self.threshold_contrast_range.value()[0],self.threshold_contrast_range.value()[0]],[ymin,ymax],c="k")
		self.max_intensity_line, = self.ax_hist.plot([self.threshold_contrast_range.value()[1],self.threshold_contrast_range.value()[1]],[ymin,ymax],c="k")

		self.canvas_hist.canvas.draw()		

	def reload_frame(self):
		
		self.current_channel = self.channels_cb.currentIndex()
		t = int(self.frame_slider.value())
		idx = t*self.nbr_channels + self.current_channel
		self.img = load_frames(idx, self.stack_path, normalize_input=False)
		self.refresh_imshow()
		self.redo_histogram()

	def redo_histogram(self):
		self.ax_hist.clear()
		self.canvas_hist.canvas.draw()


	def contrast_slider_action(self):

		self.vmin = self.contrast_slider.value()[0]
		self.vmax = self.contrast_slider.value()[1]
		self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
		self.fcanvas.canvas.draw_idle()

	def refresh_imshow(self):

		self.vmin = np.nanpercentile(self.img.flatten(), 1)
		self.vmax = np.nanpercentile(self.img.flatten(), 99.99)

		self.contrast_slider.disconnect()
		self.contrast_slider.setRange(np.amin(self.img), np.amax(self.img))
		self.contrast_slider.setValue([self.vmin, self.vmax])
		self.contrast_slider.valueChanged.connect(self.contrast_slider_action)

		#self.threshold_contrast_range.disconnect()
		self.threshold_contrast_range.setRange(np.amin(self.img), np.amax(self.img))
		self.threshold_contrast_range.setValue([np.nanpercentile(self.img.flatten(), 90), np.amax(self.img)])
		#self.threshold_contrast_range.valueChanged.connect(self.threshold_changed)
		#self.threshold_changed()

		self.im.set_data(self.img)
		self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
		self.fcanvas.canvas.draw_idle()

	def preprocess_image(self):

		self.reload_frame()
		filters = self.filters_qlist.items
		self.img = filter_image(self.img, filters)
		self.refresh_imshow()
		#self.make_histogram()

	# def threshold_changed(self):

	# 	min_value = self.threshold_contrast_range.value()[0]
	# 	max_value = self.threshold_contrast_range.value()[1]
	# 	self.min_intensity_line.set_xdata([min_value, min_value])
	# 	self.max_intensity_line.set_xdata([max_value,max_value])
	# 	self.canvas_hist.canvas.draw()

