import math

import skimage
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QScrollArea, QComboBox, QFrame, QCheckBox, \
    QFileDialog, QGridLayout, QTextEdit, QLineEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton, QTabWidget, \
    QRadioButton, QButtonGroup, QSizePolicy, QListWidget, QDialog
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from matplotlib.patches import Circle
from scipy import ndimage
from skimage.draw import disk
from skimage.morphology import disk

from celldetective.filters import std_filter, gauss_filter
from celldetective.gui.gui_utils import center_window, FeatureChoice, ListWidget, QHSeperationLine, FigureCanvas, \
    GeometryChoice, OperationChoice, ChannelChoice
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider, QLabeledSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6

from celldetective.gui.thresholds_gui import ThresholdNormalisation, ThresholdSpot
from celldetective.utils import extract_experiment_channels, get_software_location
from celldetective.io import interpret_tracking_configuration, load_frames, auto_load_number_of_frames
from celldetective.measure import compute_haralick_features, contour_of_instance_segmentation, correct_image, \
    field_normalisation, normalise_by_cell
import numpy as np
from tifffile import imread
import json
from shutil import copyfile
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
from natsort import natsorted
from tifffile import imread
from pathlib import Path, PurePath
import gc
from stardist import fill_label_holes


class ConfigMeasurements(QMainWindow):
    """
	UI to set measurement instructions.

	"""

    def __init__(self, parent=None):

        super().__init__()
        self.parent = parent
        self.setWindowTitle("Configure measurements")
        self.setWindowIcon(QIcon(os.sep.join(['celldetective', 'icons', 'mexican-hat.png'])))
        self.mode = self.parent.mode
        self.exp_dir = self.parent.exp_dir
        self.background_correction = []
        if self.mode == "targets":
            self.config_name = "btrack_config_targets.json"
            self.measure_instructions_path = self.parent.exp_dir + "configs/measurement_instructions_targets.json"
        elif self.mode == "effectors":
            self.config_name = "btrack_config_effectors.json"
            self.measure_instructions_path = self.parent.exp_dir + "configs/measurement_instructions_effectors.json"
        self.soft_path = get_software_location()
        self.clear_previous = False

        exp_config = self.exp_dir + "config.ini"
        self.config_path = self.exp_dir + self.config_name
        self.channel_names, self.channels = extract_experiment_channels(exp_config)
        self.channel_names = np.array(self.channel_names)
        self.channels = np.array(self.channels)

        self.screen_height = self.parent.parent.parent.screen_height
        center_window(self)

        self.setMinimumWidth(500)
        self.setMinimumHeight(int(0.3 * self.screen_height))
        self.setMaximumHeight(int(0.8 * self.screen_height))
        self.populate_widget()
        self.load_previous_measurement_instructions()

    def populate_widget(self):

        """
		Create the multibox design.

		"""

        # Create button widget and layout
        self.scroll_area = QScrollArea(self)
        self.button_widget = QWidget()
        main_layout = QVBoxLayout()
        self.button_widget.setLayout(main_layout)
        main_layout.setContentsMargins(30, 30, 30, 30)

        self.normalisation_frame = QFrame()
        self.normalisation_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.populate_normalisation_tabs()
        main_layout.addWidget(self.normalisation_frame)

        # first frame for FEATURES
        self.features_frame = QFrame()
        self.features_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.populate_features_frame()
        main_layout.addWidget(self.features_frame)

        # second frame for ISOTROPIC MEASUREMENTS
        self.iso_frame = QFrame()
        self.iso_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.populate_iso_frame()
        main_layout.addWidget(self.iso_frame)

        self.spot_detection_frame = QFrame()
        self.spot_detection_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.populate_spot_detection()
        main_layout.addWidget(self.spot_detection_frame)

        self.clear_previous_btn = QCheckBox('clear previous measurements')
        main_layout.addWidget(self.clear_previous_btn)

        self.submit_btn = QPushButton('Save')
        self.submit_btn.setStyleSheet(self.parent.parent.parent.button_style_sheet)
        self.submit_btn.clicked.connect(self.write_instructions)
        main_layout.addWidget(self.submit_btn)

        # self.populate_left_panel()
        # grid.addLayout(self.left_side, 0, 0, 1, 1)
        self.button_widget.adjustSize()

        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.button_widget)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setWidgetResizable(True)
        self.setCentralWidget(self.scroll_area)
        self.show()

        QApplication.processEvents()
        self.adjustScrollArea()

    def populate_iso_frame(self):

        """
		Add widgets and layout in the POST-PROCESSING frame.
		"""

        grid = QGridLayout(self.iso_frame)

        self.iso_lbl = QLabel("ISOTROPIC MEASUREMENTS")
        self.iso_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
        grid.addWidget(self.iso_lbl, 0, 0, 1, 4, alignment=Qt.AlignCenter)
        self.generate_iso_contents()
        grid.addWidget(self.ContentsIso, 1, 0, 1, 4, alignment=Qt.AlignTop)

    def populate_features_frame(self):

        """
		Add widgets and layout in the FEATURES frame.
		"""

        grid = QGridLayout(self.features_frame)

        self.feature_lbl = QLabel("FEATURES")
        self.feature_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
        grid.addWidget(self.feature_lbl, 0, 0, 1, 4, alignment=Qt.AlignCenter)

        self.generate_feature_panel_contents()
        grid.addWidget(self.ContentsFeatures, 1, 0, 1, 4, alignment=Qt.AlignTop)

    def generate_iso_contents(self):

        self.ContentsIso = QFrame()
        layout = QVBoxLayout(self.ContentsIso)
        layout.setContentsMargins(0, 0, 0, 0)

        radii_layout = QHBoxLayout()
        self.radii_lbl = QLabel('Measurement radii (from center):')
        self.radii_lbl.setToolTip('Define radii or donughts for intensity measurements.')
        radii_layout.addWidget(self.radii_lbl, 90)

        self.del_radius_btn = QPushButton("")
        self.del_radius_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
        self.del_radius_btn.setIcon(icon(MDI6.trash_can, color="black"))
        self.del_radius_btn.setToolTip("Remove radius")
        self.del_radius_btn.setIconSize(QSize(20, 20))
        radii_layout.addWidget(self.del_radius_btn, 5)

        self.add_radius_btn = QPushButton("")
        self.add_radius_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
        self.add_radius_btn.setIcon(icon(MDI6.plus, color="black"))
        self.add_radius_btn.setToolTip("Add radius")
        self.add_radius_btn.setIconSize(QSize(20, 20))
        radii_layout.addWidget(self.add_radius_btn, 5)
        layout.addLayout(radii_layout)

        self.radii_list = ListWidget(self, GeometryChoice, initial_features=["10"], dtype=int)
        layout.addWidget(self.radii_list)

        self.del_radius_btn.clicked.connect(self.radii_list.removeSel)
        self.add_radius_btn.clicked.connect(self.radii_list.addItem)

        # Operation
        operation_layout = QHBoxLayout()
        self.op_lbl = QLabel('Operation to perform:')
        self.op_lbl.setToolTip('Set the operations to perform inside the ROI.')
        operation_layout.addWidget(self.op_lbl, 90)

        self.del_op_btn = QPushButton("")
        self.del_op_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
        self.del_op_btn.setIcon(icon(MDI6.trash_can, color="black"))
        self.del_op_btn.setToolTip("Remove operation")
        self.del_op_btn.setIconSize(QSize(20, 20))
        operation_layout.addWidget(self.del_op_btn, 5)

        self.add_op_btn = QPushButton("")
        self.add_op_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
        self.add_op_btn.setIcon(icon(MDI6.plus, color="black"))
        self.add_op_btn.setToolTip("Add operation")
        self.add_op_btn.setIconSize(QSize(20, 20))
        operation_layout.addWidget(self.add_op_btn, 5)
        layout.addLayout(operation_layout)

        self.operations_list = ListWidget(self, OperationChoice, initial_features=["mean"])
        layout.addWidget(self.operations_list)

        self.del_op_btn.clicked.connect(self.operations_list.removeSel)
        self.add_op_btn.clicked.connect(self.operations_list.addItem)

    def generate_feature_panel_contents(self):

        self.ContentsFeatures = QFrame()
        layout = QVBoxLayout(self.ContentsFeatures)
        layout.setContentsMargins(0, 0, 0, 0)

        feature_layout = QHBoxLayout()
        feature_layout.setContentsMargins(0, 0, 0, 0)

        self.feature_lbl = QLabel("Add features:")
        self.del_feature_btn = QPushButton("")
        self.del_feature_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
        self.del_feature_btn.setIcon(icon(MDI6.trash_can, color="black"))
        self.del_feature_btn.setToolTip("Remove feature")
        self.del_feature_btn.setIconSize(QSize(20, 20))

        self.add_feature_btn = QPushButton("")
        self.add_feature_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
        self.add_feature_btn.setIcon(icon(MDI6.filter_plus, color="black"))
        self.add_feature_btn.setToolTip("Add feature")
        self.add_feature_btn.setIconSize(QSize(20, 20))

        self.features_list = ListWidget(self, FeatureChoice, initial_features=['area', 'intensity_mean', ])

        self.del_feature_btn.clicked.connect(self.features_list.removeSel)
        self.add_feature_btn.clicked.connect(self.features_list.addItem)

        feature_layout.addWidget(self.feature_lbl, 90)
        feature_layout.addWidget(self.del_feature_btn, 5)
        feature_layout.addWidget(self.add_feature_btn, 5)
        layout.addLayout(feature_layout)
        layout.addWidget(self.features_list)

        self.feat_sep2 = QHSeperationLine()
        layout.addWidget(self.feat_sep2)

        contour_layout = QHBoxLayout()
        self.border_dist_lbl = QLabel('Contour measurements (from edge of mask):')
        self.border_dist_lbl.setToolTip(
            'Apply the intensity measurements defined above\nto a slice of each cell mask, defined as distance\nfrom the edge.')
        contour_layout.addWidget(self.border_dist_lbl, 90)

        self.del_contour_btn = QPushButton("")
        self.del_contour_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
        self.del_contour_btn.setIcon(icon(MDI6.trash_can, color="black"))
        self.del_contour_btn.setToolTip("Remove distance")
        self.del_contour_btn.setIconSize(QSize(20, 20))
        contour_layout.addWidget(self.del_contour_btn, 5)

        self.add_contour_btn = QPushButton("")
        self.add_contour_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
        self.add_contour_btn.setIcon(icon(MDI6.plus, color="black"))
        self.add_contour_btn.setToolTip("Add distance")
        self.add_contour_btn.setIconSize(QSize(20, 20))
        contour_layout.addWidget(self.add_contour_btn, 5)

        self.view_contour_btn = QPushButton("")
        self.view_contour_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
        self.view_contour_btn.setIcon(icon(MDI6.eye_outline, color="black"))
        self.view_contour_btn.setToolTip("View contour")
        self.view_contour_btn.setIconSize(QSize(20, 20))
        contour_layout.addWidget(self.view_contour_btn, 5)

        layout.addLayout(contour_layout)

        self.contours_list = ListWidget(self, GeometryChoice, initial_features=[], dtype=int)
        layout.addWidget(self.contours_list)

        self.del_contour_btn.clicked.connect(self.contours_list.removeSel)
        self.add_contour_btn.clicked.connect(self.contours_list.addItem)
        self.view_contour_btn.clicked.connect(self.view_selected_contour)

        self.feat_sep3 = QHSeperationLine()
        layout.addWidget(self.feat_sep3)
        # self.radial_intensity_btn = QCheckBox('Measure radial intensity distribution')
        # layout.addWidget(self.radial_intensity_btn)
        # self.radial_intensity_btn.clicked.connect(self.enable_step_size)
        # self.channel_chechkboxes=[]
        # for channel in self.channel_names:
        #     channel_checkbox=QCheckBox(channel)
        #     self.channel_chechkboxes.append(channel_checkbox)
        #     layout.addWidget(channel_checkbox)
        #     channel_checkbox.setEnabled(False)
        # step_box=QHBoxLayout()
        # self.step_lbl=QLabel("Step size (in px)")
        # self.step_size=QLineEdit()
        # self.step_lbl.setEnabled(False)
        # self.step_size.setEnabled(False)
        # step_box.addWidget(self.step_lbl)
        # step_box.addWidget(self.step_size)
        # layout.addLayout(step_box)
        # self.feat_sep4 = QHSeperationLine()
        # layout.addWidget(self.feat_sep4)

        # Haralick features parameters
        self.activate_haralick_btn = QCheckBox('Measure Haralick texture features')
        self.activate_haralick_btn.toggled.connect(self.show_haralick_options)

        self.haralick_channel_choice = QComboBox()
        self.haralick_channel_choice.addItems(self.channel_names)
        self.haralick_channel_lbl = QLabel('Target channel: ')

        self.haralick_distance_le = QLineEdit("1")
        self.haralick_distance_lbl = QLabel('Distance: ')

        self.haralick_n_gray_levels_le = QLineEdit("256")
        self.haralick_n_gray_levels_lbl = QLabel('# gray levels: ')

        # Slider to set vmin & vmax
        self.haralick_scale_slider = QLabeledDoubleSlider()
        self.haralick_scale_slider.setSingleStep(0.05)
        self.haralick_scale_slider.setTickInterval(0.05)
        self.haralick_scale_slider.setSingleStep(1)
        self.haralick_scale_slider.setOrientation(1)
        self.haralick_scale_slider.setRange(0, 1)
        self.haralick_scale_slider.setValue(0.5)
        self.haralick_scale_lbl = QLabel('Scale: ')

        self.haralick_percentile_min_le = QLineEdit('0.01')
        self.haralick_percentile_max_le = QLineEdit('99.9')
        self.haralick_normalization_mode_btn = QPushButton()
        self.haralick_normalization_mode_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
        self.haralick_normalization_mode_btn.setIcon(icon(MDI6.percent_circle, color="black"))
        self.haralick_normalization_mode_btn.setIconSize(QSize(20, 20))
        self.haralick_normalization_mode_btn.setToolTip("Switch to absolute normalization values.")
        self.percentile_mode = True

        min_percentile_hbox = QHBoxLayout()
        min_percentile_hbox.addWidget(self.haralick_percentile_min_le, 90)
        min_percentile_hbox.addWidget(self.haralick_normalization_mode_btn, 10)
        min_percentile_hbox.setContentsMargins(0, 0, 0, 0)

        self.haralick_percentile_min_lbl = QLabel('Min percentile: ')
        self.haralick_percentile_max_lbl = QLabel('Max percentile: ')

        self.haralick_hist_btn = QPushButton()
        self.haralick_hist_btn.clicked.connect(self.control_haralick_intensity_histogram)
        self.haralick_hist_btn.setIcon(icon(MDI6.poll, color="k"))
        self.haralick_hist_btn.setStyleSheet(self.parent.parent.parent.button_select_all)

        self.haralick_digit_btn = QPushButton()
        self.haralick_digit_btn.clicked.connect(self.control_haralick_digitalization)
        self.haralick_digit_btn.setIcon(icon(MDI6.image_check, color="k"))
        self.haralick_digit_btn.setStyleSheet(self.parent.parent.parent.button_select_all)

        self.haralick_layout = QVBoxLayout()
        self.haralick_layout.setContentsMargins(20, 20, 20, 20)

        activate_layout = QHBoxLayout()
        activate_layout.addWidget(self.activate_haralick_btn, 80)
        activate_layout.addWidget(self.haralick_hist_btn, 10)
        activate_layout.addWidget(self.haralick_digit_btn, 10)
        self.haralick_layout.addLayout(activate_layout)

        channel_layout = QHBoxLayout()
        channel_layout.addWidget(self.haralick_channel_lbl, 40)
        channel_layout.addWidget(self.haralick_channel_choice, 60)
        self.haralick_layout.addLayout(channel_layout)

        distance_layout = QHBoxLayout()
        distance_layout.addWidget(self.haralick_distance_lbl, 40)
        distance_layout.addWidget(self.haralick_distance_le, 60)
        self.haralick_layout.addLayout(distance_layout)

        gl_layout = QHBoxLayout()
        gl_layout.addWidget(self.haralick_n_gray_levels_lbl, 40)
        gl_layout.addWidget(self.haralick_n_gray_levels_le, 60)
        self.haralick_layout.addLayout(gl_layout)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.haralick_scale_lbl, 40)
        slider_layout.addWidget(self.haralick_scale_slider, 60)
        self.haralick_layout.addLayout(slider_layout)

        slider_min_percentile_layout = QHBoxLayout()
        slider_min_percentile_layout.addWidget(self.haralick_percentile_min_lbl, 40)
        # slider_min_percentile_layout.addWidget(self.haralick_percentile_min_le,55)
        slider_min_percentile_layout.addLayout(min_percentile_hbox, 60)
        # slider_min_percentile_layout.addWidget(self.haralick_normalization_mode_btn, 5)
        self.haralick_layout.addLayout(slider_min_percentile_layout)

        slider_max_percentile_layout = QHBoxLayout()
        slider_max_percentile_layout.addWidget(self.haralick_percentile_max_lbl, 40)
        slider_max_percentile_layout.addWidget(self.haralick_percentile_max_le, 60)
        self.haralick_layout.addLayout(slider_max_percentile_layout)

        self.haralick_to_hide = [self.haralick_hist_btn, self.haralick_digit_btn, self.haralick_channel_lbl,
                                 self.haralick_channel_choice,
                                 self.haralick_distance_le, self.haralick_distance_lbl, self.haralick_n_gray_levels_le,
                                 self.haralick_n_gray_levels_lbl,
                                 self.haralick_scale_lbl, self.haralick_scale_slider, self.haralick_percentile_min_lbl,
                                 self.haralick_percentile_min_le,
                                 self.haralick_percentile_max_lbl, self.haralick_percentile_max_le,
                                 self.haralick_normalization_mode_btn]

        self.features_to_disable = [self.feature_lbl, self.del_feature_btn, self.add_feature_btn, self.features_list,
                                    self.activate_haralick_btn]

        self.activate_haralick_btn.setChecked(False)
        for f in self.haralick_to_hide:
            f.setEnabled(False)

        self.haralick_normalization_mode_btn.clicked.connect(self.switch_to_absolute_normalization_mode)
        layout.addLayout(self.haralick_layout)

    def switch_to_absolute_normalization_mode(self):

        if self.percentile_mode:
            self.percentile_mode = False
            self.haralick_normalization_mode_btn.setIcon(icon(MDI6.percent_circle_outline, color="black"))
            self.haralick_normalization_mode_btn.setIconSize(QSize(20, 20))
            self.haralick_normalization_mode_btn.setToolTip("Switch to percentile normalization values.")
            self.haralick_percentile_min_lbl.setText('Min value: ')
            self.haralick_percentile_max_lbl.setText('Max value: ')
            self.haralick_percentile_min_le.setText('0')
            self.haralick_percentile_max_le.setText('10000')

        else:
            self.percentile_mode = True
            self.haralick_normalization_mode_btn.setIcon(icon(MDI6.percent_circle, color="black"))
            self.haralick_normalization_mode_btn.setIconSize(QSize(20, 20))
            self.haralick_normalization_mode_btn.setToolTip("Switch to absolute normalization values.")
            self.haralick_percentile_min_lbl.setText('Min percentile: ')
            self.haralick_percentile_max_lbl.setText('Max percentile: ')
            self.haralick_percentile_min_le.setText('0.01')
            self.haralick_percentile_max_le.setText('99.99')

    def show_haralick_options(self):

        """
		Show the Haralick texture options.
		"""

        if self.activate_haralick_btn.isChecked():
            for element in self.haralick_to_hide:
                element.setEnabled(True)
        else:
            for element in self.haralick_to_hide:
                element.setEnabled(False)

    def adjustScrollArea(self):

        """
		Auto-adjust scroll area to fill space
		(from https://stackoverflow.com/questions/66417576/make-qscrollarea-use-all-available-space-of-qmainwindow-height-axis)
		"""

        step = 5
        while self.scroll_area.verticalScrollBar().isVisible() and self.height() < self.maximumHeight():
            self.resize(self.width(), self.height() + step)

    def write_instructions(self):

        """
		Write the selected options in a json file for later reading by the software.
		"""

        print('Writing instructions...')
        measurement_options = {}
        background_correction = self.background_correction
        if not background_correction:
            background_correction = None
        measurement_options.update({'background_correction': background_correction})
        features = self.features_list.getItems()
        if not features:
            features = None
        measurement_options.update({'features': features})

        border_distances = self.contours_list.getItems()
        if not border_distances:
            border_distances = None
        measurement_options.update({'border_distances': border_distances})
        # radial_intensity = {}
        # radial_step = int(self.step_size.text())
        # radial_channels = []
        # for checkbox in self.channel_chechkboxes:
        #     if checkbox.isChecked():
        #         radial_channels.append(checkbox.text())
        # radial_intensity={'radial_step': radial_step, 'radial_channels': radial_channels}
        # if not self.radial_intensity_btn.isChecked():
        #     radial_intensity = None
        # measurement_options.update({'radial_intensity' : radial_intensity})

        self.extract_haralick_options()
        measurement_options.update({'haralick_options': self.haralick_options})

        intensity_measurement_radii = self.radii_list.getItems()
        if not intensity_measurement_radii:
            intensity_measurement_radii = None

        isotropic_operations = self.operations_list.getItems()
        if not isotropic_operations:
            isotropic_operations = None
            intensity_measurement_radii = None
        measurement_options.update({'intensity_measurement_radii': intensity_measurement_radii,
                                    'isotropic_operations': isotropic_operations})
        spot_detection = None
        if self.spot_check.isChecked():
            spot_detection = {'channel': self.spot_channel.currentText(), 'diameter': float(self.diameter_value.text()),
                              'threshold': float(self.threshold_value.text())}
        measurement_options.update({'spot_detection': spot_detection})
        if self.clear_previous_btn.isChecked():
            self.clear_previous = True
        else:
            self.clear_previous = False
        measurement_options.update({'clear_previous': self.clear_previous})



        print('Measurement instructions: ', measurement_options)
        file_name = self.measure_instructions_path
        with open(file_name, 'w') as f:
            json.dump(measurement_options, f, indent=4)

        print('Done.')
        self.close()

    def extract_haralick_options(self):

        if self.activate_haralick_btn.isChecked():
            self.haralick_options = {"target_channel": self.haralick_channel_choice.currentIndex(),
                                     "scale_factor": float(self.haralick_scale_slider.value()),
                                     "n_intensity_bins": int(self.haralick_n_gray_levels_le.text()),
                                     "distance": int(self.haralick_distance_le.text()),
                                     }
            if self.percentile_mode:
                self.haralick_options.update({"percentiles": (
                    float(self.haralick_percentile_min_le.text()), float(self.haralick_percentile_max_le.text())),
                    "clip_values": None})
            else:
                self.haralick_options.update({"percentiles": None, "clip_values": (
                    float(self.haralick_percentile_min_le.text()), float(self.haralick_percentile_max_le.text()))})

        else:
            self.haralick_options = None

    def load_previous_measurement_instructions(self):

        """
		Read the measurmeent options from a previously written json file and format properly for the UI.
		"""

        print('Reading instructions..')
        if os.path.exists(self.measure_instructions_path):
            with open(self.measure_instructions_path, 'r') as f:
                measurement_instructions = json.load(f)
                print(measurement_instructions)
                if 'background_correction' in measurement_instructions:
                    self.background_correction = measurement_instructions['background_correction']
                    if (self.background_correction is not None) and len(self.background_correction) > 0:
                        self.normalisation_list.clear()
                        for norm_params in self.background_correction:
                            normalisation_description = ""
                            for index, (key, value) in enumerate(norm_params.items()):
                                if index > 0:
                                    normalisation_description += ", "
                                normalisation_description += str(key) + " : " + str(value)
                            self.normalisation_list.addItem(normalisation_description)
                    else:
                        self.normalisation_list.clear()
                if 'features' in measurement_instructions:
                    features = measurement_instructions['features']
                    if (features is not None) and len(features) > 0:
                        self.features_list.list_widget.clear()
                        self.features_list.list_widget.addItems(features)
                    else:
                        self.features_list.list_widget.clear()

                if 'spot_detection' in measurement_instructions:
                    spot_detection = measurement_instructions['spot_detection']
                    if spot_detection is not None:
                        self.spot_check.setChecked(True)
                        if 'channel' in spot_detection:
                            idx = spot_detection['channel']
                            self.spot_channel.setCurrentText(idx)
                        self.diameter_value.setText(str(spot_detection['diameter']))
                        self.threshold_value.setText(str(spot_detection['threshold']))


                if 'border_distances' in measurement_instructions:
                    border_distances = measurement_instructions['border_distances']
                    if border_distances is not None:
                        if isinstance(border_distances, int):
                            distances = [border_distances]
                        elif isinstance(border_distances, list):
                            distances = []
                            for d in border_distances:
                                if isinstance(d, int) | isinstance(d, float):
                                    distances.append(str(int(d)))
                                elif isinstance(d, list):
                                    distances.append(str(int(d[0])) + '-' + str(int(d[1])))
                        self.contours_list.list_widget.clear()
                        self.contours_list.list_widget.addItems(distances)

                if 'haralick_options' in measurement_instructions:
                    haralick_options = measurement_instructions['haralick_options']
                    if haralick_options is None:
                        self.activate_haralick_btn.setChecked(False)
                        self.show_haralick_options()
                    else:
                        self.activate_haralick_btn.setChecked(True)
                        self.show_haralick_options()
                        if 'target_channel' in haralick_options:
                            idx = haralick_options['target_channel']
                            self.haralick_channel_choice.setCurrentIndex(idx)
                        if 'scale_factor' in haralick_options:
                            self.haralick_scale_slider.setValue(float(haralick_options['scale_factor']))
                        if ('percentiles' in haralick_options) and (haralick_options['percentiles'] is not None):
                            perc = list(haralick_options['percentiles'])
                            self.haralick_percentile_min_le.setText(str(perc[0]))
                            self.haralick_percentile_max_le.setText(str(perc[1]))
                        if ('clip_values' in haralick_options) and (haralick_options['clip_values'] is not None):
                            values = list(haralick_options['clip_values'])
                            self.haralick_percentile_min_le.setText(str(values[0]))
                            self.haralick_percentile_max_le.setText(str(values[1]))
                            self.percentile_mode = True
                            self.switch_to_absolute_normalization_mode()
                        if 'n_intensity_bins' in haralick_options:
                            self.haralick_n_gray_levels_le.setText(str(haralick_options['n_intensity_bins']))
                        if 'distance' in haralick_options:
                            self.haralick_distance_le.setText(str(haralick_options['distance']))

                if 'intensity_measurement_radii' in measurement_instructions:
                    intensity_measurement_radii = measurement_instructions['intensity_measurement_radii']
                    if intensity_measurement_radii is not None:
                        if isinstance(intensity_measurement_radii, int):
                            radii = [intensity_measurement_radii]
                        elif isinstance(intensity_measurement_radii, list):
                            radii = []
                            for r in intensity_measurement_radii:
                                if isinstance(r, int) | isinstance(r, float):
                                    radii.append(str(int(r)))
                                elif isinstance(r, list):
                                    radii.append(str(int(r[0])) + '-' + str(int(r[1])))
                        self.radii_list.list_widget.clear()
                        self.radii_list.list_widget.addItems(radii)
                    else:
                        self.radii_list.list_widget.clear()

                if 'isotropic_operations' in measurement_instructions:
                    isotropic_operations = measurement_instructions['isotropic_operations']
                    if (isotropic_operations is not None) and len(isotropic_operations) > 0:
                        self.operations_list.list_widget.clear()
                        self.operations_list.list_widget.addItems(isotropic_operations)
                    else:
                        self.operations_list.list_widget.clear()

                # if 'radial_intensity' in measurement_instructions:
                #     radial_intensity = measurement_instructions['radial_intensity']
                #     if radial_intensity is not None:
                #         self.radial_intensity_btn.setChecked(True)
                #         self.step_size.setText(str(radial_intensity['radial_step']))
                #         self.step_size.setEnabled(True)
                #         self.step_lbl.setEnabled(True)
                #         for checkbox in self.channel_chechkboxes:
                #             checkbox.setEnabled(True)
                #             if checkbox.text() in radial_intensity['radial_channels']:
                #                 checkbox.setChecked(True)

                if 'clear_previous' in measurement_instructions:
                    self.clear_previous = measurement_instructions['clear_previous']
                    self.clear_previous_btn.setChecked(self.clear_previous)

    def locate_image(self):

        """
		Load the first frame of the first movie found in the experiment folder as a sample.
		"""

        movies = glob(self.parent.parent.pos + f"movie/{self.parent.parent.movie_prefix}*.tif")
        print(movies)
        if len(movies) == 0:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("No movies are detected in the experiment folder. Cannot load an image to test Haralick.")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                self.test_frame = None
                return None
        else:
            self.stack0 = movies[0]
            n_channels = len(self.channels)
            len_movie_auto = auto_load_number_of_frames(self.stack0)
            if len_movie_auto is None:
                stack = imread(self.stack0)
                len_movie_auto = len(stack)
                del stack
                gc.collect()
            self.mid_time = len_movie_auto // 2
            self.test_frame = load_frames(n_channels * self.mid_time + np.arange(n_channels), self.stack0, scale=None,
                                          normalize_input=False)

    def control_haralick_digitalization(self):

        """
		Load an image for the first experiment movie found.
		Apply the Haralick parameters and check the result of the digitization (normalization + binning of intensities).

		"""

        self.locate_image()
        self.extract_haralick_options()
        if self.test_frame is not None:
            digitized_img = compute_haralick_features(self.test_frame, np.zeros(self.test_frame.shape[:2]),
                                                      channels=self.channel_names, return_digit_image_only=True,
                                                      **self.haralick_options
                                                      )

            self.fig, self.ax = plt.subplots()
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            self.imshow_digit_window = FigureCanvas(self.fig, title="Haralick: control digitization")
            self.ax.clear()
            im = self.ax.imshow(digitized_img, cmap='gray')
            self.fig.colorbar(im, cax=cax, orientation='vertical')
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.fig.set_facecolor('none')  # or 'None'
            self.fig.canvas.setStyleSheet("background-color: transparent;")
            self.imshow_digit_window.canvas.draw()
            self.imshow_digit_window.show()

    def control_haralick_intensity_histogram(self):

        """
		Load an image for the first experiment movie found.
		Apply the Haralick normalization parameters and check the normalized intensity histogram.

		"""

        self.locate_image()
        self.extract_haralick_options()
        if self.test_frame is not None:
            norm_img = compute_haralick_features(self.test_frame, np.zeros(self.test_frame.shape[:2]),
                                                 channels=self.channel_names, return_norm_image_only=True,
                                                 **self.haralick_options
                                                 )
            self.fig, self.ax = plt.subplots(1, 1, figsize=(4, 3))
            self.hist_window = FigureCanvas(self.fig, title="Haralick: control digitized histogram")
            self.ax.clear()
            self.ax.hist(norm_img.flatten(), bins=self.haralick_options['n_intensity_bins'])
            self.ax.set_xlabel('gray level value')
            self.ax.set_ylabel('#')
            plt.tight_layout()
            self.fig.set_facecolor('none')  # or 'None'
            self.fig.canvas.setStyleSheet("background-color: transparent;")
            self.hist_window.canvas.draw()
            self.hist_window.show()

    def view_selected_contour(self):

        """
		Show the ROI for the selected contour measurement on experimental data.

		"""

        if self.parent.parent.position_list.currentText() == '*':
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("Please select a single position to visualize the border selection.")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return None
            else:
                return None

        self.locate_image()

        self.locate_mask()
        if self.test_mask is None:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("The segmentation results could not be found for this position.")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Yes:
                return None
            else:
                return None
        # plt.imshow(self.test_frame[:,:,0])
        # plt.pause(2)
        # plt.close()

        # plt.imshow(self.test_mask)
        # plt.pause(2)
        # plt.close()

        if (self.test_frame is not None) and (self.test_mask is not None):

            values = self.contours_list.list_widget.selectedItems()
            if len(values) > 0:
                distance = values[0].text()
                if '-' in distance:
                    if distance[0] != '-':
                        border_dist = distance.split('-')
                        border_dist = [float(d) for d in border_dist]
                    else:
                        border_dist = float(distance)
                elif distance.isnumeric():
                    border_dist = float(distance)

                print(border_dist)
                border_label = contour_of_instance_segmentation(self.test_mask, border_dist)

                self.fig_contour, self.ax_contour = plt.subplots(figsize=(5, 5))
                self.imshow_contour = FigureCanvas(self.fig_contour, title="Contour measurement", interactive=True)
                self.ax_contour.clear()
                self.im_contour = self.ax_contour.imshow(self.test_frame[:, :, 0], cmap='gray')
                self.im_mask = self.ax_contour.imshow(np.ma.masked_where(border_label == 0, border_label),
                                                      cmap='viridis', interpolation='none')
                self.ax_contour.set_xticks([])
                self.ax_contour.set_yticks([])
                self.ax_contour.set_title(border_dist)
                self.fig_contour.set_facecolor('none')  # or 'None'
                self.fig_contour.canvas.setStyleSheet("background-color: transparent;")
                self.imshow_contour.canvas.draw()

                self.imshow_contour.layout.setContentsMargins(30, 30, 30, 30)
                self.channel_hbox_contour = QHBoxLayout()
                self.channel_hbox_contour.addWidget(QLabel('channel: '), 10)
                self.channel_cb_contour = QComboBox()
                self.channel_cb_contour.addItems(self.channel_names)
                self.channel_cb_contour.currentIndexChanged.connect(self.switch_channel_contour)
                self.channel_hbox_contour.addWidget(self.channel_cb_contour, 90)
                self.imshow_contour.layout.addLayout(self.channel_hbox_contour)

                self.contrast_hbox_contour = QHBoxLayout()
                self.contrast_hbox_contour.addWidget(QLabel('contrast: '), 10)
                self.contrast_slider_contour = QLabeledDoubleRangeSlider()
                self.contrast_slider_contour.setSingleStep(0.00001)
                self.contrast_slider_contour.setTickInterval(0.00001)
                self.contrast_slider_contour.setOrientation(1)
                self.contrast_slider_contour.setRange(np.amin(self.test_frame[:, :, 0]),
                                                      np.amax(self.test_frame[:, :, 0]))
                self.contrast_slider_contour.setValue([np.percentile(self.test_frame[:, :, 0].flatten(), 1),
                                                       np.percentile(self.test_frame[:, :, 0].flatten(), 99.99)])
                self.im_contour.set_clim(vmin=np.percentile(self.test_frame[:, :, 0].flatten(), 1),
                                         vmax=np.percentile(self.test_frame[:, :, 0].flatten(), 99.99))
                self.contrast_slider_contour.valueChanged.connect(self.contrast_im_contour)
                self.contrast_hbox_contour.addWidget(self.contrast_slider_contour, 90)
                self.imshow_contour.layout.addLayout(self.contrast_hbox_contour)

                self.alpha_mask_hbox_contour = QHBoxLayout()
                self.alpha_mask_hbox_contour.addWidget(QLabel('mask transparency: '), 10)
                self.transparency_slider = QLabeledDoubleSlider()
                self.transparency_slider.setSingleStep(0.001)
                self.transparency_slider.setTickInterval(0.001)
                self.transparency_slider.setOrientation(1)
                self.transparency_slider.setRange(0, 1)
                self.transparency_slider.setValue(0.5)
                self.transparency_slider.valueChanged.connect(self.make_contour_transparent)
                self.alpha_mask_hbox_contour.addWidget(self.transparency_slider, 90)
                self.imshow_contour.layout.addLayout(self.alpha_mask_hbox_contour)

                self.imshow_contour.show()

        else:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("No contour was selected. Please first add a contour to the list.")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Yes:
                return None

    def locate_mask(self):

        """
		Load the first mask of the detected movie.
		"""

        labels_path = str(Path(self.stack0).parent.parent) + f'/labels_{self.mode}/'
        masks = natsorted(glob(labels_path + '*.tif'))
        if len(masks) == 0:
            print('no mask found')
            self.test_mask = None
        else:
            self.test_mask = imread(masks[self.mid_time])

    def switch_channel_contour(self, value):

        """
		Adjust intensity values when changing channels in the contour visualizer.

		"""

        self.im_contour.set_array(self.test_frame[:, :, value])
        self.im_contour.set_clim(vmin=np.percentile(self.test_frame[:, :, value].flatten(), 1),
                                 vmax=np.percentile(self.test_frame[:, :, value].flatten(), 99.99))
        self.contrast_slider_contour.setRange(np.amin(self.test_frame[:, :, value]),
                                              np.amax(self.test_frame[:, :, value]))
        self.contrast_slider_contour.setValue([np.percentile(self.test_frame[:, :, value].flatten(), 1),
                                               np.percentile(self.test_frame[:, :, value].flatten(), 99.99)])
        self.fig_contour.canvas.draw_idle()

    def contrast_im_contour(self, value):
        vmin = value[0]
        vmax = value[1]
        self.im_contour.set_clim(vmin=vmin, vmax=vmax)
        self.fig_contour.canvas.draw_idle()

    def make_contour_transparent(self, value):

        self.im_mask.set_alpha(value)
        self.fig_contour.canvas.draw_idle()

    def populate_normalisation_tabs(self):
        layout = QVBoxLayout(self.normalisation_frame)
        self.normalisation_lbl = QLabel("BACKGROUND CORRECTION")
        self.normalisation_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
        layout.addWidget(self.normalisation_lbl, alignment=Qt.AlignCenter)
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.normalisation_list = QListWidget()
        self.tabs.addTab(self.tab1, 'Local')
        self.tabs.addTab(self.tab2, 'Field')
        self.tab1.setLayout(self.populate_local_norm_tab())
        self.tab2.setLayout(self.populate_field_norm_tab())
        layout.addWidget(self.tabs)
        self.norm_list_lbl = QLabel('Background correction to perform:')
        hbox = QHBoxLayout()
        hbox.addWidget(self.norm_list_lbl)
        self.del_norm_btn = QPushButton("")
        self.del_norm_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
        self.del_norm_btn.setIcon(icon(MDI6.trash_can, color="black"))
        self.del_norm_btn.setToolTip("Remove background correction")
        self.del_norm_btn.setIconSize(QSize(20, 20))
        hbox.addWidget(self.del_norm_btn, alignment=Qt.AlignRight)
        layout.addLayout(hbox)
        self.del_norm_btn.clicked.connect(self.remove_item_from_list)
        layout.addWidget(self.normalisation_list)

    def populate_local_norm_tab(self):
        tab1_layout = QGridLayout(self.tab1)

        channel_lbl = QLabel()
        channel_lbl.setText("Channel: ")
        tab1_layout.addWidget(channel_lbl, 0, 0)

        self.tab1_channel_dropdown = QComboBox()
        self.tab1_channel_dropdown.addItems(self.channel_names)
        tab1_layout.addWidget(self.tab1_channel_dropdown, 0, 1)

        tab1_lbl = QLabel()
        tab1_lbl.setText("Outer distance (in px): ")
        tab1_layout.addWidget(tab1_lbl, 1, 0)

        self.tab1_txt_distance = QLineEdit()
        tab1_layout.addWidget(self.tab1_txt_distance, 1, 1)

        self.tab1_vis_brdr = QPushButton()
        self.tab1_vis_brdr.setStyleSheet(self.parent.parent.parent.button_select_all)
        self.tab1_vis_brdr.setIcon(icon(MDI6.eye_outline, color="black"))
        self.tab1_vis_brdr.setToolTip("View contour")
        self.tab1_vis_brdr.setIconSize(QSize(20, 20))
        self.tab1_vis_brdr.clicked.connect(self.view_normalisation_contour)
        tab1_layout.addWidget(self.tab1_vis_brdr, 1, 2)

        tab1_lbl_type = QLabel()
        tab1_lbl_type.setText("Type: ")
        tab1_layout.addWidget(tab1_lbl_type, 2, 0)

        self.tab1_dropdown = QComboBox()
        self.tab1_dropdown.addItem("Mean")
        self.tab1_dropdown.addItem("Median")
        tab1_layout.addWidget(self.tab1_dropdown, 2, 1)

        self.tab1_subtract = QRadioButton('Subtract')
        self.tab1_divide = QRadioButton('Divide')
        tab1_layout.addWidget(self.tab1_subtract, 3, 0, alignment=Qt.AlignRight)
        tab1_layout.addWidget(self.tab1_divide, 3, 1, alignment=Qt.AlignRight)

        tab1_submit = QPushButton()
        tab1_submit.setText('Add channel')
        tab1_submit.setStyleSheet(self.parent.parent.parent.button_style_sheet_2)
        tab1_layout.addWidget(tab1_submit, 4, 0, 1, 3)
        tab1_submit.clicked.connect(self.add_item_to_list)

        return tab1_layout

    def populate_field_norm_tab(self):

        tab2_layout = QGridLayout(self.tab2)

        channel_lbl = QLabel()
        channel_lbl.setText("Channel: ")
        tab2_layout.addWidget(channel_lbl, 0, 0)

        self.tab2_channel_dropdown = QComboBox()
        self.tab2_channel_dropdown.addItems(self.channel_names)
        tab2_layout.addWidget(self.tab2_channel_dropdown, 0, 1)

        tab2_lbl = QLabel()
        tab2_lbl.setText("std threshold: ")
        tab2_layout.addWidget(tab2_lbl, 1, 0)

        self.tab2_txt_threshold = QLineEdit()
        tab2_layout.addWidget(self.tab2_txt_threshold, 1, 1)

        self.norm_digit_btn = QPushButton()
        self.norm_digit_btn.setIcon(icon(MDI6.image_check, color="k"))
        self.norm_digit_btn.setStyleSheet(self.parent.parent.parent.button_select_all)

        self.norm_digit_btn.clicked.connect(self.show_threshold_visual)
        tab2_layout.addWidget(self.norm_digit_btn, 1, 2)

        tab2_lbl_type = QLabel()
        tab2_lbl_type.setText("Type: ")
        tab2_layout.addWidget(tab2_lbl_type, 2, 0)

        self.tab2_dropdown = QComboBox()
        self.tab2_dropdown.addItems(["Paraboloid", "Plane"])
        tab2_layout.addWidget(self.tab2_dropdown, 2, 1)

        self.tab2_subtract = QRadioButton('Subtract')
        self.tab2_divide = QRadioButton('Divide')
        self.tab2_sd_btn_group = QButtonGroup(self)
        self.tab2_sd_btn_group.addButton(self.tab2_subtract)
        self.tab2_sd_btn_group.addButton(self.tab2_divide)
        tab2_layout.addWidget(self.tab2_subtract, 3, 0, alignment=Qt.AlignRight)
        tab2_layout.addWidget(self.tab2_divide, 3, 1, alignment=Qt.AlignRight)

        self.tab2_clip = QRadioButton('Clip')
        self.tab2_no_clip = QRadioButton("Don't clip")
        self.tab2_clip_group = QButtonGroup(self)
        self.tab2_clip_group.addButton(self.tab2_clip)
        self.tab2_clip_group.addButton(self.tab2_no_clip)
        self.tab2_clip.setEnabled(False)
        self.tab2_no_clip.setEnabled(False)
        tab2_layout.addWidget(self.tab2_clip, 4, 0, alignment=Qt.AlignLeft)
        tab2_layout.addWidget(self.tab2_no_clip, 4, 1, alignment=Qt.AlignLeft)
        self.tab2_subtract.toggled.connect(self.show_clipping_options)
        self.tab2_divide.toggled.connect(self.show_clipping_options)

        self.view_norm_btn = QPushButton("")
        self.view_norm_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
        self.view_norm_btn.setIcon(icon(MDI6.eye_outline, color="black"))
        self.view_norm_btn.setToolTip("View flat image")
        self.view_norm_btn.setIconSize(QSize(20, 20))
        self.view_norm_btn.clicked.connect(self.preview_normalisation)
        tab2_layout.addWidget(self.view_norm_btn, 4, 2)

        tab2_submit = QPushButton()
        tab2_submit.setText('Add channel')
        tab2_submit.setStyleSheet(self.parent.parent.parent.button_style_sheet_2)
        tab2_layout.addWidget(tab2_submit, 5, 0, 1, 3)
        tab2_submit.clicked.connect(self.add_item_to_list)

        return tab2_layout

    def show_threshold_visual(self):
        min_threshold = self.tab2_txt_threshold.text()
        if min_threshold == "":
            min_threshold = 0
        current_channel = self.tab2_channel_dropdown.currentIndex()
        self.threshold_visual = ThresholdNormalisation(min_threshold=int(min_threshold),
                                                       current_channel=current_channel, parent=self)

    def show_clipping_options(self):
        if self.tab2_subtract.isChecked():
            for button in self.tab2_clip_group.buttons():
                button.setEnabled(True)
        if self.tab2_divide.isChecked():
            self.tab2_clip_group.setExclusive(False)
            for button in self.tab2_clip_group.buttons():
                button.setChecked(False)
                button.setEnabled(False)
            self.tab2_clip_group.setExclusive(True)

    def add_item_to_list(self):
        check = self.check_the_information()
        if check is True:

            if self.tabs.currentIndex() == 0:

                norm_params = {"target channel": self.tab1_channel_dropdown.currentText(), "mode": "local",
                               "distance": self.tab1_txt_distance.text(),
                               "type": self.tab1_dropdown.currentText()}
                if self.tab1_subtract.isChecked():
                    norm_params["operation"] = "Subtract"
                else:
                    norm_params["operation"] = "Divide"
            elif self.tabs.currentIndex() == 1:
                norm_params = {"target channel": self.tab2_channel_dropdown.currentText(), "mode": "field",
                               "threshold": self.tab2_txt_threshold.text(),
                               "type": self.tab2_dropdown.currentText()}
                if self.tab2_subtract.isChecked():
                    norm_params["operation"] = "Subtract"
                    if self.tab2_clip.isChecked():
                        norm_params["clip"] = True
                    else:
                        norm_params["clip"] = False
                elif self.tab2_divide.isChecked():
                    norm_params["operation"] = "Divide"

            self.background_correction.append(norm_params)
            normalisation_description = ""
            for index, (key, value) in enumerate(norm_params.items()):
                if index > 0:
                    normalisation_description += ", "
                normalisation_description += str(key) + " : " + str(value)
            self.normalisation_list.addItem(normalisation_description)

    def remove_item_from_list(self):
        current_item = self.normalisation_list.currentRow()
        if current_item > -1:
            del self.background_correction[current_item]
            self.normalisation_list.takeItem(current_item)

    def check_the_information(self):
        if self.tabs.currentIndex() == 0:
            if self.background_correction  is None:
                self.background_correction  = []
            for index, normalisation_opt in enumerate(self.background_correction ):
                if self.tab1_channel_dropdown.currentText() in normalisation_opt['target channel']:
                    result = self.channel_already_in_list()
                    if result != QMessageBox.Yes:
                        return False
                    else:
                        self.background_correction .remove(normalisation_opt)
                        self.normalisation_list.takeItem(index)
                        return True
            if self.tab1_txt_distance.text() == "":
                self.display_message_box('provide the outer distance')
                return False
            if not self.tab1_subtract.isChecked() and not self.tab1_divide.isChecked():
                self.display_message_box('choose the operation')
                return False
            else:
                return True

        elif self.tabs.currentIndex() == 1:
            if self.background_correction is None:
                self.background_correction = []
            for index, normalisation_opt in enumerate(self.background_correction ):
                if self.tab2_channel_dropdown.currentText() in normalisation_opt['target channel']:
                    result = self.channel_already_in_list()
                    if result != QMessageBox.Yes:
                        return False
                    else:
                        self.background_correction .remove(normalisation_opt)
                        self.normalisation_list.takeItem(index)
                        return True
            if self.tab2_txt_threshold.text() == "":
                self.display_message_box('provide the threshold')
                return False
            elif not self.tab2_divide.isChecked() and not self.tab2_subtract.isChecked():
                self.display_message_box('choose subtraction or division')
                return False
            elif self.tab2_subtract.isChecked():
                if not self.tab2_clip.isChecked() and not self.tab2_no_clip.isChecked():
                    self.display_message_box('provide the clipping mode')
                    return False
            return True

    def display_message_box(self, missing_info):
        QMessageBox.about(self, "Message Box Title", "Please " + missing_info + " for background correction")

    def channel_already_in_list(self):
        response = QMessageBox.question(self, "Message Box Title",
                                        "The background correction parameters for this channel already exist, "
                                        "continuing will erase the previous configurations. Are you sure you want to "
                                        "proceed?",
                                        QMessageBox.No | QMessageBox.Yes, QMessageBox.No)
        return response

    def fun(self, x, y):
        return x ** 2 + y

    def preview_normalisation(self):
        self.locate_image()
        normalised, bg_fit = field_normalisation(self.test_frame[:, :, self.tab2_channel_dropdown.currentIndex()],
                                                 threshold=self.tab2_txt_threshold.text(),
                                                 normalisation_operation=self.tab2_subtract.isChecked(),
                                                 clip=self.tab2_clip.isChecked(),
                                                 mode=self.tab2_dropdown.currentText())
        self.fig, self.ax = plt.subplots()
        self.normalised_img = FigureCanvas(self.fig, "Corrected background image preview")
        self.ax.clear()
        self.ax.imshow(normalised, cmap='gray')
        self.normalised_img.canvas.draw()
        self.normalised_img.show()

    def view_normalisation_contour(self):

        """
		Show the ROI for the selected contour measurement on experimental data.

		"""

        if self.parent.parent.position_list.currentText() == '*':
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("Please select a single position to visualize the border selection.")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return None
            else:
                return None

        self.locate_image()

        self.locate_mask()
        if self.test_mask is None:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("The segmentation results could not be found for this position.")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Yes:
                return None
            else:
                return None

        if (self.test_frame is not None) and (self.test_mask is not None):
            contour = float(self.tab1_txt_distance.text())
            contour = contour * (-1)
            border_label = contour_of_instance_segmentation(self.test_mask, contour)

            self.fig_contour, self.ax_contour = plt.subplots(figsize=(5, 5))
            self.imshow_contour = FigureCanvas(self.fig_contour, title="Contour measurement", interactive=True)
            self.ax_contour.clear()
            self.im_contour = self.ax_contour.imshow(self.test_frame[:, :, self.tab1_channel_dropdown.currentIndex()],
                                                     cmap='gray')
            self.im_mask = self.ax_contour.imshow(np.ma.masked_where(border_label == 0, border_label),
                                                  cmap='viridis', interpolation='none')
            self.ax_contour.set_xticks([])
            self.ax_contour.set_yticks([])
            self.ax_contour.set_title(contour * (-1))
            self.fig_contour.set_facecolor('none')  # or 'None'
            self.fig_contour.canvas.setStyleSheet("background-color: transparent;")
            self.imshow_contour.canvas.draw()

            self.imshow_contour.show()

    # def enable_step_size(self):
    #     if self.radial_intensity_btn.isChecked():
    #         self.step_lbl.setEnabled(True)
    #         self.step_size.setEnabled(True)
    #         for checkbox in self.channel_chechkboxes:
    #             checkbox.setEnabled(True)
    #     else:
    #         self.step_lbl.setEnabled(False)
    #         self.step_size.setEnabled(False)
    #         for checkbox in self.channel_chechkboxes:
    #             checkbox.setEnabled(False)

    def populate_spot_detection(self):
        layout = QGridLayout(self.spot_detection_frame)
        self.spot_detection_lbl = QLabel("SPOT DETECTION")
        self.spot_detection_lbl.setStyleSheet("""font-weight: bold;padding: 0px;""")
        layout.addWidget(self.spot_detection_lbl, 0, 0, 1, 2, alignment=Qt.AlignCenter)
        self.spot_check= QCheckBox('Perform spot detection')
        self.spot_check.toggled.connect(self.enable_spot_detection)
        layout.addWidget(self.spot_check, 1, 0)
        self.spot_channel_lbl = QLabel("Choose channel for spot detection: ")
        self.spot_channel = QComboBox()
        self.spot_channel.addItems(self.channel_names)
        layout.addWidget(self.spot_channel_lbl, 2, 0)
        layout.addWidget(self.spot_channel, 2, 1)
        self.diameter_lbl = QLabel('Spot diameter: ')
        self.diameter_value = QLineEdit()
        layout.addWidget(self.diameter_lbl, 3, 0)
        layout.addWidget(self.diameter_value, 3, 1)
        self.threshold_lbl = QLabel('Spot threshold: ')
        self.threshold_value = QLineEdit()
        layout.addWidget(self.threshold_lbl, 4, 0)
        layout.addWidget(self.threshold_value, 4, 1)
        self.preview_spot = QPushButton('Preview')
        self.preview_spot.clicked.connect(self.spot_preview)
        self.preview_spot.setStyleSheet(self.parent.parent.parent.button_style_sheet_2)
        layout.addWidget(self.preview_spot, 5, 0, 1, 2)
        self.spot_channel.setEnabled(False)
        self.spot_channel_lbl.setEnabled(False)
        self.diameter_value.setEnabled(False)
        self.diameter_lbl.setEnabled(False)
        self.threshold_value.setEnabled(False)
        self.threshold_lbl.setEnabled(False)
        self.preview_spot.setEnabled(False)


    def spot_preview(self):
        self.locate_image()
        self.locate_mask()
        for dictionary in self.background_correction:
            if self.spot_channel.currentText() in dictionary['target channel']:
                if dictionary['mode'] == 'field':
                    if dictionary['operation'] == 'Divide':
                        normalised, bg_fit = field_normalisation(
                            self.test_frame[:, :, self.spot_channel.currentIndex()],
                            threshold=dictionary['threshold'],
                            normalisation_operation=dictionary['operation'],
                            clip=False,
                            mode=dictionary['type'])
                    else:
                        normalised, bg_fit = field_normalisation(
                            self.test_frame[:, :, self.spot_channel.currentIndex()],
                            threshold=dictionary['threshold'],
                            normalisation_operation=dictionary['operation'],
                            clip=dictionary['clip'],
                            mode=dictionary['type'])
                    self.test_frame[:, :, self.spot_channel.currentIndex()] = normalised
                if dictionary['mode'] == 'local':
                    normalised_image = normalise_by_cell(self.test_frame[:, :, self.spot_channel.currentIndex()].copy(), self.test_mask,
                                                         distance=int(dictionary['distance']), mode=dictionary['type'],
                                                         operation=dictionary['operation'])
                    self.test_frame[:, :, self.spot_channel.currentIndex()] = normalised_image

        self.spot_visual = ThresholdSpot(current_channel=self.spot_channel.currentIndex(), img=self.test_frame,
                                         mask=self.test_mask, parent=self)

    def enable_spot_detection(self):
        if self.spot_check.isChecked():
            self.spot_channel.setEnabled(True)
            self.spot_channel_lbl.setEnabled(True)
            self.diameter_value.setEnabled(True)
            self.diameter_lbl.setEnabled(True)
            self.threshold_value.setEnabled(True)
            self.threshold_lbl.setEnabled(True)
            self.preview_spot.setEnabled(True)

        else:
            self.spot_channel.setEnabled(False)
            self.spot_channel_lbl.setEnabled(False)
            self.diameter_value.setEnabled(False)
            self.diameter_lbl.setEnabled(False)
            self.threshold_value.setEnabled(False)
            self.threshold_lbl.setEnabled(False)
            self.preview_spot.setEnabled(False)
