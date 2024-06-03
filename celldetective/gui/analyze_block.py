from PyQt5.QtWidgets import QFrame, QGridLayout, QComboBox, QSizePolicy, QSpacerItem, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QCheckBox, QMessageBox
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon

from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import gc
from celldetective.io import get_segmentation_models_list, control_segmentation_napari, get_signal_models_list, control_tracking_btrack
from celldetective.gui import ConfigSurvival, ConfigSignalPlot
from celldetective.gui.gui_utils import QHSeperationLine
from celldetective.segmentation import segment_at_position, segment_from_threshold_at_position
from celldetective.tracking import track_at_position
from celldetective.measure import measure_at_position
from celldetective.signals import analyze_signals_at_position
import numpy as np
from glob import glob
from natsort import natsorted
import os
import pandas as pd
from celldetective.gui import Styles

class AnalysisPanel(QFrame, Styles):
	def __init__(self, parent_window, title=None):

		super().__init__()		
		self.parent_window = parent_window
		self.title = title
		if self.title is None:
			self.title=''
		self.exp_channels = self.parent_window.exp_channels
		self.exp_dir = self.parent_window.exp_dir
		self.soft_path = self.parent_window.parent_window.soft_path

		self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.grid = QVBoxLayout(self)
		self.grid.setSpacing(20)
		self.generate_header()
	
	def generate_header(self):

		"""
		Read the mode and prepare a collapsable block to process a specific cell population.

		"""

		panel_title = QLabel("Survival")
		panel_title.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")

		self.grid.addWidget(panel_title, alignment=Qt.AlignCenter)

		self.survival_btn = QPushButton("plot survival")
		self.survival_btn.setIcon(QIcon(QIcon(os.sep.join([self.soft_path,'celldetective','icons','survival2.png']))))
		self.survival_btn.setStyleSheet(self.button_style_sheet_2)
		self.survival_btn.setIconSize(QSize(35, 35))
		self.survival_btn.clicked.connect(self.configure_survival)
		self.grid.addWidget(self.survival_btn)

		signal_lbl = QLabel("Single-cell signals")
		signal_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")

		self.grid.addWidget(signal_lbl, alignment=Qt.AlignCenter)

		self.plot_signal_btn = QPushButton("plot signals")
		self.plot_signal_btn.setIcon(QIcon(QIcon(os.sep.join([self.soft_path,'celldetective','icons','signals_icon.png']))))
		self.plot_signal_btn.setStyleSheet(self.button_style_sheet_2)
		self.plot_signal_btn.setIconSize(QSize(35, 35))
		self.plot_signal_btn.clicked.connect(self.configure_plot_signals)
		self.grid.addWidget(self.plot_signal_btn)
		
		verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
		self.grid.addItem(verticalSpacer)

	def configure_survival(self):
		print('survival analysis starting!!!')
		self.configSurvival = ConfigSurvival(self)
		self.configSurvival.show()

	def configure_plot_signals(self):
		print('plot signal analysis starting!!!')
		self.ConfigSignalPlot = ConfigSignalPlot(self)
		self.ConfigSignalPlot.show()