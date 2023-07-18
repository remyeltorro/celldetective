from PyQt5.QtWidgets import QFrame, QGridLayout, QComboBox, QLabel, QPushButton, QHBoxLayout, QCheckBox
from PyQt5.QtCore import Qt, QSize
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import gc
from celldetective.io import get_segmentation_models_list, control_segmentation_napari
from celldetective.gui import SegmentationModelLoader

class ProcessPanel(QFrame):
	def __init__(self, parent, mode):

		super().__init__()		
		self.parent = parent
		self.mode = mode
		self.exp_channels = self.parent.exp_channels

		self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.grid = QGridLayout(self)
		self.generate_header()
	
	def generate_header(self):

		"""
		Read the mode and prepare a collapsable block to process a specific cell population.

		"""

		panel_title = QLabel(f"PROCESS {self.mode.upper()}")
		panel_title.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")

		self.grid.addWidget(panel_title, 0, 0, 1, 4, alignment=Qt.AlignCenter)

		self.select_all_btn = QPushButton()
		self.select_all_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
		self.select_all_btn.setIconSize(QSize(25, 25))
		self.all_ticked = False
		self.select_all_btn.clicked.connect(self.tick_all_actions)
		self.select_all_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.grid.addWidget(self.select_all_btn, 0, 0, 1, 4, alignment=Qt.AlignLeft)
		#self.to_disable.append(self.all_tc_actions)
		
		self.collapse_btn = QPushButton()
		self.collapse_btn.setIcon(icon(MDI6.chevron_down,color="black"))
		self.collapse_btn.setIconSize(QSize(25, 25))
		self.collapse_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.grid.addWidget(self.collapse_btn, 0, 0, 1, 4, alignment=Qt.AlignRight)

		self.populate_contents()
		self.grid.addWidget(self.ContentsFrame, 1, 0, 1, 4, alignment=Qt.AlignTop)
		self.collapse_btn.clicked.connect(lambda: self.ContentsFrame.setHidden(not self.ContentsFrame.isHidden()))
		self.collapse_btn.clicked.connect(self.collapse_advanced)
		self.ContentsFrame.hide()

	def collapse_advanced(self):
		if self.ContentsFrame.isHidden():
			self.collapse_btn.setIcon(icon(MDI6.chevron_down,color="black"))
			self.collapse_btn.setIconSize(QSize(25, 25))
		else:
			self.collapse_btn.setIcon(icon(MDI6.chevron_up,color="black"))
			self.collapse_btn.setIconSize(QSize(25, 25))

	def populate_contents(self):

		self.ContentsFrame = QFrame()
		self.grid_contents = QGridLayout(self.ContentsFrame)
		self.grid_contents.setContentsMargins(0,0,0,0)
		self.generate_segmentation_options()
		

	def generate_segmentation_options(self):

		grid_segment = QHBoxLayout()
		grid_segment.setContentsMargins(0,0,0,0)
		grid_segment.setSpacing(0)

		self.segment_action = QCheckBox("SEGMENT")
		self.segment_action.setStyleSheet("""
			font-size: 14px;
			padding-left: 10px;
			""")
		self.segment_action.setIcon(icon(MDI6.bacteria, color='black'))
		self.segment_action.setToolTip("Segment the cells in the movie\nusing the selected pre-trained model and save\nthe labeled output in a sub-directory.")
		self.segment_action.toggled.connect(self.enable_segmentation_model_list)
		#self.to_disable.append(self.segment_action)
		grid_segment.addWidget(self.segment_action, 90)
		
		self.check_seg_btn = QPushButton()
		self.check_seg_btn.setIcon(icon(MDI6.eye_check_outline,color="black"))
		self.check_seg_btn.setIconSize(QSize(25, 25))
		self.check_seg_btn.clicked.connect(self.check_segmentation)
		self.check_seg_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.check_seg_btn.setEnabled(False)
		#self.to_disable.append(self.control_target_seg)
		grid_segment.addWidget(self.check_seg_btn, 10)

		self.grid_contents.addLayout(grid_segment, 0,0,1,4)
		self.grid_contents.addWidget(QLabel("Model zoo:"),2,0,1,1)
		self.seg_model_list = QComboBox()
		#self.to_disable.append(self.tc_seg_model_list)
		self.seg_model_list.setGeometry(50, 50, 200, 30)
		self.init_seg_model_list()

		self.train_btn = QPushButton("TRAIN")
		self.train_btn.setToolTip("Open a dialog box to create a new target segmentation model.")
		self.train_btn.setIcon(icon(MDI6.redo_variant,color='black'))
		#self.train_btn.setIconSize(QSize(20, 20)) 
		self.train_btn.setStyleSheet(self.parent.parent.button_style_sheet_3)
		self.grid_contents.addWidget(self.train_btn, 2, 3, 1, 1)
		# self.train_button_tc.clicked.connect(self.train_stardist_model_tc)
		# self.to_disable.append(self.train_button_tc)

		self.upload_model_btn = QPushButton("UPLOAD")
		self.upload_model_btn.setIcon(icon(MDI6.upload,color="black"))
		self.upload_model_btn.setIconSize(QSize(20, 20))
		self.upload_model_btn.setStyleSheet(self.parent.parent.button_style_sheet_3)
		self.upload_model_btn.setToolTip("Upload...")
		self.grid_contents.addWidget(self.upload_model_btn, 2, 2, 1, 1)
		self.upload_model_btn.clicked.connect(self.upload_segmentation_model)
		# self.to_disable.append(self.upload_tc_model)

		self.seg_model_list.setEnabled(False)
		self.grid_contents.addWidget(self.seg_model_list, 3, 0, 1, 4)

	def check_segmentation(self):

		#self.freeze()
		#QApplication.setOverrideCursor(Qt.WaitCursor)
		self.parent.locate_selected_position()
		control_segmentation_napari(self.parent.pos, prefix=self.parent.movie_prefix, population=self.mode[:-1])
		gc.collect()

	def enable_segmentation_model_list(self):
		if self.segment_action.isChecked():
			self.seg_model_list.setEnabled(True)
		else:
			self.seg_model_list.setEnabled(False)
	
	def init_seg_model_list(self):

		self.seg_model_list.clear()
		seg_models = get_segmentation_models_list(mode=self.mode, return_path=False)
		self.seg_model_list.addItems(["Threshold"])
		self.seg_model_list.addItems(seg_models)
		
		#if ("live_nuclei_channel" in self.exp_channels)*("dead_nuclei_channel" in self.exp_channels):
		# 	print("both channels found")
		# 	index = self.tc_seg_model_list.findText("MCF7_Hoescht_PI_w_primary_NK", Qt.MatchFixedString)
		# 	if index >= 0:
		# 		self.tc_seg_model_list.setCurrentIndex(index)
		# elif ("live_nuclei_channel" in self.exp_channels)*("dead_nuclei_channel" not in self.exp_channels):
		# 	index = self.tc_seg_model_list.findText("MCF7_Hoescht_w_primary_NK", Qt.MatchFixedString)
		# 	if index >= 0:
		# 		self.tc_seg_model_list.setCurrentIndex(index)
		# elif ("live_nuclei_channel" not in self.exp_channels)*("dead_nuclei_channel" in self.exp_channels):
		# 	index = self.tc_seg_model_list.findText("MCF7_PI_w_primary_NK", Qt.MatchFixedString)
		# 	if index >= 0:
		# 		self.tc_seg_model_list.setCurrentIndex(index)
		# elif ("live_nuclei_channel" not in self.exp_channels)*("dead_nuclei_channel" not in self.exp_channels)*("adhesion_channel" in self.exp_channels):
		# 	index = self.tc_seg_model_list.findText("RICM", Qt.MatchFixedString)
		# 	if index >= 0:
		# 		self.tc_seg_model_list.setCurrentIndex(index)

	def tick_all_actions(self):
		self.switch_all_ticks_option()
		if self.all_ticked:
			self.select_all_btn.setIcon(icon(MDI6.checkbox_outline,color="black"))
			self.select_all_btn.setIconSize(QSize(25, 25))
			self.segment_action.setChecked(True)
		else:
			self.select_all_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
			self.select_all_btn.setIconSize(QSize(25, 25))
			self.segment_action.setChecked(False)

	def switch_all_ticks_option(self):
		if self.all_ticked == True:
			self.all_ticked = False
		else:
			self.all_ticked = True

	def upload_segmentation_model(self):

		self.SegModelLoader = SegmentationModelLoader(self)
		self.SegModelLoader.show()

	def closeEvent(self, event):

		"""
		Close child windows if closed.
		"""
		
		try:
			if self.SegModelLoader:
				self.SegModelLoader.close()
		except:
			pass

		gc.collect()