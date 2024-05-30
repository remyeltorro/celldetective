from PyQt5.QtWidgets import QHBoxLayout,QMessageBox, QPushButton, QVBoxLayout, QRadioButton, QLabel, QButtonGroup, QSizePolicy, QComboBox,QSpacerItem, QGridLayout
from celldetective.gui.gui_utils import ThresholdLineEdit
from PyQt5.QtCore import Qt, QSize
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from celldetective.utils import _extract_channel_indices_from_config
from celldetective.gui.viewers import ThresholdedStackVisualizer, StackVisualizer
from celldetective.gui import Styles
from celldetective.preprocessing import correct_background_model

class BackgroundFitCorrectionLayout(QGridLayout):
	
	"""docstring for ClassName"""
	
	def __init__(self, parent=None, *args):
		super().__init__(*args)

		self.parent = parent
		self.channel_names = self.parent.channel_names # check this

		self.setContentsMargins(15,15,15,15)
		self.init_styles()
		self.generate_widgets()
		self.add_to_layout()

	def init_styles(self):

		"""
		Initialize styles.
		"""
		
		self.Styles = Styles()
		self.qtab_style = self.Styles.qtab_style
		self.button_style_sheet = self.Styles.button_style_sheet
		self.button_style_sheet_2 = self.Styles.button_style_sheet_2
		self.button_style_sheet_2_not_done = self.Styles.button_style_sheet_2_not_done
		self.button_style_sheet_3 = self.Styles.button_style_sheet_3
		self.button_select_all = self.Styles.button_select_all

	def generate_widgets(self):

		self.channel_lbl = QLabel('Channel: ')
		self.channels_cb = QComboBox()
		self.channels_cb.addItems(self.channel_names)
		
		self.thresh_lbl = QLabel('Threshold: ')
		self.thresh_lbl.setToolTip('Threshold on the STD-filtered image.\nPixel values above the threshold are\nconsidered as non-background and are\nmasked prior to background estimation.')
		self.threshold_viewer_btn = QPushButton()
		self.threshold_viewer_btn.setIcon(icon(MDI6.image_check, color="k"))
		self.threshold_viewer_btn.setStyleSheet(self.button_select_all)
		self.threshold_viewer_btn.clicked.connect(self.set_threshold_graphically)

		self.model_lbl = QLabel('Model: ')
		self.models_cb = QComboBox()
		self.models_cb.addItems(['paraboloid', 'plane'])
		
		self.corrected_stack_viewer = QPushButton("")
		self.corrected_stack_viewer.setStyleSheet(self.button_select_all)
		self.corrected_stack_viewer.setIcon(icon(MDI6.eye_outline, color="black"))
		self.corrected_stack_viewer.setToolTip("View corrected image")
		self.corrected_stack_viewer.clicked.connect(self.preview_correction)
		self.corrected_stack_viewer.setIconSize(QSize(20, 20))

		self.add_correction_btn = QPushButton('Add correction')
		self.add_correction_btn.setStyleSheet(self.button_style_sheet_2)
		self.add_correction_btn.setIcon(icon(MDI6.plus, color="#1565c0"))
		self.add_correction_btn.setToolTip('Add correction.')
		self.add_correction_btn.setIconSize(QSize(25, 25))
		self.add_correction_btn.clicked.connect(self.add_instructions_to_parent_list)

		self.threshold_le = ThresholdLineEdit(init_value=2, connected_buttons=[self.threshold_viewer_btn,
																			   self.corrected_stack_viewer,
																			   self.add_correction_btn
																			   ])
	def add_to_layout(self):
		
		channel_layout = QHBoxLayout()
		channel_layout.addWidget(self.channel_lbl, 25)
		channel_layout.addWidget(self.channels_cb, 75)
		self.addLayout(channel_layout, 0, 0, 1, 3)

		threshold_layout = QHBoxLayout()
		threshold_layout.addWidget(self.thresh_lbl, 25)
		threshold_layout.addWidget(self.threshold_le, 70)
		threshold_layout.addWidget(self.threshold_viewer_btn, 5)
		self.addLayout(threshold_layout, 1, 0, 1, 3)

		model_layout = QHBoxLayout()
		model_layout.addWidget(self.model_lbl, 25)
		model_layout.addWidget(self.models_cb, 75)
		self.addLayout(model_layout, 2, 0, 1, 3)

		self.operation_layout = OperationLayout()
		self.addLayout(self.operation_layout, 3, 0, 1, 3)

		correction_layout = QHBoxLayout()
		correction_layout.addWidget(self.add_correction_btn, 95)
		correction_layout.addWidget(self.corrected_stack_viewer, 5)
		self.addLayout(correction_layout, 4, 0, 1, 3)

		verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
		self.addItem(verticalSpacer, 5, 0, 1, 3)

	def add_instructions_to_parent_list(self):

		self.generate_instructions()
		self.parent.background_correction.append(self.instructions)
		correction_description = ""
		for index, (key, value) in enumerate(self.instructions.items()):
			if index > 0:
				correction_description += ", "
			correction_description += str(key) + " : " + str(value)
		self.parent.normalisation_list.addItem(correction_description)

	def generate_instructions(self):

		if self.operation_layout.subtract_btn.isChecked():
			operation = "subtract"
		else:
			operation = "divide"
			clip = None

		if self.operation_layout.clip_btn.isChecked() and self.operation_layout.subtract_btn.isChecked():
			clip = True
		else:
			clip = False

		self.instructions = {
					  "target_channel": self.channels_cb.currentText(),
					  "correction_type": "fit",
					  "model": self.models_cb.currentText(),
					  "threshold_on_std": self.threshold_le.get_threshold(),
					  "operation": operation,
					  "clip": clip
					 }

	def set_target_channel(self):

		channel_indices = _extract_channel_indices_from_config(self.parent.parent.exp_config, [self.channels_cb.currentText()])
		self.target_channel = channel_indices[0]

	def set_threshold_graphically(self):

		self.parent.locate_image()
		self.set_target_channel()
		thresh = self.threshold_le.get_threshold()

		if self.parent.current_stack is not None and thresh is not None:
			self.viewer = ThresholdedStackVisualizer(initial_threshold=thresh,
													 parent_le = self.threshold_le,
													 preprocessing=[['gauss',2],["std",4]],
													 stack_path=self.parent.current_stack,
													 n_channels=len(self.channel_names),
													 target_channel=self.target_channel,
													 window_title='Set the exclusion threshold',
													 )
			self.viewer.show()

	def preview_correction(self):

		if self.parent.parent.well_list.currentText()=="*" or self.parent.parent.position_list.currentText()=="*":
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Critical)
			msgBox.setText("Please select a single position...")
			msgBox.setWindowTitle("Critical")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None

		if self.operation_layout.subtract_btn.isChecked():
			operation = "subtract"
		else:
			operation = "divide"
			clip = None

		if self.operation_layout.clip_btn.isChecked() and self.operation_layout.subtract_btn.isChecked():
			clip = True
		else:
			clip = False

		corrected_stack = correct_background_model(self.parent.exp_dir, 
						   well_option=self.parent.parent.well_list.currentIndex(), #+1 ??
						   position_option=self.parent.parent.position_list.currentIndex()-1, #+1??
						   target_channel=self.channels_cb.currentText(),
						   model = self.models_cb.currentText(),
						   threshold_on_std = self.threshold_le.get_threshold(),
						   operation = operation,
						   clip = clip,
						   export= False,
						   return_stacks=True,
						   show_progress_per_well = True,
						   show_progress_per_pos = False,
							)

		self.viewer = StackVisualizer(
									  stack=corrected_stack[0],
									  window_title='Corrected channel',
									  target_channel=self.channels_cb.currentIndex(),
									  frame_slider = True,
									  contrast_slider = True
									 )
		self.viewer.show()

class OperationLayout(QVBoxLayout):
	
	"""docstring for ClassName"""
	
	def __init__(self, ratio=(0.25,0.75), *args):
		
		super().__init__(*args)

		self.ratio = ratio
		self.generate_widgets()
		self.generate_layout()

	def generate_widgets(self):
		
		self.operation_lbl = QLabel('Operation: ')
		self.operation_group = QButtonGroup()
		self.subtract_btn = QRadioButton('Subtract')
		self.divide_btn = QRadioButton('Divide')
		self.subtract_btn.toggled.connect(self.activate_clipping_options)
		self.divide_btn.toggled.connect(self.activate_clipping_options)

		self.operation_group.addButton(self.subtract_btn)
		self.operation_group.addButton(self.divide_btn)

		self.clip_group = QButtonGroup()
		self.clip_btn = QRadioButton('Clip')
		self.clip_not_btn = QRadioButton('Do not clip')

		self.clip_group.addButton(self.clip_btn)
		self.clip_group.addButton(self.clip_not_btn)

	def generate_layout(self):
		
		operation_layout = QHBoxLayout()
		operation_layout.addWidget(self.operation_lbl, 100*int(self.ratio[0]))
		operation_layout.addWidget(self.subtract_btn, 100*int(self.ratio[1])//2, alignment=Qt.AlignCenter)
		operation_layout.addWidget(self.divide_btn, 100*int(self.ratio[1])//2, alignment=Qt.AlignCenter)
		self.addLayout(operation_layout)

		clip_layout = QHBoxLayout()
		clip_layout.addWidget(QLabel(''), 100*int(self.ratio[0]))
		clip_layout.addWidget(self.clip_btn, 100*int(self.ratio[1])//4, alignment=Qt.AlignCenter)
		clip_layout.addWidget(self.clip_not_btn, 100*int(self.ratio[1])//4, alignment=Qt.AlignCenter)
		clip_layout.addWidget(QLabel(''), 100*int(self.ratio[1])//2)
		self.addLayout(clip_layout)

		self.subtract_btn.click()
		self.clip_not_btn.click()

	def activate_clipping_options(self):
		
		if self.subtract_btn.isChecked():
			self.clip_btn.setEnabled(True)
			self.clip_not_btn.setEnabled(True)
		else:
			self.clip_btn.setEnabled(False)
			self.clip_not_btn.setEnabled(False)		


