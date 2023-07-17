from PyQt5.QtWidgets import QFrame, QGridLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QSize
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import gc


class ProcessPanel(QFrame):
	def __init__(self, parent, mode):

		super().__init__()		
		self.parent = parent
		self.mode = mode

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
		#self.all_tc_actions.clicked.connect(self.tick_all_actions)
		self.select_all_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.grid.addWidget(self.select_all_btn, 0, 0, 1, 4, alignment=Qt.AlignLeft)
		#self.to_disable.append(self.all_tc_actions)
		
		self.collapse_btn = QPushButton()
		self.collapse_btn.setIcon(icon(MDI6.chevron_down,color="black"))
		self.collapse_btn.setIconSize(QSize(25, 25))
		self.collapse_btn.setStyleSheet(self.parent.parent.button_select_all)
		self.grid.addWidget(self.collapse_btn, 0, 0, 1, 4, alignment=Qt.AlignRight)

		ContentsFrame = self.populate_contents()
		# self.frame_tc_options = self.FrameTargetOptions()
		# self.grid_tc.addWidget(self.frame_tc_options, 1, 0, 1, 4, alignment=Qt.AlignTop)
		# self.expand_tc_btn.clicked.connect(lambda: self.frame_tc_options.setHidden(not self.frame_tc_options.isHidden()))
		# self.expand_tc_btn.clicked.connect(self.collapse_advanced)
		# self.frame_tc_options.hide()

	def populate_contents(self):

		ContentsFrame = QFrame()
		return ContentsFrame