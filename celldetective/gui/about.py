from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from celldetective.utils import get_software_location
import celldetective
import os
from celldetective.gui.gui_utils import center_window

class AboutWidget(QWidget):

	def __init__(self):

		super().__init__()
		self.setWindowTitle("About celldetective")
		self.setMinimumWidth(300)
		center_window(self)
		logo = QPixmap(os.sep.join([get_software_location(),'celldetective','icons','logo.png']))

		# Create the layout
		layout = QVBoxLayout(self)
		img_label = QLabel('')
		img_label.setPixmap(logo)
		layout.addWidget(img_label, alignment=Qt.AlignCenter)

		self.soft_name = QLabel('celldetective')
		self.soft_name.setStyleSheet("""font-weight: bold;
										font-size: 18px;
									""")
		layout.addWidget(self.soft_name, alignment=Qt.AlignCenter)

		self.version_lbl = QLabel(f"Version X.X.X <a href=\"https://github.com/remyeltorro/celldetective/releases\">(release notes)</a>")
		self.version_lbl.setOpenExternalLinks(True)
		layout.addWidget(self.version_lbl, alignment=Qt.AlignCenter)

		self.lab_lbl = QLabel('Developped at Laboratoire Adhésion et Inflammation')
		layout.addWidget(self.lab_lbl, alignment=Qt.AlignCenter)

