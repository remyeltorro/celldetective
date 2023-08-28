#!/usr/bin/env python3

import sys
import os
from PyQt5.QtWidgets import QMainWindow, QApplication,QFileDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from glob import glob
from celldetective.gui import Styles, ControlPanel, ConfigNewExperiment
from celldetective.gui.gui_utils import center_window
from celldetective.utils import get_software_location
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import gc

class AppInitWindow(QMainWindow):

	"""
	Initial window to set the experiment folder or create a new one.
	"""

	def __init__(self):
		super().__init__()

		self.Styles = Styles()
		self.init_styles()
		self.setWindowTitle("celldetective")
		print(os.getcwd())
		self.soft_path = get_software_location()
		print(self.soft_path)
		self.setWindowIcon(QIcon(os.sep.join([self.soft_path,'celldetective','icons','mexican-hat.png'])))
		print(os.sep.join([self.soft_path,'celldetective','icons','mexican-hat.png']))
		center_window(self)

		app = QApplication.instance()
		self.screen = app.primaryScreen()
		self.geometry = self.screen.availableGeometry()
		self.screen_width, self.screen_height = self.geometry.getRect()[-2:]

		central_widget = QWidget()
		self.vertical_layout = QVBoxLayout(central_widget)
		self.vertical_layout.setContentsMargins(15,15,15,15)
		self.vertical_layout.addWidget(QLabel("Experiment folder:"))
		self.create_locate_exp_hbox()
		self.create_buttons_hbox()
		self.setCentralWidget(central_widget)
		self.show()

	def create_locate_exp_hbox(self):

		self.locate_exp_layout = QHBoxLayout()
		self.locate_exp_layout.setContentsMargins(0,5,0,0)
		self.experiment_path_selection = QLineEdit()
		self.experiment_path_selection.setAlignment(Qt.AlignLeft)	
		self.experiment_path_selection.setEnabled(True)
		self.experiment_path_selection.setDragEnabled(True)
		self.experiment_path_selection.setFixedWidth(400)
		self.experiment_path_selection.textChanged[str].connect(self.check_path_and_enable_opening)
		self.foldername = os.getcwd()
		self.experiment_path_selection.setPlaceholderText('/path/to/experiment/folder/')
		self.locate_exp_layout.addWidget(self.experiment_path_selection, 90)

		self.browse_button = QPushButton("Browse...")
		self.browse_button.clicked.connect(self.browse_experiment_folder)
		self.browse_button.setStyleSheet(self.button_style_sheet)
		self.browse_button.setIcon(icon(MDI6.folder, color="white"))
		self.locate_exp_layout.addWidget(self.browse_button, 10)
		self.vertical_layout.addLayout(self.locate_exp_layout)

	def create_buttons_hbox(self):

		self.buttons_layout = QHBoxLayout()
		self.buttons_layout.setContentsMargins(30,15,30,5)
		self.new_exp_button = QPushButton("New")
		self.new_exp_button.clicked.connect(self.create_new_experiment)
		self.new_exp_button.setShortcut("Ctrl+N")
		self.new_exp_button.setStyleSheet(self.button_style_sheet_2)
		self.buttons_layout.addWidget(self.new_exp_button, 50)

		self.validate_button = QPushButton("Open")
		self.validate_button.clicked.connect(self.open_directory)
		self.validate_button.setStyleSheet(self.button_style_sheet)
		self.validate_button.setEnabled(False)
		self.validate_button.setShortcut("Return")
		self.buttons_layout.addWidget(self.validate_button, 50)
		self.vertical_layout.addLayout(self.buttons_layout)

	def check_path_and_enable_opening(self):
		
		"""
		Enable 'Open' button if the text is a valid path.
		"""

		text = self.experiment_path_selection.text()
		if (os.path.exists(text)) and os.path.exists(os.sep.join([text,"config.ini"])):
			self.validate_button.setEnabled(True)
		else:
			self.validate_button.setEnabled(False)

	def init_styles(self):

		"""
		Initialize styles.
		"""
		
		self.qtab_style = self.Styles.qtab_style
		self.button_style_sheet = self.Styles.button_style_sheet
		self.button_style_sheet_2 = self.Styles.button_style_sheet_2
		self.button_style_sheet_2_not_done = self.Styles.button_style_sheet_2_not_done
		self.button_style_sheet_3 = self.Styles.button_style_sheet_3
		self.button_select_all = self.Styles.button_select_all

	def set_experiment_path(self, path):
		self.experiment_path_selection.setText(path)

	def create_new_experiment(self):

		print("Configuring new experiment...")
		self.new_exp_window = ConfigNewExperiment(self)
		self.new_exp_window.show()

	def open_directory(self):

		self.exp_dir = self.experiment_path_selection.text().replace('/', os.sep)
		print(f"Setting current directory to {self.exp_dir}...")

		wells = glob(os.sep.join([self.exp_dir,"W*"]))
		self.number_of_wells = len(wells)
		if self.number_of_wells==0:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Critical)
			msgBox.setText("No well was found in the experiment folder.\nPlease respect the W*/ nomenclature...")
			msgBox.setWindowTitle("Error")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None
		else:
			if self.number_of_wells==1:
				print(f"Found {self.number_of_wells} well...")
			elif self.number_of_wells>1:
				print(f"Found {self.number_of_wells} wells...")
			number_pos = []
			for w in wells:
				position_folders = glob(os.sep.join([w,f"{w.split(os.sep)[-2][1]}*", os.sep]))
				number_pos.append(len(position_folders))
			print(f"Number of positions per well: {number_pos}")

			self.control_panel = ControlPanel(self, self.exp_dir)
			self.control_panel.show()

	def browse_experiment_folder(self):

		"""
		Locate an experiment folder. If no configuration file is in the experiment, display a warning.
		"""

		self.foldername = str(QFileDialog.getExistingDirectory(self, 'Select directory'))
		self.experiment_path_selection.setText(self.foldername)
		if not os.path.exists(self.foldername+"/config.ini"):
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("No configuration can be found in the selected folder...")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None
	
	def closeEvent(self, event):

		"""
		Close child windows if closed.
		"""
		
		try:
			if self.control_panel:
				self.control_panel.close()
		except:
			pass
		try:
			if self.new_exp_window:
				self.new_exp_window.close()
		except:
			pass

		gc.collect()

if __name__ == "__main__":
	# import ctypes
	# myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
	# ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

	
	App = QApplication(sys.argv)
	App.setStyle("Fusion")
	window = AppInitWindow()
	sys.exit(App.exec())