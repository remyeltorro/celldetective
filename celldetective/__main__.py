#!/usr/bin/env python3
import sys
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap
from os import sep
from celldetective.utils import get_software_location
from time import time, sleep
#os.environ['QT_DEBUG_PLUGINS'] = '1'

if __name__ == "__main__":

	splash=True
	print('Loading the libraries...')

	App = QApplication(sys.argv)
	App.setStyle("Fusion")

	if splash:
		start = time()
		splash_pix = QPixmap(sep.join([get_software_location(),'celldetective','icons','splash.png']))
		splash = QSplashScreen(splash_pix)
		splash.setMask(splash_pix.mask())
		splash.show()
		#App.processEvents(QEventLoop.AllEvents, 300)
		while time() - start < 1:
			sleep(0.001)
			App.processEvents()

	from PyQt5.QtWidgets import QFileDialog, QWidget, QVBoxLayout, QCheckBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QMenu, QAction
	from PyQt5.QtGui import QIcon, QDesktopServices, QIntValidator
	from celldetective.gui.InitWindow import AppInitWindow

	print('Libraries successfully loaded...')

	window = AppInitWindow(App)

	if splash:
		splash.finish(window)

	sys.exit(App.exec())