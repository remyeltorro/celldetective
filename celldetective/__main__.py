#!/usr/bin/env python3
import sys
from PyQt5.QtWidgets import QApplication, QSplashScreen, QMainWindow
from PyQt5.QtGui import QPixmap
from os import sep
from celldetective.utils import get_software_location
# from PyQt5.QtCore import QEventLoop
from time import time, sleep
import os


# os.environ['QT_DEBUG_PLUGINS'] = '1'

class AppInitWindow(QMainWindow):
    """
	Initial window to set the experiment folder or create a new one.
	"""

    def __init__(self, parent=None):
        super().__init__()

        self.parent = parent
        self.Styles = Styles()
        self.init_styles()
        self.setWindowTitle("celldetective")

        self.n_threads = min([1, psutil.cpu_count()])

        try:
            subprocess.check_output('nvidia-smi')
            print('Nvidia GPU detected')
            self.use_gpu = True
        except Exception:  # this command not being found can raise quite a few different errors depending on the configuration
            print('No Nvidia GPU in system!')
            self.use_gpu = False

        self.soft_path = get_software_location()
        self.onlyInt = QIntValidator()
        self.setWindowIcon(QIcon(os.sep.join([self.soft_path, 'celldetective', 'icons', 'logo.png'])))
        center_window(self)
        self._createActions()
        self._createMenuBar()

        app = QApplication.instance()
        self.screen = app.primaryScreen()
        self.geometry = self.screen.availableGeometry()
        self.screen_width, self.screen_height = self.geometry.getRect()[-2:]

        central_widget = QWidget()
        self.vertical_layout = QVBoxLayout(central_widget)
        self.vertical_layout.setContentsMargins(15, 15, 15, 15)
        self.vertical_layout.addWidget(QLabel("Experiment folder:"))
        self.create_locate_exp_hbox()
        self.create_buttons_hbox()
        self.setCentralWidget(central_widget)
        self.reload_previous_gpu_threads()
        self.show()

    def create_locate_exp_hbox(self):

        self.locate_exp_layout = QHBoxLayout()
        self.locate_exp_layout.setContentsMargins(0, 5, 0, 0)
        self.experiment_path_selection = QLineEdit()
        self.experiment_path_selection.setAlignment(Qt.AlignLeft)
        self.experiment_path_selection.setEnabled(True)
        self.experiment_path_selection.setDragEnabled(True)
        self.experiment_path_selection.setFixedWidth(430)
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

    def _createMenuBar(self):

        menuBar = self.menuBar()
        menuBar.clear()
        # Creating menus using a QMenu object

        fileMenu = QMenu("File", self)
        fileMenu.clear()
        fileMenu.addAction(self.newExpAction)
        fileMenu.addAction(self.openAction)

        fileMenu.addMenu(self.OpenRecentAction)
        self.OpenRecentAction.clear()
        if len(self.recentFileActs) > 0:
            for i in range(len(self.recentFileActs)):
                self.OpenRecentAction.addAction(self.recentFileActs[i])

        fileMenu.addAction(self.openModels)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)
        menuBar.addMenu(fileMenu)

        OptionsMenu = QMenu("Options", self)
        OptionsMenu.addAction(self.MemoryAndThreadsAction)
        menuBar.addMenu(OptionsMenu)

        helpMenu = QMenu("Help", self)
        helpMenu.clear()
        helpMenu.addAction(self.DocumentationAction)
        helpMenu.addAction(self.SoftwareAction)
        helpMenu.addSeparator()
        helpMenu.addAction(self.AboutAction)
        menuBar.addMenu(helpMenu)

    # editMenu = menuBar.addMenu("&Edit")
    # helpMenu = menuBar.addMenu("&Help")

    def _createActions(self):
        # Creating action using the first constructor
        # self.newAction = QAction(self)
        # self.newAction.setText("&New")
        # Creating actions using the second constructor
        self.openAction = QAction('Open...', self)
        self.openAction.setShortcut("Ctrl+O")
        self.openAction.setShortcutVisibleInContextMenu(True)

        self.MemoryAndThreadsAction = QAction('Memory & Threads...')

        self.newExpAction = QAction('New', self)
        self.newExpAction.setShortcut("Ctrl+N")
        self.newExpAction.setShortcutVisibleInContextMenu(True)
        self.exitAction = QAction('Exit', self)

        self.openModels = QAction('Open Models Location')
        self.openModels.setShortcut("Ctrl+L")
        self.openModels.setShortcutVisibleInContextMenu(True)

        self.OpenRecentAction = QMenu('Open Recent')
        self.reload_previous_experiments()

        self.DocumentationAction = QAction("Documentation", self)
        self.DocumentationAction.setShortcut("Ctrl+D")
        self.DocumentationAction.setShortcutVisibleInContextMenu(True)

        self.SoftwareAction = QAction("Software", self)  # 1st arg icon(MDI6.information)
        self.AboutAction = QAction("About celldetective", self)

        # self.DocumentationAction.triggered.connect(self.load_previous_config)
        self.openAction.triggered.connect(self.open_experiment)
        self.newExpAction.triggered.connect(self.create_new_experiment)
        self.exitAction.triggered.connect(self.close)
        self.openModels.triggered.connect(self.open_models_folder)
        self.AboutAction.triggered.connect(self.open_about_window)
        self.MemoryAndThreadsAction.triggered.connect(self.set_memory_and_threads)

        self.DocumentationAction.triggered.connect(self.open_documentation)

    def reload_previous_gpu_threads(self):

        self.recentFileActs = []
        self.threads_config_path = os.sep.join([self.soft_path, 'celldetective', 'threads.json'])
        print(self.threads_config_path)
        if os.path.exists(self.threads_config_path):
            with open(self.threads_config_path, 'r') as f:
                self.threads_config = json.load(f)
            if 'use_gpu' in self.threads_config:
                self.use_gpu = bool(self.threads_config['use_gpu'])
            if 'n_threads' in self.threads_config:
                self.n_threads = int(self.threads_config['n_threads'])

    def reload_previous_experiments(self):

        recentExps = []
        self.recentFileActs = []
        if os.path.exists(os.sep.join([self.soft_path, 'celldetective', 'recent.txt'])):
            recentExps = open(os.sep.join([self.soft_path, 'celldetective', 'recent.txt']), 'r')
            recentExps = recentExps.readlines()
            recentExps = [r.strip() for r in recentExps]
            recentExps.reverse()
            recentExps = list(dict.fromkeys(recentExps))
            self.recentFileActs = [QAction(r, self) for r in recentExps]
            for r in self.recentFileActs:
                r.triggered.connect(lambda checked, item=r: self.load_recent_exp(item.text()))

    def set_memory_and_threads(self):

        print('setting memory and threads')

        self.ThreadsWidget = QWidget()
        self.ThreadsWidget.setWindowTitle("Threads")
        layout = QVBoxLayout()
        self.ThreadsWidget.setLayout(layout)

        self.threads_le = QLineEdit(str(self.n_threads))
        self.threads_le.setValidator(self.onlyInt)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Parallel threads: '), 33)
        hbox.addWidget(self.threads_le, 66)
        layout.addLayout(hbox)

        self.use_gpu_checkbox = QCheckBox()
        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel('Use GPU: '), 33)
        hbox2.addWidget(self.use_gpu_checkbox, 66)
        layout.addLayout(hbox2)
        if self.use_gpu:
            self.use_gpu_checkbox.setChecked(True)

        self.validateThreadBtn = QPushButton('Submit')
        self.validateThreadBtn.setStyleSheet(self.button_style_sheet)
        self.validateThreadBtn.clicked.connect(self.set_threads)
        layout.addWidget(self.validateThreadBtn)
        center_window(self.ThreadsWidget)
        self.ThreadsWidget.show()

    def set_threads(self):
        self.n_threads = int(self.threads_le.text())
        print(self.n_threads)
        self.use_gpu = bool(self.use_gpu_checkbox.isChecked())
        dico = {"use_gpu": self.use_gpu, "n_threads": self.n_threads}
        with open(self.threads_config_path, 'w') as f:
            json.dump(dico, f, indent=4)
        self.ThreadsWidget.close()

    def open_experiment(self):
        print('ok')
        self.browse_experiment_folder()
        if self.experiment_path_selection.text() != '':
            self.open_directory()

    def load_recent_exp(self, path):
        print('loading?')
        print('you selected path ', path)
        self.experiment_path_selection.setText(path)
        self.open_directory()

    def open_about_window(self):
        self.about_wdw = AboutWidget()
        self.about_wdw.show()

    def open_documentation(self):
        doc_url = QUrl('https://celldetective.readthedocs.io/')
        QDesktopServices.openUrl(doc_url)

    def open_models_folder(self):
        path = os.sep.join([self.soft_path, 'celldetective', 'models', os.sep])
        try:
            subprocess.Popen(f'explorer {os.path.realpath(path)}')
        except:

            try:
                os.system('xdg-open "%s"' % path)
            except:
                return None

    # os.system(f'start {os.path.realpath(path)}')

    def create_buttons_hbox(self):

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setContentsMargins(30, 15, 30, 5)
        self.new_exp_button = QPushButton("New")
        self.new_exp_button.clicked.connect(self.create_new_experiment)
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
        if (os.path.exists(text)) and os.path.exists(os.sep.join([text, "config.ini"])):
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

        wells = glob(os.sep.join([self.exp_dir, "W*"]))
        self.number_of_wells = len(wells)
        if self.number_of_wells == 0:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("No well was found in the experiment folder.\nPlease respect the W*/ nomenclature...")
            msgBox.setWindowTitle("Error")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return None
        else:
            if self.number_of_wells == 1:
                print(f"Found {self.number_of_wells} well...")
            elif self.number_of_wells > 1:
                print(f"Found {self.number_of_wells} wells...")
            number_pos = []
            for w in wells:
                position_folders = glob(os.sep.join([w, f"{w.split(os.sep)[-2][1]}*", os.sep]))
                number_pos.append(len(position_folders))
            print(f"Number of positions per well: {number_pos}")

            with open(os.sep.join([self.soft_path, 'celldetective', 'recent.txt']), 'a+') as f:
                f.write(self.exp_dir + '\n')

            self.control_panel = ControlPanel(self, self.exp_dir)
            self.control_panel.show()

            self.reload_previous_experiments()
            self._createMenuBar()

    def browse_experiment_folder(self):

        """
		Locate an experiment folder. If no configuration file is in the experiment, display a warning.
		"""

        self.foldername = str(QFileDialog.getExistingDirectory(self, 'Select directory'))
        if self.foldername != '':
            self.experiment_path_selection.setText(self.foldername)
        else:
            return None
        if not os.path.exists(self.foldername + "/config.ini"):
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("No configuration can be found in the selected folder...")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                self.experiment_path_selection.setText('')
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
    splash = True

    App = QApplication(sys.argv)
    # App.setWindowIcon(QIcon(os.sep.join([get_software_location(),'celldetective','icons','mexican-hat.png'])))
    App.setStyle("Fusion")

    if splash:
        start = time()
        splash_pix = QPixmap(sep.join([get_software_location(), 'celldetective', 'icons', 'splash.png']))
        splash = QSplashScreen(splash_pix)
        splash.setMask(splash_pix.mask())
        splash.show()
        # App.processEvents(QEventLoop.AllEvents, 300)
        while time() - start < 1:
            sleep(0.001)
            App.processEvents()

    from PyQt5.QtWidgets import QFileDialog, QWidget, QVBoxLayout, QCheckBox, QHBoxLayout, QLabel, QLineEdit, \
        QPushButton, QMessageBox, QMenu, QAction
    from PyQt5.QtCore import Qt, QUrl
    from PyQt5.QtGui import QIcon, QDesktopServices, QIntValidator
    from glob import glob
    from superqt.fonticon import icon
    from fonticon_mdi6 import MDI6
    import gc
    from celldetective.gui import Styles, ControlPanel, ConfigNewExperiment
    from celldetective.gui.gui_utils import center_window
    import subprocess
    import os
    from celldetective.gui.about import AboutWidget
    import psutil
    import subprocess
    import json

    window = AppInitWindow(App)

    if splash:
        splash.finish(window)

    sys.exit(App.exec())