from multiprocessing import Queue
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QProgressBar
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal, QThreadPool, QSize, Qt
from celldetective.gui.gui_utils import center_window
import time
import math

class ProgressWindow(QDialog):

	def __init__(self, process=None, parent_window=None):
		QDialog.__init__(self)

		self.setWindowTitle('Progress')
		self.__process = process
		self.parent_window = parent_window

		#self.__btn_run = QPushButton("Start")
		self.__btn_stp = QPushButton("Cancel")
		self.__label   = QLabel("Idle")
		self.time_left_lbl = QLabel('')
		self.progress_bar = QProgressBar()
		self.progress_bar.setValue(0)

		self.__runner  = Runner(process=self.__process,progress_bar=self.progress_bar, parent_window=self.parent_window, remaining_time_label=self.time_left_lbl)
		self.pool = QThreadPool.globalInstance()
		print("Multithreading with maximum %d threads" % self.pool.maxThreadCount())

		#self.__btn_run.clicked.connect(self.__run_net)
		self.__btn_stp.clicked.connect(self.__stp_net)
		self.__runner.signals.finished.connect(self.__on_finished)
		self.__runner.signals.finished.connect(self.__on_error)
		self.__runner.signals.update.connect(self.progress_bar.setValue)
		self.__runner.signals.update_time.connect(self.time_left_lbl.setText)

		self.__btn_stp.setDisabled(True)

		self.layout = QVBoxLayout()
		self.layout.addWidget(self.time_left_lbl)
		self.layout.addWidget(self.progress_bar)
		self.btn_layout = QHBoxLayout()
		self.btn_layout.addWidget(self.__btn_stp)
		self.btn_layout.addWidget(self.__label)
		self.layout.addLayout(self.btn_layout)

		self.setLayout(self.layout)
		self.setFixedSize(QSize(250, 100))
		self.__run_net()
		self.setModal(True)
		center_window(self)

	def closeEvent(self, evnt):
		# if self._want_to_close:
		# 	super(MyDialog, self).closeEvent(evnt)
		# else:
		evnt.ignore()
		self.setWindowState(Qt.WindowMinimized)

	def __run_net(self):
		#self.__btn_run.setDisabled(True)
		self.__btn_stp.setEnabled(True)
		self.__label.setText("Running...")
		self.pool.start(self.__runner)

	def __stp_net(self):
		self.__runner.close()
		print('Job cancelled... Abort.')
		self.reject()

	def __on_finished(self):
		self.__btn_stp.setDisabled(True)
		self.__label.setText("Finished!")
		self.__runner.close()
		self.accept()

	def __on_error(self):
		self.__btn_stp.setDisabled(True)
		self.__label.setText("Error")
		self.__runner.close()
		self.accept()

class Runner(QRunnable):

	def __init__(self, process=None, progress_bar=None, remaining_time_label=None, parent_window=None):
		QRunnable.__init__(self)

		self.progress_bar = progress_bar
		self.parent_window = parent_window
		self.remaining_time_label = remaining_time_label
		self.__queue = Queue()
		self.__process = process(self.__queue, parent_window=self.parent_window)
		self.signals = RunnerSignal()

	def run(self):
		self.__process.start()
		while True:
			try:
				data = self.__queue.get()
				if len(data)==2:
					progress, time = data
					time = int(time)
					remaining_minutes = time // 60
					remaining_seconds = int(time % 60)
					time_left = "About "+str(remaining_minutes)+" min and "+str(remaining_seconds)+" s remaining"
					if remaining_minutes == 0:
						time_left = "About "+str(remaining_seconds)+" s remaining"
					self.signals.update_time.emit(time_left)
					if isinstance(progress, float | int):
						self.signals.update.emit(math.ceil(progress))
				if data == "finished":
					self.signals.finished.emit()
					break
				elif data == "error":
					self.signals.error.emit()
			except Exception as e:
				print(e)
				pass

	def close(self):
		self.__process.end_process()


class RunnerSignal(QObject):

	update = pyqtSignal(int)
	update_time = pyqtSignal(str)
	finished = pyqtSignal()
	error = pyqtSignal()