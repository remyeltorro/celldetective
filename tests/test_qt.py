import pytest
from PyQt5 import QtCore
from celldetective.gui.InitWindow import AppInitWindow
import time
import os

abs_path = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]])
print(abs_path)

@pytest.fixture
def app(qtbot):
	test_app = AppInitWindow()
	qtbot.addWidget(test_app)
	return test_app

# def test_launch_demo(app, qtbot):
# 	app.experiment_path_selection.setText(abs_path + os.sep + 'examples/demo')
# 	qtbot.mouseClick(app.validate_button, QtCore.Qt.LeftButton)

# def test_preprocessing_panel(app, qtbot):

# 	app.experiment_path_selection.setText(abs_path + os.sep + 'examples/demo')
# 	qtbot.mouseClick(app.validate_button, QtCore.Qt.LeftButton)

# 	qtbot.mouseClick(app.control_panel.PreprocessingPanel.collapse_btn, QtCore.Qt.LeftButton)
# 	qtbot.mouseClick(app.control_panel.PreprocessingPanel.fit_correction_layout.add_correction_btn, QtCore.Qt.LeftButton)
# 	qtbot.mouseClick(app.control_panel.PreprocessingPanel.collapse_btn, QtCore.Qt.LeftButton)

def test_app(app, qtbot):

	# Set an experiment folder and open
	app.experiment_path_selection.setText(os.sep.join([abs_path,'examples','demo']))
	qtbot.mouseClick(app.validate_button, QtCore.Qt.LeftButton)

	# Set a position
	app.control_panel.position_list.setCurrentIndex(1)
	app.control_panel.update_position_options()

	# View stacl
	qtbot.mouseClick(app.control_panel.view_stack_btn, QtCore.Qt.LeftButton)
	#qtbot.wait(1000)
	app.control_panel.viewer.close()

	# Expand process block
	qtbot.mouseClick(app.control_panel.ProcessEffectors.collapse_btn, QtCore.Qt.LeftButton)

	# Use Threshold Config Wizard
	qtbot.mouseClick(app.control_panel.ProcessEffectors.upload_model_btn, QtCore.Qt.LeftButton)
	qtbot.wait(1000)
	qtbot.mouseClick(app.control_panel.ProcessEffectors.SegModelLoader.threshold_config_button, QtCore.Qt.LeftButton)
	app.control_panel.ProcessEffectors.SegModelLoader.ThreshWizard.close()
	app.control_panel.ProcessEffectors.SegModelLoader.close()

	# Check segmentation with napari
	#qtbot.mouseClick(app.control_panel.ProcessEffectors.check_seg_btn, QtCore.Qt.LeftButton)
	# close napari?

	# Train model
	qtbot.mouseClick(app.control_panel.ProcessEffectors.train_btn, QtCore.Qt.LeftButton)
	qtbot.wait(1000)
	app.control_panel.ProcessEffectors.ConfigSegmentationTrain.close()

	# Config tracking
	qtbot.mouseClick(app.control_panel.ProcessEffectors.track_config_btn, QtCore.Qt.LeftButton)
	qtbot.wait(1000)
	app.control_panel.ProcessEffectors.ConfigTracking.close()

	# Config measurements
	qtbot.mouseClick(app.control_panel.ProcessEffectors.measurements_config_btn, QtCore.Qt.LeftButton)
	qtbot.wait(1000)
	app.control_panel.ProcessEffectors.ConfigMeasurements.close()

	# Classifier widget
	qtbot.mouseClick(app.control_panel.ProcessEffectors.classify_btn, QtCore.Qt.LeftButton)
	qtbot.wait(1000)
	app.control_panel.ProcessEffectors.ClassifierWidget.close()

	# Config signal annotator
	qtbot.mouseClick(app.control_panel.ProcessEffectors.config_signal_annotator_btn, QtCore.Qt.LeftButton)
	qtbot.mouseClick(app.control_panel.ProcessEffectors.ConfigSignalAnnotator.rgb_btn, QtCore.Qt.LeftButton)
	qtbot.wait(1000)
	app.control_panel.ProcessEffectors.ConfigSignalAnnotator.close()

	# Signal annotator widget
	qtbot.mouseClick(app.control_panel.ProcessEffectors.check_signals_btn, QtCore.Qt.LeftButton)
	qtbot.wait(1000)
	app.control_panel.ProcessEffectors.SignalAnnotator.close()

	# Table widget
	qtbot.mouseClick(app.control_panel.ProcessEffectors.view_tab_btn, QtCore.Qt.LeftButton)
	qtbot.wait(1000)
	app.control_panel.ProcessEffectors.tab_ui.close()

	#qtbot.mouseClick(app.control_panel.PreprocessingPanel.fit_correction_layout.add_correction_btn, QtCore.Qt.LeftButton)
	qtbot.mouseClick(app.control_panel.ProcessEffectors.collapse_btn, QtCore.Qt.LeftButton)



# def test_click(app, qtbot):
# 	qtbot.mouseClick(app.new_exp_button, QtCore.Qt.LeftButton)
#	qtbot.wait(10000)
