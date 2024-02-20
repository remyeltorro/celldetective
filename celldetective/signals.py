import numpy as np
import os
import subprocess
import json

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model,clone_model
from tensorflow.config.experimental import list_physical_devices, set_memory_growth
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dense, Activation, Add, MaxPooling1D, Dropout, GlobalAveragePooling1D, Concatenate, ZeroPadding1D, Flatten
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import jaccard_score, balanced_accuracy_score, precision_score, recall_score
from scipy.interpolate import interp1d
from scipy.ndimage import shift

from celldetective.io import get_signal_models_list, locate_signal_model
from celldetective.tracking import clean_trajectories
from celldetective.utils import regression_plot, train_test_split, compute_weights
import matplotlib.pyplot as plt
from natsort import natsorted
from glob import glob
import shutil
import random
from celldetective.utils import color_from_status, color_from_class
from math import floor, ceil
from scipy.optimize import curve_fit
import time
import math

abs_path = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],'celldetective'])


class TimeHistory(Callback):
	
	def on_train_begin(self, logs={}):
		self.times = []

	def on_epoch_begin(self, epoch, logs={}):
		self.epoch_time_start = time.time()

	def on_epoch_end(self, epoch, logs={}):
		self.times.append(time.time() - self.epoch_time_start)


def analyze_signals(trajectories, model, interpolate_na=True,
					selected_signals=None,
					column_labels = {'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'},
					plot_outcome=False, output_dir=None):

	"""

	Perform signal analysis on trajectories using a trained model.

	Parameters
	----------
	trajectories : pandas.DataFrame
		DataFrame containing the trajectories data.
	model : str
		Name of the trained signal detection model.
	interpolate_na : bool, optional
		Flag indicating whether to interpolate missing values in the trajectories. Default is True.
	selected_signals : list, optional
		List of selected signals to be used for analysis. If None, the required signals specified in the model
		configuration will be automatically selected based on the available signals in the trajectories DataFrame.
		Default is None.
	column_labels : dict, optional
		Dictionary containing the column labels for the trajectories DataFrame. Default is
		{'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.

	Returns
	-------
	pandas.DataFrame
		DataFrame containing the trajectories data with additional columns for the predicted class labels and
		time of interest.

	Notes
	-----
	This function performs signal analysis on the trajectories using a trained signal detection model. The model
	should be located in the models directory specified in the software. Call io.get_signal_models_list to get the list of models.
	The trajectories DataFrame is expected to contain the required columns specified in the model configuration. If 
	the `selected_signals` parameter is not provided, the function automatically selects the required signals 
	based on the available signals in the trajectories DataFrame. The missing values in the trajectories can be interpolated 
	if `interpolate_na` is set to True.

	Examples
	--------
	>>> signal_analysis(trajectories_df, model='my_model', interpolate_na=True, selected_signals=['signal1', 'signal2'])
	# Perform signal analysis on trajectories using a trained model.
	
	"""


	model_path = locate_signal_model(model)
	complete_path = model_path #+model
	complete_path = rf"{complete_path}"
	model_config_path = os.sep.join([complete_path,'config_input.json'])
	model_config_path = rf"{model_config_path}"
	assert os.path.exists(complete_path),f'Model {model} could not be located in folder {model_path}... Abort.'
	assert os.path.exists(model_config_path),f'Model configuration could not be located in folder {model_path}... Abort.'

	available_signals = list(trajectories.columns)
	print('The available_signals are : ',available_signals)

	f = open(model_config_path)
	config = json.load(f)
	required_signals = config["channels"]

	try:
		label = config['label']
		if label=='':
			label = None
	except:
		label = None
	
	if selected_signals is None:
		selected_signals = []
		for s in required_signals:
			pattern_test = [s in a or s==a for a in available_signals]
			print(f'Pattern test for signal {s}: ', pattern_test)
			assert np.any(pattern_test),f'No signal matches with the requirements of the model {required_signals}. Please pass the signals manually with the argument selected_signals or add measurements. Abort.'
			valid_columns = np.array(available_signals)[np.array(pattern_test)]
			if len(valid_columns)==1:
				selected_signals.append(valid_columns[0])
			else:
				#print(test_number_of_nan(trajectories, valid_columns))
				print(f'Found several candidate signals: {valid_columns}')
				for vc in natsorted(valid_columns):
					if 'circle' in vc:
						selected_signals.append(vc)
						break
				else:
					selected_signals.append(valid_columns[0])
				# do something more complicated in case of one to many columns
				#pass
	else:
		assert len(selected_signals)==len(required_signals),f'Mismatch between the number of required signals {required_signals} and the provided signals {selected_signals}... Abort.'

	print(f'The following channels will be passed to the model: {selected_signals}')
	trajectories_clean = clean_trajectories(trajectories, interpolate_na=interpolate_na, interpolate_position_gaps=interpolate_na, column_labels=column_labels)

	max_signal_size = int(trajectories_clean[column_labels['time']].max()) + 2
	tracks = trajectories_clean[column_labels['track']].unique()
	signals = np.zeros((len(tracks),max_signal_size, len(selected_signals)))

	for i,(tid,group) in enumerate(trajectories_clean.groupby(column_labels['track'])):
		frames = group[column_labels['time']].to_numpy().astype(int)
		for j,col in enumerate(selected_signals):
			signal = group[col].to_numpy()
			signals[i,frames,j] = signal

	# for i in range(5):
	# 	print('pre model')
	# 	plt.plot(signals[i,:,0])
	# 	plt.show()

	model = SignalDetectionModel(pretrained=complete_path)
	print('signal shape: ', signals.shape)

	classes = model.predict_class(signals)
	times_recast = model.predict_time_of_interest(signals)

	if label is None:
		class_col = 'class'
		time_col = 't0'
		status_col = 'status'
	else:
		class_col = 'class_'+label
		time_col = 't_'+label
		status_col = 'status_'+label

	for i,(tid,group) in enumerate(trajectories.groupby(column_labels['track'])):
		indices = group.index
		trajectories.loc[indices,class_col] = classes[i]
		trajectories.loc[indices,time_col] = times_recast[i]
	print('Done.')

	for tid, group in trajectories.groupby(column_labels['track']):
		
		indices = group.index
		t0 = group[time_col].to_numpy()[0]
		cclass = group[class_col].to_numpy()[0]
		timeline = group[column_labels['time']].to_numpy()
		status = np.zeros_like(timeline)
		if t0 > 0:
			status[timeline>=t0] = 1.
		if cclass==2:
			status[:] = 2
		if cclass>2:
			status[:] = 42
		status_color = [color_from_status(s) for s in status]
		class_color = [color_from_class(cclass) for i in range(len(status))]

		trajectories.loc[indices, status_col] = status
		trajectories.loc[indices, 'status_color'] = status_color
		trajectories.loc[indices, 'class_color'] = class_color

	if plot_outcome:
		fig,ax = plt.subplots(1,len(selected_signals), figsize=(10,5))
		for i,s in enumerate(selected_signals):
			for k,(tid,group) in enumerate(trajectories.groupby(column_labels['track'])):
				cclass = group[class_col].to_numpy()[0]
				t0 = group[time_col].to_numpy()[0]
				timeline = group[column_labels['time']].to_numpy()
				if cclass==0:
					if len(selected_signals)>1:
						ax[i].plot(timeline - t0, group[s].to_numpy(),c='tab:blue',alpha=0.1)
					else:
						ax.plot(timeline - t0, group[s].to_numpy(),c='tab:blue',alpha=0.1)
		if len(selected_signals)>1:				
			for a,s in zip(ax,selected_signals):
				a.set_title(s)
				a.set_xlabel(r'time - t$_0$ [frame]')
				a.spines['top'].set_visible(False)
				a.spines['right'].set_visible(False)
		else:
			ax.set_title(s)
			ax.set_xlabel(r'time - t$_0$ [frame]')
			ax.spines['top'].set_visible(False)
			ax.spines['right'].set_visible(False)			
		plt.tight_layout()
		if output_dir is not None:
			plt.savefig(output_dir+'signal_collapse.png',bbox_inches='tight',dpi=300)
		plt.pause(3)
		plt.close()

	return trajectories

def analyze_signals_at_position(pos, model, mode, use_gpu=True):
	
	pos = pos.replace('\\','/')
	pos = rf"{pos}"
	assert os.path.exists(pos),f'Position {pos} is not a valid path.'
	if not pos.endswith('/'):
		pos += '/'

	script_path = os.sep.join([abs_path, 'scripts', 'analyze_signals.py'])
	cmd = f'python "{script_path}" --pos "{pos}" --model "{model}" --mode "{mode}" --use_gpu "{use_gpu}"'
	subprocess.call(cmd, shell=True)
	
	return None


class SignalDetectionModel(object):
	
	def __init__(self, path=None, pretrained=None, channel_option=["live_nuclei_channel"], model_signal_length=128, n_channels=1, 
				n_conv=2, n_classes=3, dense_collection=512, dropout_rate=0.1, label=''):
		
		self.prep_gpu()

		self.model_signal_length = model_signal_length
		self.channel_option = channel_option
		self.pretrained = pretrained
		self.n_channels = n_channels
		self.n_conv = n_conv
		self.n_classes = n_classes
		self.dense_collection = dense_collection
		self.dropout_rate = dropout_rate
		self.label = label


		if self.pretrained is not None:
			print(f"Load pretrained models from {path}...")
			self.load_pretrained_model()
		else:
			print("Create models from scratch...")
			self.create_models_from_scratch()

	
	def load_pretrained_model(self):
		
		"""
		
		Load model from pretrained path and set as current model
		
		"""
		
		# Load keras model
		try:
			self.model_class = load_model(os.sep.join([self.pretrained,"classifier.h5"]),compile=False)
			self.model_class.load_weights(os.sep.join([self.pretrained,"classifier.h5"]))
			print("Classifier successfully loaded...")
		except Exception as e:
			print(f"Error {e}...")
			self.model_class = None
		try:
			self.model_reg = load_model(os.sep.join([self.pretrained,"regressor.h5"]),compile=False)
			self.model_reg.load_weights(os.sep.join([self.pretrained,"regressor.h5"]))
			print("Regressor successfully loaded...")
		except Exception as e:
			print(f"Error {e}...")
			self.model_reg = None

		# load config
		with open(os.sep.join([self.pretrained,"config_input.json"])) as config_file:
			model_config = json.load(config_file)

		req_channels = model_config["channels"]
		print(f"Required channels read from pretrained model: {req_channels}")
		self.channel_option = req_channels
		if 'normalize' in model_config:
			self.normalize = model_config['normalize']
		if 'normalization_percentile' in model_config:
			self.normalization_percentile = model_config['normalization_percentile']
		if 'normalization_values' in model_config:
			self.normalization_values = model_config['normalization_values']
		if 'normalization_percentile' in model_config:
			self.normalization_clip = model_config['normalization_clip']
		if 'label' in model_config:
			self.label = model_config['label']

		self.n_channels = self.model_class.layers[0].input_shape[0][-1]
		self.model_signal_length = self.model_class.layers[0].input_shape[0][-2]
		self.n_classes = self.model_class.layers[-1].output_shape[-1]

		assert self.model_class.layers[0].input_shape[0] == self.model_reg.layers[0].input_shape[0], f"mismatch between input shape of classification: {self.model_class.layers[0].input_shape[0]} and regression {self.model_reg.layers[0].input_shape[0]} models... Error."


	def create_models_from_scratch(self):

		"""
		
		Generate new ResNet models for classification and regression with the chosen specifications

		"""

		self.model_class = ResNetModelCurrent(n_channels=self.n_channels,
									n_slices=self.n_conv,
									n_classes = self.n_classes,
									dense_collection=self.dense_collection,
									dropout_rate=self.dropout_rate, 
									header="classifier", 
									model_signal_length = self.model_signal_length
									)

		self.model_reg = ResNetModelCurrent(n_channels=self.n_channels,
									n_slices=self.n_conv,
									n_classes = self.n_classes,
									dense_collection=self.dense_collection,
									dropout_rate=self.dropout_rate, 
									header="regressor", 
									model_signal_length = self.model_signal_length
									)

	def prep_gpu(self):
		
		try:
			physical_devices = list_physical_devices('GPU')
			for gpu in physical_devices:
				set_memory_growth(gpu, True)
		except:
			pass
	
	def fit_from_directory(self, ds_folders, normalize=True, normalization_percentile=None, normalization_values = None, 
						  normalization_clip = None, channel_option=["live_nuclei_channel"], model_name=None, target_directory=None, 
						  augment=True, augmentation_factor=2, validation_split=0.20, test_split=0.0, batch_size = 64, epochs=300, 
						  recompile_pretrained=False, learning_rate=0.01, loss_reg="mse", loss_class = CategoricalCrossentropy(from_logits=False)):
		"""
		
		Load annotations in directory, create dataset, fit model
		
		"""

		if not hasattr(self, 'normalization_percentile'):
			self.normalization_percentile = normalization_percentile
		if not hasattr(self, 'normalization_values'):
			self.normalization_values = normalization_values
		if not hasattr(self, 'normalization_clip'):
			self.normalization_clip = normalization_clip
		print('Actual clip option:', self.normalization_clip)
		
		self.normalize = normalize
		self.normalization_percentile, self. normalization_values, self.normalization_clip =  _interpret_normalization_parameters(self.n_channels, self.normalization_percentile, self.normalization_values, self.normalization_clip)

		self.ds_folders = [rf'{d}' for d in ds_folders]
		self.batch_size = batch_size
		self.epochs = epochs
		self.validation_split = validation_split
		self.test_split = test_split
		self.augment = augment
		self.augmentation_factor = augmentation_factor
		self.model_name = rf'{model_name}'
		self.target_directory = rf'{target_directory}'
		self.model_folder = os.sep.join([self.target_directory,self.model_name])
		self.recompile_pretrained = recompile_pretrained
		self.learning_rate = learning_rate
		self.loss_reg = loss_reg
		self.loss_class = loss_class


		if not os.path.exists(self.model_folder):
			#shutil.rmtree(self.model_folder)
			os.mkdir(self.model_folder)

		self.channel_option = channel_option
		assert self.n_channels==len(self.channel_option), f'Mismatch between the channel option and the number of channels of the model...'
		
		self.list_of_sets = []
		print(self.ds_folders)
		for f in self.ds_folders:
			self.list_of_sets.extend(glob(os.sep.join([f,"*.npy"])))
		print(f"Found {len(self.list_of_sets)} annotation files...")
		self.generate_sets()

		self.train_classifier()
		self.train_regressor()

		config_input = {"channels": self.channel_option, "model_signal_length": self.model_signal_length, 'label': self.label, 'normalize': self.normalize, 'normalization_percentile': self.normalization_percentile, 'normalization_values': self.normalization_values, 'normalization_clip': self.normalization_clip}
		json_string = json.dumps(config_input)
		with open(os.sep.join([self.model_folder,"config_input.json"]), 'w') as outfile:
			outfile.write(json_string)

	def fit(self, x_train, y_time_train, y_class_train, normalize=True, normalization_percentile=None, normalization_values = None, normalization_clip = None, pad=True, validation_data=None, test_data=None, channel_option=["live_nuclei_channel","dead_nuclei_channel"], model_name=None, 
			target_directory=None, augment=True, augmentation_factor=3, validation_split=0.25, batch_size = 64, epochs=300,
			recompile_pretrained=False, learning_rate=0.001, loss_reg="mse", loss_class = CategoricalCrossentropy(from_logits=False)):

		self.normalize = normalize
		if not hasattr(self, 'normalization_percentile'):
			self.normalization_percentile = normalization_percentile
		if not hasattr(self, 'normalization_values'):
			self.normalization_values = normalization_values
		if not hasattr(self, 'normalization_clip'):
			self.normalization_clip = normalization_clip
		self.normalization_percentile, self. normalization_values, self.normalization_clip =  _interpret_normalization_parameters(self.n_channels, self.normalization_percentile, self.normalization_values, self.normalization_clip)

		self.x_train = x_train
		self.y_class_train = y_class_train
		self.y_time_train = y_time_train
		self.channel_option = channel_option
		
		assert self.n_channels==len(self.channel_option), f'Mismatch between the channel option and the number of channels of the model...'

		if pad:
			self.x_train = pad_to_model_length(self.x_train, self.model_signal_length)

		assert self.x_train.shape[1:] == (self.model_signal_length, self.n_channels), f"Shape mismatch between the provided training fluorescence signals and the model..."

		# If y-class is not one-hot encoded, encode it
		if self.y_class_train.shape[-1] != self.n_classes:
			self.class_weights = compute_weights(self.y_class_train)
			self.y_class_train = to_categorical(self.y_class_train)

		if self.normalize:
			self.y_time_train = self.y_time_train.astype(np.float32)/self.model_signal_length
			self.x_train = normalize_signal_set(self.x_train, self.channel_option, normalization_percentile=self.normalization_percentile, 
												normalization_values=self.normalization_values, normalization_clip=self.normalization_clip,
												)


		if validation_data is not None:
			try:
				self.x_val = validation_data[0]
				if pad:
					self.x_val = pad_to_model_length(self.x_val, self.model_signal_length)
				self.y_class_val = validation_data[1]
				if self.y_class_val.shape[-1] != self.n_classes:
					self.y_class_val = to_categorical(self.y_class_val)		
				self.y_time_val = validation_data[2]
				if self.normalize:
					self.y_time_val = self.y_time_val.astype(np.float32)/self.model_signal_length
					self.x_val = normalize_signal_set(self.x_val, self.channel_option, normalization_percentile=self.normalization_percentile, 
												normalization_values=self.normalization_values, normalization_clip=self.normalization_clip,
												)

			except Exception as e:
				print("Could not load validation data, error {e}...")
		else:
			self.validation_split = validation_split

		if test_data is not None:
			try:
				self.x_test = test_data[0]
				if pad:
					self.x_test = pad_to_model_length(self.x_test, self.model_signal_length)
				self.y_class_test = test_data[1]
				if self.y_class_test.shape[-1] != self.n_classes:
					self.y_class_test = to_categorical(self.y_class_test)
				self.y_time_test = test_data[2]
				if self.normalize:
					self.y_time_test = self.y_time_test.astype(np.float32)/self.model_signal_length
					self.x_test = normalize_signal_set(self.x_test, self.channel_option, normalization_percentile=self.normalization_percentile, 
												normalization_values=self.normalization_values, normalization_clip=self.normalization_clip,
												)
			except Exception as e:
				print("Could not load test data, error {e}...")


		self.batch_size = batch_size
		self.epochs = epochs
		self.augment = augment
		self.augmentation_factor = augmentation_factor
		if self.augmentation_factor==1:
			self.augment = False
		self.model_name = model_name
		self.target_directory = target_directory
		self.model_folder = os.sep.join([self.target_directory,self.model_name])
		self.recompile_pretrained = recompile_pretrained
		self.learning_rate = learning_rate
		self.loss_reg = loss_reg
		self.loss_class = loss_class

		if os.path.exists(self.model_folder):
			shutil.rmtree(self.model_folder)
		os.mkdir(self.model_folder)

		self.train_classifier()
		self.train_regressor()

	def predict_class(self, x, normalize=True, pad=True, return_one_hot=False, interpolate=True):

		self.x = np.copy(x)
		self.normalize = normalize
		self.pad = pad
		self.return_one_hot = return_one_hot
		# self.max_relevant_time = np.shape(self.x)[1]
		# print(f'Max relevant time: {self.max_relevant_time}')

		if self.pad:
			self.x = pad_to_model_length(self.x, self.model_signal_length)

		if self.normalize:
			self.x = normalize_signal_set(self.x, self.channel_option, normalization_percentile=self.normalization_percentile, 
												normalization_values=self.normalization_values, normalization_clip=self.normalization_clip,
												)

		# implement auto interpolation here!!
		#self.x = self.interpolate_signals(self.x)

		# for i in range(5):
		# 	plt.plot(self.x[i,:,0])
		# 	plt.show()

		assert self.x.shape[-1] == self.model_class.layers[0].input_shape[0][-1], f"Shape mismatch between the input shape and the model input shape..."
		assert self.x.shape[-2] == self.model_class.layers[0].input_shape[0][-2], f"Shape mismatch between the input shape and the model input shape..."

		self.class_predictions_one_hot = self.model_class.predict(self.x)
		self.class_predictions = self.class_predictions_one_hot.argmax(axis=1)

		if self.return_one_hot:
			return self.class_predictions_one_hot
		else:
			return self.class_predictions

	def predict_time_of_interest(self, x, class_predictions=None, normalize=True, pad=True):

		self.x = np.copy(x)
		self.normalize = normalize
		self.pad = pad
		# self.max_relevant_time = np.shape(self.x)[1]
		# print(f'Max relevant time: {self.max_relevant_time}')

		if class_predictions is not None:
			self.class_predictions = class_predictions

		if self.pad:
			self.x = pad_to_model_length(self.x, self.model_signal_length)

		if self.normalize:
			self.x = normalize_signal_set(self.x, self.channel_option, normalization_percentile=self.normalization_percentile,
												normalization_values=self.normalization_values, normalization_clip=self.normalization_clip,
												)

		assert self.x.shape[-1] == self.model_reg.layers[0].input_shape[0][-1], f"Shape mismatch between the input shape and the model input shape..."
		assert self.x.shape[-2] == self.model_reg.layers[0].input_shape[0][-2], f"Shape mismatch between the input shape and the model input shape..."

		if np.any(self.class_predictions==0):
			self.time_predictions = self.model_reg.predict(self.x[self.class_predictions==0])*self.model_signal_length
			self.time_predictions = self.time_predictions[:,0]
			self.time_predictions_recast = np.zeros(len(self.x)) - 1.
			self.time_predictions_recast[self.class_predictions==0] = self.time_predictions
		else:
			self.time_predictions_recast = np.zeros(len(self.x)) - 1.
		return self.time_predictions_recast

	def interpolate_signals(self, x_set):

		print(x_set.shape)
		for i in range(len(x_set)):
			for k in range(x_set.shape[-1]):
				x = x_set[i,:,k]
				not_nan = np.logical_not(np.isnan(x))
				indices = np.arange(len(x))
				interp = interp1d(indices[not_nan], x[not_nan],fill_value=(0.,0.), bounds_error=False)
				x_set[i,:,k] = interp(indices)
		return x_set


		
	def train_classifier(self):

		"""
		
		Train a classifier model to detect events in the signals.

		"""

		# if pretrained model
		if self.pretrained is not None:
			# if recompile
			if self.recompile_pretrained:
				print('Recompiling the pretrained classifier model... Warning, this action reinitializes all the weights; are you sure that this is what you intended?')
				self.model_class.set_weights(clone_model(self.model_class).get_weights())
				self.model_class.compile(optimizer=Adam(learning_rate=self.learning_rate), 
							  loss=self.loss_class, 
							  metrics=['accuracy', Precision(), Recall()])
		else:
			print("Compiling the classifier...")
			self.model_class.compile(optimizer=Adam(learning_rate=self.learning_rate), 
						  loss=self.loss_class, 
						  metrics=['accuracy', Precision(), Recall()])
			
		self.gather_callbacks("classifier")


		# for i in range(30):
		# 	for j in range(self.x_train.shape[-1]):
		# 		plt.plot(self.x_train[i,:,j])
		# 	plt.show()

		if hasattr(self, 'x_val'):
			self.history_classifier = self.model_class.fit(x=self.x_train,
								y=self.y_class_train,
								batch_size=self.batch_size,
								class_weight=self.class_weights,
								epochs=self.epochs, 
								validation_data=(self.x_val,self.y_class_val),
								callbacks=self.cb,
								verbose=1)
		else:
			self.history_classifier = self.model_class.fit(x=self.x_train,
								y=self.y_class_train,
								batch_size=self.batch_size,
								class_weight=self.class_weights,
								epochs=self.epochs, 
								callbacks=self.cb,
								validation_split = self.validation_split,
								verbose=1)			

		self.plot_model_history(mode="classifier")

		# Set current classification model as the best model
		self.model_class = load_model(os.sep.join([self.model_folder,"classifier.h5"]))
		self.model_class.load_weights(os.sep.join([self.model_folder,"classifier.h5"]))
		
		self.dico = {"history_classifier": self.history_classifier, "execution_time_classifier": self.cb[-1].times}

		if hasattr(self, 'x_test'):
			
			predictions = self.model_class.predict(self.x_test).argmax(axis=1)
			ground_truth = self.y_class_test.argmax(axis=1)
			assert predictions.shape==ground_truth.shape,"Mismatch in shape between the predictions and the ground truth..."
			
			title="Test data"
			IoU_score = jaccard_score(ground_truth, predictions, average=None)
			balanced_accuracy = balanced_accuracy_score(ground_truth, predictions)
			precision = precision_score(ground_truth, predictions, average=None)
			recall = recall_score(ground_truth, predictions, average=None)

			print(f"Test IoU score: {IoU_score}")
			print(f"Test Balanced accuracy score: {balanced_accuracy}")
			print(f'Test Precision: {precision}')
			print(f'Test Recall: {recall}')

			# Confusion matrix on test set
			results = confusion_matrix(ground_truth,predictions)
			self.dico.update({"test_IoU": IoU_score, "test_balanced_accuracy": balanced_accuracy, "test_confusion": results, 'test_precision': precision, 'test_recall': recall})

			try:
				plot_confusion_matrix(results, ["dead","alive","miscellaneous"], output_dir=self.model_folder+os.sep, title=title)
			except:
				pass
			print("Test set: ",classification_report(ground_truth,predictions))

		if hasattr(self, 'x_val'):
			predictions = self.model_class.predict(self.x_val).argmax(axis=1)
			ground_truth = self.y_class_val.argmax(axis=1)
			assert ground_truth.shape==predictions.shape,"Mismatch in shape between the predictions and the ground truth..."
			title="Validation data"

			# Validation scores
			IoU_score = jaccard_score(ground_truth, predictions, average=None)
			balanced_accuracy = balanced_accuracy_score(ground_truth, predictions)
			precision = precision_score(ground_truth, predictions, average=None)
			recall = recall_score(ground_truth, predictions, average=None)

			print(f"Validation IoU score: {IoU_score}")
			print(f"Validation Balanced accuracy score: {balanced_accuracy}")
			print(f'Validation Precision: {precision}')
			print(f'Validation Recall: {recall}')

			# Confusion matrix on validation set
			results = confusion_matrix(ground_truth,predictions)
			self.dico.update({"val_IoU": IoU_score, "val_balanced_accuracy": balanced_accuracy, "val_confusion": results, 'val_precision': precision, 'val_recall': recall})

			try:
				plot_confusion_matrix(results, ["dead","alive","miscellaneous"], output_dir=self.model_folder+os.sep, title=title)
			except:
				pass
			print("Validation set: ",classification_report(ground_truth,predictions))


	def train_regressor(self):

		"""
		
		Train a regressor model to estimate time of interest for the subset of target cells that have an event during the movies.

		"""

		# Compile model
		# if pretrained model
		if self.pretrained is not None:
			# if recompile
			if self.recompile_pretrained:
				print('Recompiling the pretrained regressor model... Warning, this action reinitializes all the weights; are you sure that this is what you intended?')
				self.model_reg.set_weights(clone_model(self.model_reg).get_weights())
				self.model_reg.compile(optimizer=Adam(learning_rate=self.learning_rate), 
							  loss=self.loss_reg, 
							  metrics=['mse','mae'])
			else:
				pass
		else:
			print("Compiling the regressor...")
			self.model_reg.compile(optimizer=Adam(learning_rate=self.learning_rate), 
						  loss=self.loss_reg, 
						  metrics=['mse','mae'])
		
			
		self.gather_callbacks("regressor")

		# Train on subset of data with event 

		subset = self.x_train[np.argmax(self.y_class_train,axis=1)==0]
		# for i in range(30):
		# 	plt.plot(subset[i,:,0],c="tab:red")
		# 	plt.plot(subset[i,:,1],c="tab:blue")
		# 	plt.show()

		if hasattr(self, 'x_val'):
			self.history_regressor = self.model_reg.fit(x=self.x_train[np.argmax(self.y_class_train,axis=1)==0],
								y=self.y_time_train[np.argmax(self.y_class_train,axis=1)==0],
								batch_size=self.batch_size,
								epochs=self.epochs*2, 
								validation_data=(self.x_val[np.argmax(self.y_class_val,axis=1)==0],self.y_time_val[np.argmax(self.y_class_val,axis=1)==0]),
								callbacks=self.cb,
								verbose=1)
		else:
			self.history_regressor = self.model_reg.fit(x=self.x_train[np.argmax(self.y_class_train,axis=1)==0],
								y=self.y_time_train[np.argmax(self.y_class_train,axis=1)==0],
								batch_size=self.batch_size,
								epochs=self.epochs*2, 
								callbacks=self.cb,
								validation_split = self.validation_split,
								verbose=1)			

		self.plot_model_history(mode="regressor")
		self.dico.update({"history_regressor": self.history_regressor, "execution_time_regressor": self.cb[-1].times})
		

		# Evaluate best model 
		self.model_reg = load_model(os.sep.join([self.model_folder,"regressor.h5"]))
		self.model_reg.load_weights(os.sep.join([self.model_folder,"regressor.h5"]))
		self.evaluate_regression_model()
		
		np.save(os.sep.join([self.model_folder,"scores.npy"]), self.dico)
		


	def plot_model_history(self, mode="regressor"):

		"""
		
		Plot the learning curves for the specified mode (regressor or classifier).

		Args:
		mode (str, optional): The mode to plot the learning curve for. Default is "regressor".

		Note:
		- If mode is set to "regressor", it will plot the loss curves for regressor training history.
		- If mode is set to "classifier", it will plot the precision curves for classifier training history.
		- Saves the plot as an image file in the model_folder.

		"""

		if mode=="regressor":
			try:
				plt.plot(self.history_regressor.history['loss'])
				plt.plot(self.history_regressor.history['val_loss'])
				plt.title('model loss')
				plt.ylabel('loss')
				plt.xlabel('epoch')
				plt.yscale('log')
				plt.legend(['train', 'val'], loc='upper left')
				plt.pause(3)
				plt.savefig(os.sep.join([self.model_folder,"regression_loss.png"]),bbox_inches="tight",dpi=300)
				plt.close()
			except Exception as e:
				print(f"Error {e}; could not generate plot...")
		elif mode=="classifier":
			try:
				plt.plot(self.history_classifier.history['precision'])
				plt.plot(self.history_classifier.history['val_precision'])
				plt.title('model precision')
				plt.ylabel('precision')
				plt.xlabel('epoch')
				plt.legend(['train', 'val'], loc='upper left')
				plt.pause(3)
				plt.savefig(os.sep.join([self.model_folder,"classification_loss.png"]),bbox_inches="tight",dpi=300)
				plt.close()
			except Exception as e:
				print(f"Error {e}; could not generate plot...")
		else:
			return None			

	def evaluate_regression_model(self):

		"""

		Evaluate the regression model on the test and validation sets and save the regression plots.

		Note:
		- Computes the Mean Squared Error (MSE) for the predictions of the model on the test (if it exists) and validation sets.
		- Calls regression_plot to generate and save the regression plots for both sets.
		- Saves the plots as image files in the model_folder.

		"""

		mse = MeanSquaredError()
		mae = MeanAbsoluteError()

		if hasattr(self, 'x_test'):

			print("Evaluate on test set...")
			predictions = self.model_reg.predict(self.x_test[np.argmax(self.y_class_test,axis=1)==0], batch_size=self.batch_size)[:,0]
			ground_truth = self.y_time_test[np.argmax(self.y_class_test,axis=1)==0]
			assert predictions.shape==ground_truth.shape,"Shape mismatch between predictions and ground truths..."
			
			test_mse = mse(ground_truth, predictions).numpy()
			test_mae = mae(ground_truth, predictions).numpy()
			print(f"MSE on test set: {test_mse}...")
			print(f"MAE on test set: {test_mae}...")
			regression_plot(predictions, ground_truth, savepath=os.sep.join([self.model_folder,"test_regression.png"]))
			self.dico.update({"test_mse": test_mse, "test_mae": test_mae})

		if hasattr(self, 'x_val'):
			# Validation set
			predictions = self.model_reg.predict(self.x_val[np.argmax(self.y_class_val,axis=1)==0], batch_size=self.batch_size)[:,0]
			ground_truth = self.y_time_val[np.argmax(self.y_class_val,axis=1)==0]
			assert predictions.shape==ground_truth.shape,"Shape mismatch between predictions and ground truths..."
			
			val_mse = mse(ground_truth, predictions).numpy()
			val_mae = mae(ground_truth, predictions).numpy()

			regression_plot(predictions, ground_truth, savepath=os.sep.join([self.model_folder,"validation_regression.png"]))
			print(f"MSE on validation set: {val_mse}...")
			print(f"MAE on validation set: {val_mae}...")

			self.dico.update({"val_mse": val_mse, "val_mae": val_mae})


	def gather_callbacks(self, mode):

		"""

		Gather a list of callbacks based on the training mode.
		
		Parameters:
			- mode: str
				Training mode, either "classifier" or "regressor".
		
		"""
		
		self.cb = []
		
		if mode=="classifier":
			
			reduce_lr = ReduceLROnPlateau(monitor='val_precision', factor=0.1, patience=50,
										  cooldown=10, min_lr=5e-10, min_delta=1.0E-10,
										  verbose=1,mode="max")
			self.cb.append(reduce_lr)
			csv_logger = CSVLogger(os.sep.join([self.model_folder,'log_classifier.csv']), append=True, separator=';')
			self.cb.append(csv_logger)
			checkpoint_path = os.sep.join([self.model_folder,"classifier.h5"])
			cp_callback = ModelCheckpoint(checkpoint_path,monitor="val_precision",mode="max",verbose=1,save_best_only=True,save_weights_only=False,save_freq="epoch")
			self.cb.append(cp_callback)
			
			callback_stop = EarlyStopping(monitor='val_precision', patience=200)
			self.cb.append(callback_stop)
			
		elif mode=="regressor":
			
			reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50,
										  cooldown=10, min_lr=5e-10, min_delta=1.0E-10,
										  verbose=1,mode="min")
			self.cb.append(reduce_lr)

			csv_logger = CSVLogger(os.sep.join([self.model_folder,'log_regressor.csv']), append=True, separator=';')
			self.cb.append(csv_logger)
			
			checkpoint_path = os.sep.join([self.model_folder,"regressor.h5"])
			cp_callback = ModelCheckpoint(checkpoint_path,monitor="val_loss",mode="min",verbose=1,save_best_only=True,save_weights_only=False,save_freq="epoch")
			self.cb.append(cp_callback)
			
			callback_stop = EarlyStopping(monitor='val_loss', patience=200)
			self.cb.append(callback_stop)            
		
		log_dir = self.model_folder+os.sep
		cb_tb = TensorBoard(log_dir=log_dir, update_freq='batch')
		self.cb.append(cb_tb)

		cb_time = TimeHistory()
		self.cb.append(cb_time)
		
		
	
	def generate_sets(self):
		
		"""
		
		Load, preprocess and split dataset
		
		"""
		
		self.x_set = []
		self.y_time_set = []
		self.y_class_set = []
		
		for s in self.list_of_sets:
			self.load_and_normalize(s)

		self.x_set = np.array(self.x_set).astype(np.float32)
		self.x_set = self.interpolate_signals(self.x_set)

		self.y_time_set = np.array(self.y_time_set).astype(np.float32)
		self.y_class_set = np.array(self.y_class_set).astype(np.float32)

		class_test = np.isin(self.y_class_set, [0,1,2])
		self.x_set = self.x_set[class_test]
		self.y_time_set = self.y_time_set[class_test]
		self.y_class_set = self.y_class_set[class_test]
		
		# Compute class weights and one-hot encode
		self.class_weights = compute_weights(self.y_class_set)
		self.nbr_classes = len(np.unique(self.y_class_set))
		self.y_class_set = to_categorical(self.y_class_set)

		ds = train_test_split(self.x_set, 
							  self.y_time_set, 
							  self.y_class_set, 
							  validation_size=self.validation_split, 
							  test_size=self.test_split)
		
		self.x_train = ds["x_train"]
		self.x_val = ds["x_val"]
		self.y_time_train = ds["y1_train"].astype(np.float32)
		print(np.amax(self.y_time_train),np.amin(self.y_time_train))
		self.y_time_val = ds["y1_val"].astype(np.float32)
		self.y_class_train = ds["y2_train"]
		self.y_class_val = ds["y2_val"]
		
		if self.test_split>0:
			self.x_test = ds["x_test"]
			self.y_time_test = ds["y1_test"].astype(np.float32)
			self.y_class_test = ds["y2_test"]
		
		if self.augment:
			self.augment_training_set()        
	
	def augment_training_set(self, time_shift=True):
		
		"""
		
		Augment training set
		
		"""
		
		nbr_augment = self.augmentation_factor*len(self.x_train)
		randomize = np.arange(len(self.x_train))
		indices = random.choices(randomize,k=nbr_augment)

		x_train_aug = []
		y_time_train_aug = []
		y_class_train_aug = []

		for k in indices:
			aug = augmenter(self.x_train[k], 
							self.y_time_train[k], 
							self.y_class_train[k], 
							self.model_signal_length,
							time_shift=time_shift)
			x_train_aug.append(aug[0])
			y_time_train_aug.append(aug[1])
			y_class_train_aug.append(aug[2])

		# Save augmented training set
		self.x_train = np.array(x_train_aug)
		self.y_time_train = np.array(y_time_train_aug)
		self.y_class_train = np.array(y_class_train_aug)
		


	def load_and_normalize(self, subset):
		
		"""
		Load npy annotation set, reshape into a numpy array (MxTx2)
		Remove dubious annotations
		Normalize fluo and time of interest
		To do: set 1, 2, 3 channels
		
		"""
		
		set_k = np.load(subset,allow_pickle=True)
		### here do a mapping between channel option and existing signals

		required_signals = self.channel_option
		available_signals = list(set_k[0].keys())

		selected_signals = []
		for s in required_signals:
			pattern_test = [s in a for a in available_signals]
			if np.any(pattern_test):
				valid_columns = np.array(available_signals)[np.array(pattern_test)]
				if len(valid_columns)==1:
					selected_signals.append(valid_columns[0])
				else:
					print(f'Found several candidate signals: {valid_columns}')
					for vc in natsorted(valid_columns):
						if 'circle' in vc:
							selected_signals.append(vc)
							break
					else:
						selected_signals.append(valid_columns[0])
			else:
				return None	
		

		key_to_check = selected_signals[0] #self.channel_option[0]
		signal_lengths = [len(l[key_to_check]) for l in set_k]
		max_length = np.amax(signal_lengths)

		fluo = np.zeros((len(set_k),max_length,self.n_channels))
		classes = np.zeros(len(set_k))
		times_of_interest = np.zeros(len(set_k))
		
		for k in range(len(set_k)):
			
			for i in range(self.n_channels):
				try:
					# take into account timeline for accurate time regression
					timeline = set_k[k]['FRAME'].astype(int)
					fluo[k,timeline,i] = set_k[k][selected_signals[i]]
				except:
					print(f"Attribute {selected_signals[i]} matched to {self.channel_option[i]} not found in annotation...")
					pass

			classes[k] = set_k[k]["class"]
			times_of_interest[k] = set_k[k]["time_of_interest"]

		# Correct absurd times of interest
		times_of_interest[np.nonzero(classes)] = -1
		times_of_interest[(times_of_interest<=0.0)] = -1

		# Attempt per-set normalization
		fluo = pad_to_model_length(fluo, self.model_signal_length)
		if self.normalize:
			fluo = normalize_signal_set(fluo, self.channel_option, normalization_percentile=self.normalization_percentile, 
										normalization_values=self.normalization_values, normalization_clip=self.normalization_clip,
										)
			
		# Trivial normalization for time of interest
		times_of_interest /= self.model_signal_length
		
		# Add to global dataset
		self.x_set.extend(fluo)
		self.y_time_set.extend(times_of_interest)
		self.y_class_set.extend(classes)

def _interpret_normalization_parameters(n_channels, normalization_percentile, normalization_values, normalization_clip):
	
	if normalization_percentile is None:
		normalization_percentile = [True]*n_channels
	if normalization_values is None:
		normalization_values = [[0.1,99.9]]*n_channels
	if normalization_clip is None:
		normalization_clip = [False]*n_channels
	
	if isinstance(normalization_percentile, bool):
		normalization_percentile = [normalization_percentile]*n_channels
	if isinstance(normalization_clip, bool):
		normalization_clip = [normalization_clip]*n_channels
	if len(normalization_values)==2 and not isinstance(normalization_values[0], list):
		normalization_values = [normalization_values]*n_channels

	assert len(normalization_values)==n_channels
	assert len(normalization_clip)==n_channels
	assert len(normalization_percentile)==n_channels

	return normalization_percentile, normalization_values, normalization_clip


def normalize_signal_set(signal_set, channel_option, percentile_alive=[0.01,99.99], percentile_dead=[0.5,99.999], percentile_generic=[0.01,99.99], normalization_percentile=None, normalization_values=None, normalization_clip=None):

	"""

	Normalize the signals from a set of single-cell signals.

	"""

	# Check normalization params are ok
	n_channels = len(channel_option)
	normalization_percentile, normalization_values, normalization_clip = _interpret_normalization_parameters(n_channels,
																											normalization_percentile,
																											normalization_values,
																											normalization_clip)

	for k,channel in enumerate(channel_option):

		zero_values = []
		for i in range(len(signal_set)):
			zeros_loc = np.where(signal_set[i,:,k]==0)
			zero_values.append(zeros_loc)

		values = signal_set[:,:,k]

		if normalization_percentile[k]:
			min_val = np.nanpercentile(values[values!=0.], normalization_values[k][0])
			max_val = np.nanpercentile(values[values!=0.], normalization_values[k][1])
		else:
			min_val = normalization_values[k][0]
			max_val = normalization_values[k][1]

		signal_set[:,:,k] -= min_val
		signal_set[:,:,k] /= (max_val - min_val)

		if normalization_clip[k]:
			to_clip_low = []
			to_clip_high = []
			for i in range(len(signal_set)):
				clip_low_loc = np.where(signal_set[i,:,k]<=0)
				clip_high_loc = np.where(signal_set[i,:,k]>=1.0)
				to_clip_low.append(clip_low_loc)
				to_clip_high.append(clip_high_loc)

			for i,z in enumerate(to_clip_low):
				signal_set[i,z,k] = 0.
			for i,z in enumerate(to_clip_high):
				signal_set[i,z,k] = 1.					

		for i,z in enumerate(zero_values):
			signal_set[i,z,k] = 0.

	# for k,channel in enumerate(channel_option):
	#
	# 	zero_values = []
	# 	for i in range(len(signal_set)):
	# 		zeros_loc = np.where(signal_set[i,:,k]==0)
	# 		zero_values.append(zeros_loc)
	#
	# 	if ("dead_nuclei_channel" in channel and 'haralick' not in channel) or ("RED" in channel):
	# 		print('red normalization')
	#
	# 		min_percentile_dead, max_percentile_dead = percentile_dead
	# 		min_set = signal_set[:,:5,k]
	# 		max_set = signal_set[:,:,k]
	# 		min_fluo_dead = np.nanpercentile(min_set[min_set!=0.], min_percentile_dead) # 5 % on initial frame where barely any dead are expected
	# 		max_fluo_dead = np.nanpercentile(max_set[max_set!=0.], max_percentile_dead) # 99th percentile on last fluo frame
	# 		signal_set[:,:,k] -= min_fluo_dead
	# 		signal_set[:,:,k] /= (max_fluo_dead - min_fluo_dead)
	#
	# 	elif ("live_nuclei_channel" in channel and 'haralick' not in channel) or ("BLUE" in channel):
	#
	# 		print('blue normalization')
	# 		min_percentile_alive, max_percentile_alive = percentile_alive
	# 		values = signal_set[:,:5,k]
	# 		min_fluo_alive = np.nanpercentile(values[values!=0.], min_percentile_alive) # safe 0.5% of Hoescht on initial frame
	# 		max_fluo_alive = np.nanpercentile(values[values!=0.], max_percentile_alive)
	# 		signal_set[:,:,k] -= min_fluo_alive
	# 		signal_set[:,:,k] /= (max_fluo_alive - min_fluo_alive)
	#
	# 	elif 0.8<np.mean(signal_set[:,:,k])<1.2:
	# 		print('detected normalized signal; assume min max in 0.5-1.5 range')
	# 		min_fluo_alive = 0.5
	# 		max_fluo_alive = 1.5
	# 		signal_set[:,:,k] -= min_fluo_alive
	# 		signal_set[:,:,k] /= (max_fluo_alive - min_fluo_alive)
	#
	# 	else:
	#
	# 		min_percentile, max_percentile = percentile_generic
	# 		values = signal_set[:,:,k]
	# 		min_signal = np.nanpercentile(values[values!=0.], min_percentile)
	# 		max_signal= np.nanpercentile(values[values!=0.], max_percentile)
	# 		signal_set[:,:,k] -= min_signal
	# 		signal_set[:,:,k] /= (max_signal - min_signal)
	#
	# 	for i,z in enumerate(zero_values):
	# 		signal_set[i,z,k] = 0.

	return signal_set

def pad_to_model_length(signal_set, model_signal_length):  

	"""

	Pad the signal set to match the specified model signal length.

	Parameters
	----------
	signal_set : array-like
		The signal set to be padded.
	model_signal_length : int
		The desired length of the model signal.

	Returns
	-------
	array-like
		The padded signal set.

	Notes
	-----
	This function pads the signal set with zeros along the second dimension (axis 1) to match the specified model signal
	length. The padding is applied to the end of the signals, increasing their length.

	Examples
	--------
	>>> signal_set = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	>>> padded_signals = pad_to_model_length(signal_set, 5)
	
	"""

	padded = np.pad(signal_set, [(0,0),(0,model_signal_length - signal_set.shape[1]),(0,0)]) 
	
	return padded

def random_intensity_change(signal):

	"""

	Randomly change the intensity of a signal.

	Parameters
	----------
	signal : array-like
		The input signal to be modified.

	Returns
	-------
	array-like
		The modified signal with randomly changed intensity.

	Notes
	-----
	This function applies a random intensity change to each channel of the input signal. The intensity change is
	performed by multiplying each channel with a random value drawn from a uniform distribution between 0.7 and 1.0.

	Examples
	--------
	>>> signal = np.array([[1, 2, 3], [4, 5, 6]])
	>>> modified_signal = random_intensity_change(signal)
	
	"""

	for k in range(signal.shape[1]):
		signal[:,k] = signal[:,k]*np.random.uniform(0.7,1.)

	return signal

def gauss_noise(signal):

	"""
	
	Add Gaussian noise to a signal.

	Parameters
	----------
	signal : array-like
		The input signal to which noise will be added.

	Returns
	-------
	array-like
		The signal with Gaussian noise added.

	Notes
	-----
	This function adds Gaussian noise to the input signal. The noise is generated by drawing random values from a
	standard normal distribution and scaling them by a factor of 0.08 times the input signal. The scaled noise values
	are then added to the original signal.

	Examples
	--------
	>>> signal = np.array([1, 2, 3, 4, 5])
	>>> noisy_signal = gauss_noise(signal)

	"""

	sig = 0.08*np.random.uniform(0,1)
	signal = signal + sig*np.random.normal(0,1,signal.shape)*signal
	return signal

def random_time_shift(signal, time_of_interest, model_signal_length):

	"""

	Randomly shift the signals to another time.

	Parameters
	----------
	signal : array-like
		The signal to be shifted.
	time_of_interest : int or float
		The original time of interest for the signal. Use -1 if not applicable.
	model_signal_length : int
		The length of the model signal.

	Returns
	-------
	array-like
		The shifted fluorescence signal.
	int or float
		The new time of interest if available; otherwise, the original time of interest.

	Notes
	-----
	This function randomly selects a target time within the specified model signal length and shifts the
	signal accordingly. The shift is performed along the first dimension (axis 0) of the signal. The function uses
	nearest-neighbor interpolation for shifting.

	If the original time of interest (`time_of_interest`) is provided (not equal to -1), the function returns the
	shifted signal along with the new time of interest. Otherwise, it returns the shifted signal along with the
	original time of interest.

	The `max_time` is set to the `model_signal_length` unless the original time of interest is provided. In that case,
	`max_time` is set to `model_signal_length - 3` to prevent shifting too close to the edge.

	Examples
	--------
	>>> signal = np.array([[1, 2, 3], [4, 5, 6]])
	>>> shifted_signal, new_time = random_time_shift(signal, 1, 5)
	
	"""
	
	max_time = model_signal_length
	return_target = False
	if time_of_interest != -1:
		return_target = True
		max_time = model_signal_length - 3 # to prevent approaching too much to the edge
	
	times = np.linspace(0,max_time,1000)
	target_time = np.random.choice(times)
	
	delta_t = target_time - time_of_interest
	signal = shift(signal, [delta_t,0], order=0, mode="nearest")

	if return_target:
		return signal,target_time
	else:
		return signal, time_of_interest

def augmenter(signal, time_of_interest, cclass, model_signal_length, time_shift=True, probability=0.8):

	"""

	Augment randomely each single cell signals to explore new noise,
	 intensity ratios and times

	"""
	if np.amax(time_of_interest)<=1.0:
		time_of_interest *= model_signal_length

	# augment with a certain probability
	r = random.random()
	if r<= probability:

		if time_shift:
			# do not time shift miscellaneous cells
			if cclass.argmax()!=2.:
				assert time_of_interest is not None, f"Please provide valid lysis times"
				signal,time_of_interest = random_time_shift(signal, time_of_interest, model_signal_length)

		signal = random_intensity_change(signal)
		signal = gauss_noise(signal)

	return signal, time_of_interest/model_signal_length, cclass


def residual_block1D(x, number_of_filters, kernel_size=8, match_filter_size=True, connection='identity'):

	"""

	Create a 1D residual block.

	Parameters
	----------
	x : Tensor
		Input tensor.
	number_of_filters : int
		Number of filters in the convolutional layers.
	match_filter_size : bool, optional
		Whether to match the filter size of the skip connection to the output. Default is True.

	Returns
	-------
	Tensor
		Output tensor of the residual block.

	Notes
	-----
	This function creates a 1D residual block by performing the original mapping followed by adding a skip connection
	and applying non-linear activation. The skip connection allows the gradient to flow directly to earlier layers and
	helps mitigate the vanishing gradient problem. The residual block consists of three convolutional layers with
	batch normalization and ReLU activation functions.

	If `match_filter_size` is True, the skip connection is adjusted to have the same number of filters as the output.
	Otherwise, the skip connection is kept as is.

	Examples
	--------
	>>> inputs = Input(shape=(10, 3))
	>>> x = residual_block1D(inputs, 64)
	# Create a 1D residual block with 64 filters and apply it to the input tensor.
	
	"""


	# Create skip connection
	x_skip = x

	# Perform the original mapping
	if connection=='identity':
		x = Conv1D(number_of_filters, kernel_size=kernel_size, strides=1,padding="same")(x_skip)
	elif connection=='projection':
		x = ZeroPadding1D(padding=kernel_size//2)(x_skip)
		x = Conv1D(number_of_filters, kernel_size=kernel_size, strides=2,padding="valid")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)

	x = Conv1D(number_of_filters, kernel_size=kernel_size, strides=1,padding="same")(x)
	x = BatchNormalization()(x)

	if match_filter_size and connection=='identity':
		x_skip = Conv1D(number_of_filters, kernel_size=1, padding="same")(x_skip)
	elif match_filter_size and connection=='projection':
		x_skip = Conv1D(number_of_filters, kernel_size=1, strides=2, padding="valid")(x_skip)


	# Add the skip connection to the regular mapping
	x = Add()([x, x_skip])

	# Nonlinearly activate the result
	x = Activation("relu")(x)

	# Return the result
	return x


def MultiscaleResNetModel(n_channels, n_classes = 3, dropout_rate=0, dense_collection=0, use_pooling=True,
				 header="classifier", model_signal_length = 128):

	"""

	Define a generic ResNet 1D encoder model.

	Parameters
	----------
	n_channels : int
		Number of input channels.
	n_blocks : int
		Number of residual blocks in the model.
	n_classes : int, optional
		Number of output classes. Default is 3.
	dropout_rate : float, optional
		Dropout rate to be applied. Default is 0.
	dense_collection : int, optional
		Number of neurons in the dense layer. Default is 0.
	header : str, optional
		Type of the model header. "classifier" for classification, "regressor" for regression. Default is "classifier".
	model_signal_length : int, optional
		Length of the input signal. Default is 128.

	Returns
	-------
	keras.models.Model
		ResNet 1D encoder model.

	Notes
	-----
	This function defines a generic ResNet 1D encoder model with the specified number of input channels, residual
	blocks, output classes, dropout rate, dense collection, and model header. The model architecture follows the
	ResNet principles with 1D convolutional layers and residual connections. The final activation and number of
	neurons in the output layer are determined based on the header type.

	Examples
	--------
	>>> model = ResNetModel(n_channels=3, n_blocks=4, n_classes=2, dropout_rate=0.2)
	# Define a ResNet 1D encoder model with 3 input channels, 4 residual blocks, and 2 output classes.
	
	"""

	if header=="classifier":
		final_activation = "softmax"
		neurons_final = n_classes
	elif header=="regressor":
		final_activation = "linear"
		neurons_final = 1
	else:
		return None

	inputs = Input(shape=(model_signal_length,n_channels,))
	x = ZeroPadding1D(3)(inputs)
	x = Conv1D(64, kernel_size=7, strides=2, padding="valid", use_bias=False)(x)
	x = BatchNormalization()(x)
	x = ZeroPadding1D(1)(x)
	x_common = MaxPooling1D(pool_size=3, strides=2, padding='valid')(x)

	# Block 1
	x1 = residual_block1D(x_common, 64, kernel_size=7,connection='projection')
	x1 = residual_block1D(x1, 128, kernel_size=7,connection='projection')
	x1 = residual_block1D(x1, 256, kernel_size=7,connection='projection')
	x1 = GlobalAveragePooling1D()(x1)

	# Block 2
	x2 = residual_block1D(x_common, 64, kernel_size=5,connection='projection')
	x2 = residual_block1D(x2, 128, kernel_size=5,connection='projection')
	x2 = residual_block1D(x2, 256, kernel_size=5,connection='projection')
	x2 = GlobalAveragePooling1D()(x2)

	# Block 3
	x3 = residual_block1D(x_common, 64, kernel_size=3,connection='projection')
	x3 = residual_block1D(x3, 128, kernel_size=3,connection='projection')
	x3 = residual_block1D(x3, 256, kernel_size=3,connection='projection')
	x3 = GlobalAveragePooling1D()(x3)

	x_combined = Concatenate()([x1, x2, x3])
	x_combined = Flatten()(x_combined)

	if dense_collection>0:
		x_combined = Dense(dense_collection)(x_combined)
	if dropout_rate>0:
		x_combined = Dropout(dropout_rate)(x_combined)

	x_combined = Dense(neurons_final,activation=final_activation,name=header)(x_combined)
	model = Model(inputs, x_combined, name=header) 

	return model

def ResNetModelCurrent(n_channels, n_slices, depth=2, use_pooling=True, n_classes = 3, dropout_rate=0.1, dense_collection=512,
				 	   header="classifier", model_signal_length = 128):
	
	if header=="classifier":
		final_activation = "softmax"
		neurons_final = n_classes
	elif header=="regressor":
		final_activation = "linear"
		neurons_final = 1
	else:
		return None

	inputs = Input(shape=(model_signal_length,n_channels,))
	x2 = Conv1D(64, kernel_size=1,strides=1,padding='same')(inputs)

	n_filters = 64
	for k in range(depth):
		for i in range(n_slices):
				x2 = residual_block1D(x2,n_filters,kernel_size=8)
		n_filters *= 2
		if use_pooling and k!=(depth-1):
			x2 = MaxPooling1D()(x2)

	x2 = GlobalAveragePooling1D()(x2)
	if dense_collection>0:
		x2 = Dense(dense_collection)(x2)
	if dropout_rate>0:
		x2 = Dropout(dropout_rate)(x2)

	x2 = Dense(neurons_final,activation=final_activation,name=header)(x2)
	model = Model(inputs, x2, name=header) 

	return model


def train_signal_model(config):

	config = config.replace('\\','/')
	config = rf"{config}"
	assert os.path.exists(config),f'Config {config} is not a valid path.'

	script_path = os.sep.join([abs_path, 'scripts', 'train_signal_model.py'])
	cmd = f'python "{script_path}" --config "{config}"'
	subprocess.call(cmd, shell=True)

def derivative(x, timeline, window, mode='bi'):
	
	"""
	Compute the derivative of a given array of values with respect to time using a specified numerical differentiation method.

	Parameters
	----------
	x : array_like
		The input array of values.
	timeline : array_like
		The array representing the time points corresponding to the input values.
	window : int
		The size of the window used for numerical differentiation. Must be a positive odd integer.
	mode : {'bi', 'forward', 'backward'}, optional
		The numerical differentiation method to be used:
		- 'bi' (default): Bidirectional differentiation using a symmetric window.
		- 'forward': Forward differentiation using a one-sided window.
		- 'backward': Backward differentiation using a one-sided window.

	Returns
	-------
	dxdt : ndarray
		The computed derivative values of the input array with respect to time.

	Raises
	------
	AssertionError
		If the window size is not an odd integer and mode is 'bi'.

	Notes
	-----
	- For 'bi' mode, the window size must be an odd number.
	- For 'forward' mode, the derivative at the edge points may not be accurate due to the one-sided window.
	- For 'backward' mode, the derivative at the first few points may not be accurate due to the one-sided window.

	Examples
	--------
	>>> import numpy as np
	>>> x = np.array([1, 2, 4, 7, 11])
	>>> timeline = np.array([0, 1, 2, 3, 4])
	>>> window = 3
	>>> derivative(x, timeline, window, mode='bi')
	array([3., 3., 3.])

	>>> derivative(x, timeline, window, mode='forward')
	array([1., 2., 3.])

	>>> derivative(x, timeline, window, mode='backward')
	array([3., 3., 3., 3.])
	"""

	# modes = bi, forward, backward
	dxdt = np.zeros(len(x))
	dxdt[:] = np.nan
	
	if mode=='bi':
		assert window%2==1,'Please set an odd window for the bidirectional mode'
		lower_bound = window//2
		upper_bound = len(x) - window//2 - 1
	elif mode=='forward':
		lower_bound = 0
		upper_bound = len(x) - window
	elif mode=='backward':
		lower_bound = window
		upper_bound = len(x)

	for t in range(lower_bound,upper_bound):
		if mode=='bi':
			dxdt[t] = (x[t+window//2+1] - x[t-window//2]) / (timeline[t+window//2+1] - timeline[t-window//2])
		elif mode=='forward':
			dxdt[t] = (x[t+window] - x[t]) /  (timeline[t+window] - timeline[t])
		elif mode=='backward':
			dxdt[t] = (x[t] - x[t-window]) /  (timeline[t] - timeline[t-window])
	return dxdt

def velocity(x,y,timeline,window,mode='bi'):

	"""
	Compute the velocity vector of a given 2D trajectory represented by arrays of x and y coordinates
	with respect to time using a specified numerical differentiation method.

	Parameters
	----------
	x : array_like
		The array of x-coordinates of the trajectory.
	y : array_like
		The array of y-coordinates of the trajectory.
	timeline : array_like
		The array representing the time points corresponding to the x and y coordinates.
	window : int
		The size of the window used for numerical differentiation. Must be a positive odd integer.
	mode : {'bi', 'forward', 'backward'}, optional
		The numerical differentiation method to be used:
		- 'bi' (default): Bidirectional differentiation using a symmetric window.
		- 'forward': Forward differentiation using a one-sided window.
		- 'backward': Backward differentiation using a one-sided window.

	Returns
	-------
	v : ndarray
		The computed velocity vector of the 2D trajectory with respect to time.
		The first column represents the x-component of velocity, and the second column represents the y-component.

	Raises
	------
	AssertionError
		If the window size is not an odd integer and mode is 'bi'.

	Notes
	-----
	- For 'bi' mode, the window size must be an odd number.
	- For 'forward' mode, the velocity at the edge points may not be accurate due to the one-sided window.
	- For 'backward' mode, the velocity at the first few points may not be accurate due to the one-sided window.

	Examples
	--------
	>>> import numpy as np
	>>> x = np.array([1, 2, 4, 7, 11])
	>>> y = np.array([0, 3, 5, 8, 10])
	>>> timeline = np.array([0, 1, 2, 3, 4])
	>>> window = 3
	>>> velocity(x, y, timeline, window, mode='bi')
	array([[3., 3.],
		   [3., 3.]])

	>>> velocity(x, y, timeline, window, mode='forward')
	array([[2., 2.],
		   [3., 3.]])

	>>> velocity(x, y, timeline, window, mode='backward')
	array([[3., 3.],
		   [3., 3.]])
	"""

	v = np.zeros((len(x),2))
	v[:,:] = np.nan
	
	v[:,0] = derivative(x, timeline, window, mode=mode)
	v[:,1] = derivative(y, timeline, window, mode=mode)

	return v

def magnitude_velocity(v_matrix):

	"""
	Compute the magnitude of velocity vectors given a matrix representing 2D velocity vectors.

	Parameters
	----------
	v_matrix : array_like
		The matrix where each row represents a 2D velocity vector with the first column
		being the x-component and the second column being the y-component.

	Returns
	-------
	magnitude : ndarray
		The computed magnitudes of the input velocity vectors.

	Notes
	-----
	- If a velocity vector has NaN components, the corresponding magnitude will be NaN.
	- The function handles NaN values in the input matrix gracefully.

	Examples
	--------
	>>> import numpy as np
	>>> v_matrix = np.array([[3, 4],
	...                      [2, 2],
	...                      [3, 3]])
	>>> magnitude_velocity(v_matrix)
	array([5., 2.82842712, 4.24264069])

	>>> v_matrix_with_nan = np.array([[3, 4],
	...                               [np.nan, 2],
	...                               [3, np.nan]])
	>>> magnitude_velocity(v_matrix_with_nan)
	array([5., nan, nan])
	"""

	magnitude = np.zeros(len(v_matrix))
	magnitude[:] = np.nan
	for i in range(len(v_matrix)):
		if v_matrix[i,0]==v_matrix[i,0]:
			magnitude[i] = np.sqrt(v_matrix[i,0]**2 + v_matrix[i,1]**2)
	return magnitude
		
def orientation(v_matrix):

	"""
	Compute the orientation angles (in radians) of 2D velocity vectors given a matrix representing velocity vectors.

	Parameters
	----------
	v_matrix : array_like
		The matrix where each row represents a 2D velocity vector with the first column
		being the x-component and the second column being the y-component.

	Returns
	-------
	orientation_array : ndarray
		The computed orientation angles of the input velocity vectors in radians.
		If a velocity vector has NaN components, the corresponding orientation angle will be NaN.

	Examples
	--------
	>>> import numpy as np
	>>> v_matrix = np.array([[3, 4],
	...                      [2, 2],
	...                      [-3, -3]])
	>>> orientation(v_matrix)
	array([0.92729522, 0.78539816, -2.35619449])

	>>> v_matrix_with_nan = np.array([[3, 4],
	...                               [np.nan, 2],
	...                               [3, np.nan]])
	>>> orientation(v_matrix_with_nan)
	array([0.92729522, nan, nan])
	"""

	orientation_array = np.zeros(len(v_matrix))
	for t in range(len(orientation_array)):
		if v_matrix[t,0]==v_matrix[t,0]:
			orientation_array[t] = np.arctan2(v_matrix[t,0],v_matrix[t,1])
	return orientation_array

def T_MSD(x,y,dt):

	"""
	Compute the Time-Averaged Mean Square Displacement (T-MSD) of a 2D trajectory.

	Parameters
	----------
	x : array_like
		The array of x-coordinates of the trajectory.
	y : array_like
		The array of y-coordinates of the trajectory.
	dt : float
		The time interval between successive data points in the trajectory.

	Returns
	-------
	msd : list
		A list containing the Time-Averaged Mean Square Displacement values for different time lags.
	timelag : ndarray
		The array representing the time lags corresponding to the calculated MSD values.

	Notes
	-----
	- T-MSD is a measure of the average spatial extent explored by a particle over a given time interval.
	- The input trajectories (x, y) are assumed to be in the same unit of length.
	- The time interval (dt) should be consistent with the time unit used in the data.

	Examples
	--------
	>>> import numpy as np
	>>> x = np.array([1, 2, 4, 7, 11])
	>>> y = np.array([0, 3, 5, 8, 10])
	>>> dt = 1.0  # Time interval between data points
	>>> T_MSD(x, y, dt)
	([6.0, 9.0, 4.666666666666667, 1.6666666666666667],
	 array([1., 2., 3., 4.]))
	"""

	msd = []
	N = len(x)
	for n in range(1,N):
		s = 0
		for i in range(0,N-n):
			s+=(x[n+i] - x[i])**2 + (y[n+i] - y[i])**2
		msd.append(1/(N-n)*s)

	timelag = np.linspace(dt,(N-1)*dt,N-1)
	return msd,timelag 

def linear_msd(t, m):

	"""
	Function to compute Mean Square Displacement (MSD) with a linear scaling relationship.

	Parameters
	----------
	t : array_like
		Time lag values.
	m : float
		Linear scaling factor representing the slope of the MSD curve.

	Returns
	-------
	msd : ndarray
		Computed MSD values based on the linear scaling relationship.

	Examples
	--------
	>>> import numpy as np
	>>> t = np.array([1, 2, 3, 4])
	>>> m = 2.0
	>>> linear_msd(t, m)
	array([2., 4., 6., 8.])
	"""

	return m*t

def alpha_msd(t, m, alpha):

	"""
	Function to compute Mean Square Displacement (MSD) with a power-law scaling relationship.

	Parameters
	----------
	t : array_like
		Time lag values.
	m : float
		Scaling factor.
	alpha : float
		Exponent representing the scaling relationship between MSD and time.

	Returns
	-------
	msd : ndarray
		Computed MSD values based on the power-law scaling relationship.

	Examples
	--------
	>>> import numpy as np
	>>> t = np.array([1, 2, 3, 4])
	>>> m = 2.0
	>>> alpha = 0.5
	>>> alpha_msd(t, m, alpha)
	array([2.        , 4.        , 6.        , 8.        ])
	"""

	return m*t**alpha

def sliding_msd(x, y, timeline, window, mode='bi', n_points_migration=7,  n_points_transport=7):

	"""
	Compute sliding mean square displacement (sMSD) and anomalous exponent (alpha) for a 2D trajectory using a sliding window approach.

	Parameters
	----------
	x : array_like
		The array of x-coordinates of the trajectory.
	y : array_like
		The array of y-coordinates of the trajectory.
	timeline : array_like
		The array representing the time points corresponding to the x and y coordinates.
	window : int
		The size of the sliding window used for computing local MSD and alpha values.
	mode : {'bi', 'forward', 'backward'}, optional
		The sliding window mode:
		- 'bi' (default): Bidirectional sliding window.
		- 'forward': Forward sliding window.
		- 'backward': Backward sliding window.
	n_points_migration : int, optional
		The number of points used for fitting the linear function in the MSD calculation.
	n_points_transport : int, optional
		The number of points used for fitting the alpha function in the anomalous exponent calculation.

	Returns
	-------
	s_msd : ndarray
		Sliding Mean Square Displacement values calculated using the sliding window approach.
	s_alpha : ndarray
		Sliding anomalous exponent (alpha) values calculated using the sliding window approach.

	Raises
	------
	AssertionError
		If the window size is not larger than the number of fit points.

	Notes
	-----
	- The input trajectories (x, y) are assumed to be in the same unit of length.
	- The time unit used in the data should be consistent with the time intervals in the timeline array.

	Examples
	--------
	>>> import numpy as np
	>>> x = np.array([1, 2, 4, 7, 11, 15, 20])
	>>> y = np.array([0, 3, 5, 8, 10, 14, 18])
	>>> timeline = np.array([0, 1, 2, 3, 4, 5, 6])
	>>> window = 3
	>>> s_msd, s_alpha = sliding_msd(x, y, timeline, window, n_points_migration=2, n_points_transport=3)
	"""

	assert window > n_points_migration,'Please set a window larger than the number of fit points...'
	
	# modes = bi, forward, backward
	s_msd = np.zeros(len(x))
	s_msd[:] = np.nan
	s_alpha = np.zeros(len(x))
	s_alpha[:] = np.nan
	dt = timeline[1] - timeline[0]
	
	if mode=='bi':
		assert window%2==1,'Please set an odd window for the bidirectional mode'
		lower_bound = window//2
		upper_bound = len(x) - window//2 - 1
	elif mode=='forward':
		lower_bound = 0
		upper_bound = len(x) - window
	elif mode=='backward':
		lower_bound = window
		upper_bound = len(x)
	
	for t in range(lower_bound,upper_bound):
		if mode=='bi':
			x_sub = x[t-window//2:t+window//2+1]
			y_sub = y[t-window//2:t+window//2+1]
			msd,timelag = T_MSD(x_sub,y_sub,dt)
			# dxdt[t] = (x[t+window//2+1] - x[t-window//2]) / (timeline[t+window//2+1] - timeline[t-window//2])
		elif mode=='forward':
			x_sub = x[t:t+window]
			y_sub = y[t:t+window]
			msd,timelag = T_MSD(x_sub,y_sub,dt)
			# dxdt[t] = (x[t+window] - x[t]) /  (timeline[t+window] - timeline[t])
		elif mode=='backward':
			x_sub = x[t-window:t]
			y_sub = y[t-window:t]
			msd,timelag = T_MSD(x_sub,y_sub,dt)
			# dxdt[t] = (x[t] - x[t-window]) /  (timeline[t] - timeline[t-window])
		popt,pcov = curve_fit(linear_msd,timelag[:n_points_migration],msd[:n_points_migration])
		s_msd[t] = popt[0]
		popt_alpha,pcov_alpha = curve_fit(alpha_msd,timelag[:n_points_transport],msd[:n_points_transport])
		s_alpha[t] = popt_alpha[1]
		
	return s_msd, s_alpha

def drift_msd(t, d, v):
    return 4*d*t + v**2*t**2

def sliding_msd_drift(x, y, timeline, window, mode='bi', n_points_migration=7,  n_points_transport=7, r2_threshold=0.75):

	"""
	"""

	assert window > n_points_migration,'Please set a window larger than the number of fit points...'
	
	# modes = bi, forward, backward
	s_diffusion = np.zeros(len(x))
	s_diffusion[:] = np.nan
	s_velocity = np.zeros(len(x))
	s_velocity[:] = np.nan
	dt = timeline[1] - timeline[0]
	
	if mode=='bi':
		assert window%2==1,'Please set an odd window for the bidirectional mode'
		lower_bound = window//2
		upper_bound = len(x) - window//2 - 1
	elif mode=='forward':
		lower_bound = 0
		upper_bound = len(x) - window
	elif mode=='backward':
		lower_bound = window
		upper_bound = len(x)
	
	for t in range(lower_bound,upper_bound):
		if mode=='bi':
			x_sub = x[t-window//2:t+window//2+1]
			y_sub = y[t-window//2:t+window//2+1]
			msd,timelag = T_MSD(x_sub,y_sub,dt)
			# dxdt[t] = (x[t+window//2+1] - x[t-window//2]) / (timeline[t+window//2+1] - timeline[t-window//2])
		elif mode=='forward':
			x_sub = x[t:t+window]
			y_sub = y[t:t+window]
			msd,timelag = T_MSD(x_sub,y_sub,dt)
			# dxdt[t] = (x[t+window] - x[t]) /  (timeline[t+window] - timeline[t])
		elif mode=='backward':
			x_sub = x[t-window:t]
			y_sub = y[t-window:t]
			msd,timelag = T_MSD(x_sub,y_sub,dt)
			# dxdt[t] = (x[t] - x[t-window]) /  (timeline[t] - timeline[t-window])

		popt,pcov = curve_fit(drift_msd,timelag[:n_points_migration],msd[:n_points_migration])
		#if not np.any([math.isinf(a) for a in pcov.flatten()]):
		s_diffusion[t] = popt[0]
		s_velocity[t] = popt[1]
		
	return s_diffusion, s_velocity

def columnwise_mean(matrix, min_nbr_values = 1):
	
	"""
	Calculate the column-wise mean and standard deviation of non-NaN elements in the input matrix.

	Parameters:
	----------
	matrix : numpy.ndarray
		The input matrix for which column-wise mean and standard deviation are calculated.
	min_nbr_values : int, optional
		The minimum number of non-NaN values required in a column to calculate mean and standard deviation.
		Default is 8.

	Returns:
	-------
	mean_line : numpy.ndarray
		An array containing the column-wise mean of non-NaN elements. Elements with fewer than `min_nbr_values` non-NaN
		values are replaced with NaN.
	mean_line_std : numpy.ndarray
		An array containing the column-wise standard deviation of non-NaN elements. Elements with fewer than `min_nbr_values`
		non-NaN values are replaced with NaN.

	Notes:
	------
	1. This function calculates the mean and standard deviation of non-NaN elements in each column of the input matrix.
	2. Columns with fewer than `min_nbr_values` non-zero elements will have NaN as the mean and standard deviation.
	3. NaN values in the input matrix are ignored during calculation.
	"""

	mean_line = np.zeros(matrix.shape[1])
	mean_line[:] = np.nan
	mean_line_std = np.zeros(matrix.shape[1])
	mean_line_std[:] = np.nan  
	
	for k in range(matrix.shape[1]):
		values = matrix[:,k]
		values = values[values!=0]
		if len(values[values==values])>min_nbr_values:
			mean_line[k] = np.nanmean(values)
			mean_line_std[k] = np.nanstd(values)
	return mean_line, mean_line_std


def mean_signal(df, signal_name, class_col, time_col=None, class_value=[0], return_matrix=False, forced_max_duration=None, min_nbr_values=2):

	"""
	Calculate the mean and standard deviation of a specified signal for tracks of a given class in the input DataFrame.

	Parameters:
	----------
	df : pandas.DataFrame
		Input DataFrame containing tracking data.
	signal_name : str
		Name of the signal (column) in the DataFrame for which mean and standard deviation are calculated.
	class_col : str
		Name of the column in the DataFrame containing class labels.
	time_col : str, optional
		Name of the column in the DataFrame containing time information. Default is None.
	class_value : int, optional
		Value representing the class of interest. Default is 0.

	Returns:
	-------
	mean_signal : numpy.ndarray
		An array containing the mean signal values for tracks of the specified class. Tracks with class not equal to
		`class_value` are excluded from the calculation.
	std_signal : numpy.ndarray
		An array containing the standard deviation of signal values for tracks of the specified class. Tracks with class
		not equal to `class_value` are excluded from the calculation.
	actual_timeline : numpy.ndarray
		An array representing the time points corresponding to the mean signal values.

	Notes:
	------
	1. This function calculates the mean and standard deviation of the specified signal for tracks of a given class.
	2. Tracks with class not equal to `class_value` are excluded from the calculation.
	3. Tracks with missing or NaN values in the specified signal are ignored during calculation.
	4. Tracks are aligned based on their 'FRAME' values and the specified `time_col` (if provided).
	"""
	
	assert signal_name in list(df.columns),"The signal you want to plot is not one of the measured features."
	if isinstance(class_value,int):
		class_value = [class_value]

	if forced_max_duration is None:
		max_duration = ceil(np.amax(df.groupby(['position','TRACK_ID']).size().values))
	else:
		max_duration = forced_max_duration
	n_tracks = len(df.groupby(['position','TRACK_ID']))
	signal_matrix = np.zeros((n_tracks,max_duration*2 + 1))
	signal_matrix[:,:] = np.nan

	trackid=0
	for track,track_group in df.loc[df[class_col].isin(class_value)].groupby(['position','TRACK_ID']):
		track_group = track_group.sort_values(by='FRAME')
		cclass = track_group[class_col].to_numpy()[0]
		if cclass != 0:
			ref_time = 0
		else:
			try:
				ref_time = floor(track_group[time_col].to_numpy()[0])
			except:
				continue
		signal = track_group[signal_name].to_numpy()
		timeline = track_group['FRAME'].to_numpy().astype(int)
		timeline_shifted = timeline - ref_time + max_duration
		signal_matrix[trackid,timeline_shifted] = signal
		trackid+=1
	
	mean_signal, std_signal = columnwise_mean(signal_matrix, min_nbr_values=min_nbr_values)
	actual_timeline = np.linspace(-max_duration, max_duration, 2*max_duration+1)
	if return_matrix:
		return mean_signal, std_signal, actual_timeline, signal_matrix
	else:
		return mean_signal, std_signal, actual_timeline

if __name__ == "__main__":

	# model = MultiScaleResNetModel(3, n_classes = 3, dropout_rate=0, dense_collection=1024, header="classifier", model_signal_length = 128)
	# print(model.summary())
	model = ResNetModelCurrent(1, 2, depth=2, use_pooling=True, n_classes = 3, dropout_rate=0.1, dense_collection=512,
				 	   header="classifier", model_signal_length = 128)
	print(model.summary())
	#plot_model(model, to_file='test.png', show_shapes=True)