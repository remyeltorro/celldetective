import numpy as np
import os
import json

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import Precision
from tensorflow.keras.models import load_model,clone_model
from tensorflow.config.experimental import list_physical_devices, set_memory_growth
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import jaccard_score, balanced_accuracy_score
from scipy.interpolate import interp1d

from .io import get_signal_models_list
from .tracking import clean_trajectories
from .utils import regression_plot, train_test_split, compute_weights


def analyze_signals(trajectories, model, interpolate_na=True,
					selected_signals=None,
					column_labels = {'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}):

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


	_,model_path = get_signal_models_list(return_path=True)
	complete_path = model_path+model
	model_config_path = complete_path+'/config_input.json'
	assert os.path.exists(complete_path),f'Model {model} could not be located in folder {model_path}... Abort.'
	assert os.path.exists(model_config_path),f'Model configuration could not be located in folder {model_path}... Abort.'

	available_signals = trajectories.columns

	f = open(model_config_path)
	config = json.load(f)
	required_signals = config["channels"]
	
	if selected_signals is None:
		selected_signals = []
		for s in required_signals:
			pattern_test = [s in a for a in available_signals]
			assert np.any(pattern_test),f'No signal matches with the requirements of the model {required_signals}. Please pass the signals manually with the argument selected_signals or add measurements. Abort.'
			valid_columns = np.array(available_signals)[np.array(pattern_test)]
			if len(valid_columns)==1:
				selected_signals.append(valid_columns[0])
			else:
				#print(test_number_of_nan(trajectories, valid_columns))
				selected_signals.append(valid_columns[0])
				# do something more complicated in case of one to many columns
				#pass
	else:
		assert len(selected_signals)==len(required_signals),f'Mismatch between the number of required signals {required_signals} and the provided signals {selected_signals}... Abort.'

	trajectories = clean_trajectories(trajectories, interpolate_na=interpolate_na, interpolate_position_gaps=interpolate_na, column_labels=column_labels)

	max_signal_size = np.amax(trajectories.groupby(column_labels['track']).size().to_numpy())
	tracks = trajectories[column_labels['track']].unique()
	signals = np.zeros((len(tracks),max_signal_size, len(selected_signals)))

	for i,(tid,group) in enumerate(trajectories.groupby(column_labels['track'])):
		frames = group[column_labels['time']].to_numpy().astype(int)
		for j,col in enumerate(selected_signals):
			signal = group[col].to_numpy()
			signals[i,frames,j] = signal
	
	model = SignalDetectionModel(pretrained=complete_path)
	classes = model.predict_class(signals)
	times_recast = model.predict_time_of_interest(signals)

	for i,(tid,group) in enumerate(trajectories.groupby(column_labels['track'])):
		indices = group.index
		trajectories.loc[indices,'class'] = classes[i]
		trajectories.loc[indices,'t0'] = times_recast[i]
	print('Done.')

	return trajectories

class SignalDetectionModel(object):
	
	def __init__(self, path=None, pretrained=None, channel_option=["live_nuclei_channel","dead_nuclei_channel"], model_signal_length=128, n_channels=2, 
				n_conv=3, n_classes=3, dense_collection=128, dropout_rate=0.1):
		
		self.prep_gpu()

		self.model_signal_length = model_signal_length
		self.channel_option = channel_option
		self.pretrained = pretrained
		self.n_channels = n_channels
		self.n_conv = n_conv
		self.n_classes = n_classes
		self.dense_collection = dense_collection
		self.dropout_rate = dropout_rate

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
			self.model_class = load_model(self.pretrained+"/classifier.h5")
			self.model_class.load_weights(self.pretrained+"/classifier.h5")
			print("Classifier successfully loaded...")
		except Exception as e:
			print(f"Error {e}...")
			self.model_class = None
		try:
			self.model_reg = load_model(self.pretrained+"/regressor.h5")
			self.model_reg.load_weights(self.pretrained+"/regressor.h5")
			print("Regressor successfully loaded...")
		except Exception as e:
			print(f"Error {e}...")
			self.model_reg = None

		# load config
		with open(self.pretrained+"/config_input.json") as config_file:
			model_config = json.load(config_file)

		req_channels = model_config["channels"]
		print(f"Required channels read from pretrained model: {req_channels}")
		self.channel_option = req_channels

		self.n_channels = self.model_class.layers[0].input_shape[0][-1]
		self.model_signal_length = self.model_class.layers[0].input_shape[0][-2]
		self.n_classes = self.model_class.layers[-1].output_shape[-1]

		assert self.model_class.layers[0].input_shape[0] == self.model_reg.layers[0].input_shape[0], f"mismatch between input shape of classification: {self.model_class.layers[0].input_shape[0]} and regression {self.model_reg.layers[0].input_shape[0]} models... Error."


	def create_models_from_scratch(self):

		"""
		
		Generate new ResNet models for classification and regression with the chosen specifications

		"""

		self.model_class = ResNetModel(n_channels=self.n_channels,
									n_blocks=self.n_conv,
									n_classes = self.n_classes,
									dense_collection=self.dense_collection,
									dropout_rate=self.dropout_rate, 
									header="classifier", 
									model_signal_length = self.model_signal_length
									)

		self.model_reg = ResNetModel(n_channels=self.n_channels,
									n_blocks=self.n_conv,
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
	
	def fit_from_directory(self, ds_folders, channel_option=["live_nuclei_channel","dead_nuclei_channel"], model_name=None, target_directory=None, augment=True, augmentation_factor=2, 
						  validation_split=0.25, test_split=0.0, batch_size = 64, epochs=300, recompile_pretrained=False, learning_rate=0.01,
						  loss_reg="mse", loss_class = CategoricalCrossentropy(from_logits=False)):
		"""
		
		Load annotations in directory, create dataset, fit model
		
		"""
		
		self.ds_folders = ds_folders
		self.batch_size = batch_size
		self.epochs = epochs
		self.validation_split = validation_split
		self.test_split = test_split
		self.augment = augment
		self.augmentation_factor = augmentation_factor
		self.model_name = model_name
		self.target_directory = target_directory
		self.model_folder = self.target_directory + "/" + self.model_name
		self.recompile_pretrained = recompile_pretrained
		self.learning_rate = learning_rate
		self.loss_reg = loss_reg
		self.loss_class = loss_class

		if os.path.exists(self.model_folder):
			shutil.rmtree(self.model_folder)
		os.mkdir(self.model_folder)

		self.channel_option = channel_option
		assert self.n_channels==len(self.channel_option), f'Mismatch between the channel option and the number of channels of the model...'
		
		self.list_of_sets = []
		print(self.ds_folders)
		for f in self.ds_folders:
			self.list_of_sets.extend(glob(f+"/*.npy"))
		print(f"Found {len(self.list_of_sets)} annotation files...")
		self.generate_sets()

		self.train_classifier()
		self.train_regressor()

		config_input = {"channels": self.channel_option, "model_signal_length": self.model_signal_length}
		json_string = json.dumps(config_input)
		with open(self.model_folder+f"/config_input.json", 'w') as outfile:
			outfile.write(json_string)

	def fit(self, x_train, y_time_train, y_class_train, normalize=True, pad=True, validation_data=None, test_data=None, channel_option=["live_nuclei_channel","dead_nuclei_channel"], model_name=None, 
			target_directory=None, augment=True, augmentation_factor=3, validation_split=0.25, batch_size = 64, epochs=300,
			recompile_pretrained=False, learning_rate=0.001, loss_reg="mse", loss_class = CategoricalCrossentropy(from_logits=False)):

		self.normalize = normalize
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
			self.x_train = normalize_signal_set(self.x_train, self.channel_option)


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
					self.x_val = normalize_signal_set(self.x_val, self.channel_option)
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
					self.x_test = normalize_signal_set(self.x_test, self.channel_option)
			except Exception as e:
				print("Could not load test data, error {e}...")


		self.batch_size = batch_size
		self.epochs = epochs
		self.augment = augment
		self.augmentation_factor = augmentation_factor
		self.model_name = model_name
		self.target_directory = target_directory
		self.model_folder = self.target_directory + "/" + self.model_name
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

		if self.normalize:
			self.x = normalize_signal_set(self.x, self.channel_option)

		if self.pad:
			self.x = pad_to_model_length(self.x, self.model_signal_length)

		# implement auto interpolation here!!

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
		if class_predictions is not None:
			self.class_predictions = class_predictions

		if self.normalize:
			self.x = normalize_signal_set(self.x, self.channel_option)

		if self.pad:
			self.x = pad_to_model_length(self.x, self.model_signal_length)

		self.x = self.interpolate_signals(self.x)

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
							  metrics=['accuracy', Precision()])
		else:
			print("Compiling the classifier...")
			self.model_class.compile(optimizer=Adam(learning_rate=self.learning_rate), 
						  loss=self.loss_class, 
						  metrics=['accuracy', Precision()])
			
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
		self.model_class = load_model(self.model_folder+"/classifier.h5")
		self.model_class.load_weights(self.model_folder+"/classifier.h5")
		
		if hasattr(self, 'x_test'):
			
			predictions = self.model_class.predict(self.x_test).argmax(axis=1)
			ground_truth = self.y_class_test.argmax(axis=1)
			assert predictions.shape==ground_truth.shape,"Mismatch in shape between the predictions and the ground truth..."
			
			title="Test data"
			IoU_score = jaccard_score(ground_truth, predictions, average=None)
			balanced_accuracy = balanced_accuracy_score(ground_truth, predictions)
			print(f"Test IoU score: {IoU_score}")
			print(f"Test Balanced accuracy score: {balanced_accuracy}")

			# Confusion matrix on test set
			results = confusion_matrix(ground_truth,predictions)
			try:
				plot_confusion_matrix(results, ["dead","alive","miscellaneous"], output_dir=self.model_folder+"/", title=title)
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
			print(f"Validation IoU score: {IoU_score}")
			print(f"Validation Balanced accuracy score: {balanced_accuracy}")

			# Confusion matrix on validation set
			results = confusion_matrix(ground_truth,predictions)
			try:
				plot_confusion_matrix(results, ["dead","alive","miscellaneous"], output_dir=self.model_folder+"/", title=title)
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
		

		# Evaluate best model 
		self.model_reg = load_model(self.model_folder+"/regressor.h5")
		self.model_reg.load_weights(self.model_folder+"/regressor.h5")
		self.evaluate_regression_model()
		


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
				plt.savefig(self.model_folder+"/regression_loss.png",bbox_inches="tight",dpi=300)
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
				plt.savefig(self.model_folder+"/classification_loss.png",bbox_inches="tight",dpi=300)
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

		if hasattr(self, 'x_test'):

			print("Evaluate on test set...")
			predictions = self.model_reg.predict(self.x_test[np.argmax(self.y_class_test,axis=1)==0], batch_size=self.batch_size)[:,0]
			ground_truth = self.y_time_test[np.argmax(self.y_class_test,axis=1)==0]
			assert predictions.shape==ground_truth.shape,"Shape mismatch between predictions and ground truths..."
			test_error = mse(ground_truth, predictions).numpy()
			print(f"MSE on test set: {test_error}...")
			regression_plot(predictions, ground_truth, savepath=self.model_folder+"/test_regression.png")

		if hasattr(self, 'x_val'):
			# Validation set
			predictions = self.model_reg.predict(self.x_val[np.argmax(self.y_class_val,axis=1)==0], batch_size=self.batch_size)[:,0]
			ground_truth = self.y_time_val[np.argmax(self.y_class_val,axis=1)==0]
			assert predictions.shape==ground_truth.shape,"Shape mismatch between predictions and ground truths..."
			val_error = mse(ground_truth, predictions).numpy()
			regression_plot(predictions, ground_truth, savepath=self.model_folder+"/validation_regression.png")
			print(f"MSE on validation set: {val_error}...")


	def gather_callbacks(self, mode):

		"""

		Gather a list of callbacks based on the training mode.
		
		Parameters:
			- mode: str
				Training mode, either "classifier" or "regressor".
		
		"""
		
		self.cb = []
		
		if mode=="classifier":
			
			reduce_lr = ReduceLROnPlateau(monitor='val_precision', factor=0.1, patience=200,
										  cooldown=10, min_lr=5e-10, min_delta=1.0E-10,
										  verbose=1,mode="max")
			self.cb.append(reduce_lr)
			csv_logger = CSVLogger(self.model_folder+'/log_classifier.csv', append=True, separator=';')
			self.cb.append(csv_logger)
			checkpoint_path = self.model_folder+"/classifier.h5"
			cp_callback = ModelCheckpoint(checkpoint_path,monitor="val_precision",mode="max",verbose=1,save_best_only=True,save_weights_only=False,save_freq="epoch")
			self.cb.append(cp_callback)
			
			callback_stop = EarlyStopping(monitor='val_precision', patience=1000)
			self.cb.append(callback_stop)
			
		elif mode=="regressor":
			
			reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=200,
										  cooldown=10, min_lr=5e-10, min_delta=1.0E-10,
										  verbose=1,mode="min")
			self.cb.append(reduce_lr)

			csv_logger = CSVLogger(self.model_folder+'/log_regressor.csv', append=True, separator=';')
			self.cb.append(csv_logger)
			
			checkpoint_path = self.model_folder+"/regressor.h5"
			cp_callback = ModelCheckpoint(checkpoint_path,monitor="val_loss",mode="min",verbose=1,save_best_only=True,save_weights_only=False,save_freq="epoch")
			self.cb.append(cp_callback)
			
			callback_stop = EarlyStopping(monitor='val_loss', patience=1000)
			self.cb.append(callback_stop)            
		
		log_dir = self.model_folder+"/"
		cb_tb = TensorBoard(log_dir=log_dir, update_freq='batch')
		self.cb.append(cb_tb)
		
		
	
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
		key_to_check = self.channel_option[0]
		if key_to_check in set_k[0]:

			signal_lengths = [len(l[key_to_check]) for l in set_k]
			max_length = np.amax(signal_lengths)

			fluo = np.zeros((len(set_k),max_length,self.n_channels))
			classes = np.zeros(len(set_k))
			times_of_interest = np.zeros(len(set_k))
			
			for k in range(len(set_k)):
				
				for i in range(self.n_channels):
					try:
						fluo[k,:,i] = set_k[k][self.channel_option[i]]
					except:
						print(f"Attribute {self.channel_option[i]} not found in annotation...")
						pass

				classes[k] = set_k[k]["class"]
				times_of_interest[k] = set_k[k]["time_of_interest"]

			# Correct absurd times of interest
			times_of_interest[np.nonzero(classes)] = -1
			times_of_interest[(times_of_interest<=0.0)] = -1

			# Attempt per-set normalization
			fluo = normalize_signal_set(fluo, self.channel_option)
			fluo = pad_to_model_length(fluo, self.model_signal_length)
			# Trivial normalization for time of interest
			times_of_interest /= self.model_signal_length
			
			# Add to global dataset
			self.x_set.extend(fluo)
			self.y_time_set.extend(times_of_interest)
			self.y_class_set.extend(classes)

def normalize_signal_set(signal_set, channel_option, percentile_alive=[0.01,99.99], percentile_dead=[0.5,99.99], percentile_generic=[0.01,99.99]):

	"""

	Normalize the signals from a set of single-cell signals.

	Parameters
	----------
	signal_set : array-like
		The set of single-cell signals to be normalized.
	channel_option : list
		The list of channel names specifying the channels in the signal set.
	percentile_alive : list, optional
		The percentile range to use for normalization of live nuclei or blue channel signals. Default is [0.01, 99.99].
	percentile_dead : list, optional
		The percentile range to use for normalization of dead nuclei or red channel signals. Default is [0.5, 99.99].
	percentile_generic : list, optional
		The percentile range to use for normalization of generic signals (other channels). Default is [0.01, 99.99].

	Returns
	-------
	array-like
		The normalized signal set.

	Notes
	-----
	This function performs signal normalization on a set of single-cell signals based on the provided channel names and
	percentile ranges. The normalization process is specific to different channel types: live nuclei/blue channels,
	dead nuclei/red channels, and generic channels.

	For channels specified as dead nuclei (using the string "dead_nuclei_channel" or containing "RED" in the channel name),
	the function calculates the minimum and maximum percentiles within the specified percentile ranges on the initial and
	final frames of the signal set. It then subtracts the minimum percentile value and divides by the range (max - min)
	for each channel.

	For channels specified as live nuclei (using the string "live_nuclei_channel" or containing "BLUE" in the channel name),
	the function calculates the minimum and maximum percentiles within the specified percentile ranges on the initial frame
	of the signal set. It then subtracts the minimum percentile value and divides by the range (max - min) for each channel.

	For channels not specified as live or dead nuclei, the function calculates the minimum and maximum percentiles within
	the specified percentile ranges for the entire signal set. It then subtracts the minimum percentile value and divides
	by the range (max - min) for each channel.

	Examples
	--------
	>>> signal_set = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
	>>> channel_option = ["dead_nuclei_channel", "live_nuclei_channel", "generic_channel"]
	>>> normalized_signals = normalize_signal_set(signal_set, channel_option)

	"""

	for k,channel in enumerate(channel_option):


		if ("dead_nuclei_channel" in channel) or ("RED" in channel):

			min_percentile_dead, max_percentile_dead = percentile_dead
			min_set = signal_set[:,0,k]
			max_set = signal_set[:,-1,k]
			min_fluo_dead = np.nanpercentile(min_set[np.nonzero(min_set)], min_percentile_dead) # 5 % on initial frame where barely any dead are expected
			max_fluo_dead = np.nanpercentile(max_set[np.nonzero(max_set)], max_percentile_dead) # 99th percentile on last fluo frame
			signal_set[:,:,k] -= min_fluo_dead
			signal_set[:,:,k] /= (max_fluo_dead - min_fluo_dead)

		if ("live_nuclei_channel" in channel) or ("BLUE" in channel):
		
			min_percentile_alive, max_percentile_alive = percentile_alive
			values = signal_set[:,0,k]
			min_fluo_alive = np.nanpercentile(values[np.nonzero(values)], min_percentile_alive) # safe 0.5% of Hoescht on initial frame
			max_fluo_alive = np.nanpercentile(values[np.nonzero(values)], max_percentile_alive)
			signal_set[:,:,k] -= min_fluo_alive
			signal_set[:,:,k] /= (max_fluo_alive - min_fluo_alive)

		else:

			min_percentile, max_percentile = percentile_generic
			values = signal_set[:,:,k]
			min_signal = np.nanpercentile(values[np.nonzero(values)], min_percentile)
			max_signal= np.nanpercentile(values[np.nonzero(values)], max_percentile)
			signal_set[:,:,k] -= min_signal
			signal_set[:,:,k] /= (max_signal - min_signal)

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
	r = random()
	if r<= probability:

		if time_shift:
			# do not time shift miscellaneous cells
			if cclass.argmax()!=2.:
				assert time_of_interest is not None, f"Please provide valid lysis times"
				signal,time_of_interest = random_time_shift(signal, time_of_interest, model_signal_length)

		signal = random_intensity_change(signal)
		signal = gauss_noise(signal)

	return signal, time_of_interest/model_signal_length, cclass


def residual_block1D(x, number_of_filters,match_filter_size=True):

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
	x = Conv1D(number_of_filters, kernel_size=8, strides=1,padding="same")(x_skip)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)

	x = Conv1D(number_of_filters, kernel_size=5, strides=1,padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)

	x = Conv1D(number_of_filters, kernel_size=3,padding="same")(x)
	x = BatchNormalization()(x)

	if match_filter_size:
		x_skip = Conv1D(number_of_filters, kernel_size=1, padding="same")(x_skip)

	# Add the skip connection to the regular mapping
	x = Add()([x, x_skip])

	# Nonlinearly activate the result
	x = Activation("relu")(x)

	# Return the result
	return x

def ResNetModel(n_channels, n_blocks, n_classes = 3, dropout_rate=0, dense_collection=0,
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

	for i in range(n_blocks):
		if i==0:
			x2 = residual_block1D(inputs,64)
		else:
			x2 = residual_block1D(x2,128)
	x2 = MaxPooling1D()(x2)

	for i in range(n_blocks):
		x2 = residual_block1D(x2,128)

	x2 = GlobalAveragePooling1D()(x2)
	if dense_collection>0:
		x2 = Dense(dense_collection)(x2)
	if dropout_rate>0:
		x2 = Dropout(dropout_rate)(x2)

	x2 = Dense(neurons_final,activation=final_activation,name=header)(x2)
	model = Model(inputs, x2, name=header) 

	return model

def analyze_signals_at_position(pos, mode, use_gpu=True):
	pass