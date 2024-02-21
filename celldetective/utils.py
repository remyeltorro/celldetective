import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from scipy.ndimage import shift, zoom
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.config import list_physical_devices
import configparser
from sklearn.utils.class_weight import compute_class_weight
from skimage.util import random_noise
from skimage.filters import gaussian
import random
from tifffile import imread
import json
from csbdeep.utils import normalize_mi_ma
from glob import glob
from urllib.request import urlopen
from urllib.parse import urlparse
import zipfile
from tqdm import tqdm
import shutil
import tempfile

def create_patch_mask(h, w, center=None, radius=None):

	"""

	Create a circular patch mask of given dimensions.
	Adapted from alkasm on https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array

	Parameters
	----------
	h : int
		Height of the mask. Prefer odd value.
	w : int
		Width of the mask. Prefer odd value.
	center : tuple, optional
		Coordinates of the center of the patch. If not provided, the middle of the image is used.
	radius : int or float or list, optional
		Radius of the circular patch. If not provided, the smallest distance between the center and image walls is used.
		If a list is provided, it should contain two elements representing the inner and outer radii of a circular annular patch.

	Returns
	-------
	numpy.ndarray
		Boolean mask where True values represent pixels within the circular patch or annular patch, and False values represent pixels outside.

	Notes
	-----
	The function creates a circular patch mask of the given dimensions by determining which pixels fall within the circular patch or annular patch.
	The circular patch or annular patch is centered at the specified coordinates or at the middle of the image if coordinates are not provided.
	The radius of the circular patch or annular patch is determined by the provided radius parameter or by the minimum distance between the center and image walls.
	If an annular patch is desired, the radius parameter should be a list containing the inner and outer radii respectively.

	Examples
	--------
	>>> mask = create_patch_mask(100, 100, center=(50, 50), radius=30)
	>>> print(mask)

	"""

	if center is None: # use the middle of the image
		center = (int(w/2), int(h/2))
	if radius is None: # use the smallest distance between the center and image walls
		radius = min(center[0], center[1], w-center[0], h-center[1])

	Y, X = np.ogrid[:h, :w]
	dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

	if isinstance(radius,int) or isinstance(radius,float):
		mask = dist_from_center <= radius
	elif isinstance(radius,list):
		mask = (dist_from_center <= radius[1])*(dist_from_center >= radius[0])
	else:
		print("Please provide a proper format for the radius")
		return None
		
	return mask

def rename_intensity_column(df, channels):
	
	"""
	
	Rename intensity columns in a DataFrame based on the provided channel names.

	Parameters
	----------
	df : pandas DataFrame
		The DataFrame containing the intensity columns.
	channels : list
		A list of channel names corresponding to the intensity columns.

	Returns
	-------
	pandas DataFrame
		The DataFrame with renamed intensity columns.

	Notes
	-----
	This function renames the intensity columns in a DataFrame based on the provided channel names.
	It searches for columns containing the substring 'intensity' in their names and replaces it with
	the respective channel name. The renaming is performed according to the order of the channels
	provided in the `channels` list. If multiple channels are provided, the function assumes that the
	intensity columns have a naming pattern that includes a numerical index indicating the channel.
	If only one channel is provided, the function replaces 'intensity' with the single channel name.

	Examples
	--------
	>>> data = {'intensity_0': [1, 2, 3], 'intensity_1': [4, 5, 6]}
	>>> df = pd.DataFrame(data)
	>>> channels = ['channel1', 'channel2']
	>>> renamed_df = rename_intensity_column(df, channels)
	# Rename the intensity columns in the DataFrame based on the provided channel names.

	"""

	channel_names = np.array(channels)
	channel_indices = np.arange(len(channel_names),dtype=int)

	if np.any(['intensity' in c for c in df.columns]):

		intensity_indices = [s.startswith('intensity') for s in df.columns]
		intensity_columns = df.columns[intensity_indices]

		if len(channel_names)>1:
			to_rename = {}
			for k in range(len(intensity_columns)):
				#print(intensity_columns[k])
				
				sections = np.array(re.split('-|_', intensity_columns[k]))
				test_digit = np.array([s.isdigit() for s in sections])
				index = int(sections[np.where(test_digit)[0]][-1])

				channel_name = channel_names[np.where(channel_indices==index)[0]][0]

				new_name = np.delete(sections, -1) #np.where(test_digit)[0]
				new_name = '_'.join(list(new_name))
				new_name = new_name.replace('intensity', channel_name)
				to_rename.update({intensity_columns[k]: new_name.replace('-','_')})
		else:
			to_rename = {}
			for k in range(len(intensity_columns)):
				
				sections = np.array(re.split('_', intensity_columns[k]))
				channel_name = channel_names[0]
				new_name = '_'.join(list(sections))
				new_name = new_name.replace('intensity', channel_name)
				to_rename.update({intensity_columns[k]: new_name.replace('-','_')})    


		df = df.rename(columns=to_rename)

	return df

def regression_plot(y_pred, y_true, savepath=None):

	"""

	Create a regression plot to compare predicted and ground truth values.

	Parameters
	----------
	y_pred : array-like
		Predicted values.
	y_true : array-like
		Ground truth values.
	savepath : str or None, optional
		File path to save the plot. If None, the plot is displayed but not saved. Default is None.

	Returns
	-------
	None

	Notes
	-----
	This function creates a scatter plot comparing the predicted values (`y_pred`) to the ground truth values (`y_true`)
	for regression analysis. The plot also includes a diagonal reference line to visualize the ideal prediction scenario.

	If `savepath` is provided, the plot is saved as an image file at the specified path. The file format and other
	parameters can be controlled by the `savepath` argument.

	Examples
	--------
	>>> y_pred = [1.5, 2.0, 3.2, 4.1]
	>>> y_true = [1.7, 2.1, 3.5, 4.2]
	>>> regression_plot(y_pred, y_true)
	# Create a scatter plot comparing the predicted values to the ground truth values.

	>>> regression_plot(y_pred, y_true, savepath="regression_plot.png")
	# Create a scatter plot and save it as "regression_plot.png".
	
	"""

	fig,ax = plt.subplots(1,1,figsize=(4,3))
	ax.scatter(y_pred, y_true)
	ax.set_xlabel("prediction")
	ax.set_ylabel("ground truth")
	line = np.linspace(np.amin([y_pred,y_true]),np.amax([y_pred,y_true]),1000)
	ax.plot(line,line,linestyle="--",c="k",alpha=0.7)
	plt.tight_layout()
	if savepath is not None:
		plt.savefig(savepath,bbox_inches="tight",dpi=300)
	plt.pause(2)
	plt.close()

def split_by_ratio(arr, *ratios):

	"""

	Split an array into multiple chunks based on given ratios.

	Parameters
	----------
	arr : array-like
		The input array to be split.
	*ratios : float
		Ratios specifying the proportions of each chunk. The sum of ratios should be less than or equal to 1.

	Returns
	-------
	list
		A list of arrays containing the splits/chunks of the input array.

	Notes
	-----
	This function randomly permutes the input array (`arr`) and then splits it into multiple chunks based on the provided ratios.
	The ratios determine the relative sizes of the resulting chunks. The sum of the ratios should be less than or equal to 1.
	The function uses the accumulated ratios to determine the split indices.

	The function returns a list of arrays representing the splits of the input array. The number of splits is equal to the number
	of provided ratios. If there are more ratios than splits, the extra ratios are ignored.

	Examples
	--------
	>>> arr = np.arange(10)
	>>> splits = split_by_ratio(arr, 0.6, 0.2, 0.2)
	>>> print(len(splits))
	3
	# Split the array into 3 chunks with ratios 0.6, 0.2, and 0.2.

	>>> arr = np.arange(100)
	>>> splits = split_by_ratio(arr, 0.5, 0.25)
	>>> print([len(split) for split in splits])
	[50, 25]
	# Split the array into 2 chunks with ratios 0.5 and 0.25.
	
	"""

	arr = np.random.permutation(arr)
	ind = np.add.accumulate(np.array(ratios) * len(arr)).astype(int)
	return [x.tolist() for x in np.split(arr, ind)][:len(ratios)]

def compute_weights(y):

	"""

	Compute class weights based on the input labels.

	Parameters
	----------
	y : array-like
		Array of labels.

	Returns
	-------
	dict
		A dictionary containing the computed class weights.

	Notes
	-----
	This function calculates the class weights based on the input labels (`y`) using the "balanced" method.
	The class weights are computed to address the class imbalance problem, where the weights are inversely
	proportional to the class frequencies.

	The function returns a dictionary (`class_weights`) where the keys represent the unique classes in `y`
	and the values represent the computed weights for each class.

	Examples
	--------
	>>> labels = np.array([0, 1, 0, 1, 1])
	>>> weights = compute_weights(labels)
	>>> print(weights)
	{0: 1.5, 1: 0.75}
	# Compute class weights for the binary labels.

	>>> labels = np.array([0, 1, 2, 0, 1, 2, 2])
	>>> weights = compute_weights(labels)
	>>> print(weights)
	{0: 1.1666666666666667, 1: 1.1666666666666667, 2: 0.5833333333333334}
	# Compute class weights for the multi-class labels.
	
	"""

	class_weights = compute_class_weight(
											class_weight = "balanced",
											classes = np.unique(y),
											y = y,                                                  
								)
	class_weights = dict(zip(np.unique(y), class_weights))

	return class_weights

def train_test_split(data_x, data_y1, data_y2=None, validation_size=0.25, test_size=0):

	"""

	Split the dataset into training, validation, and test sets.

	Parameters
	----------
	data_x : array-like
		Input features or independent variables.
	data_y1 : array-like
		Target variable 1.
	data_y2 : array-like
		Target variable 2.
	validation_size : float, optional
		Proportion of the dataset to include in the validation set. Default is 0.25.
	test_size : float, optional
		Proportion of the dataset to include in the test set. Default is 0.

	Returns
	-------
	dict
		A dictionary containing the split datasets.
		Keys: "x_train", "x_val", "y1_train", "y1_val", "y2_train", "y2_val".
		If test_size > 0, additional keys: "x_test", "y1_test", "y2_test".

	Notes
	-----
	This function divides the dataset into training, validation, and test sets based on the specified proportions.
	It shuffles the data and splits it according to the proportions defined by `validation_size` and `test_size`.

	The input features (`data_x`) and target variables (`data_y1`, `data_y2`) should be arrays or array-like objects
	with compatible dimensions.

	The function returns a dictionary containing the split datasets. The training set is assigned to "x_train",
	"y1_train", and "y2_train". The validation set is assigned to "x_val", "y1_val", and "y2_val". If `test_size` is
	greater than 0, the test set is assigned to "x_test", "y1_test", and "y2_test".

	"""

	n_values = len(data_x)
	randomize = np.arange(n_values)
	np.random.shuffle(randomize)

	train_percentage = 1- validation_size - test_size
	chunks = split_by_ratio(randomize, train_percentage, validation_size, test_size)

	x_train = data_x[chunks[0]]
	y1_train = data_y1[chunks[0]]
	if data_y2 is not None:
		y2_train = data_y2[chunks[0]]


	x_val = data_x[chunks[1]]
	y1_val = data_y1[chunks[1]]
	if data_y2 is not None:
		y2_val = data_y2[chunks[1]]

	ds = {"x_train": x_train, "x_val": x_val,
		 "y1_train": y1_train, "y1_val": y1_val}
	if data_y2 is not None:
		ds.update({"y2_train": y2_train, "y2_val": y2_val})

	if test_size>0:
		x_test = data_x[chunks[2]]
		y1_test = data_y1[chunks[2]]
		ds.update({"x_test": x_test, "y1_test": y1_test})
		if data_y2 is not None:
			y2_test = data_y2[chunks[2]]
			ds.update({"y2_test": y2_test})
	return ds

def remove_redundant_features(features, reference_features, channel_names=None):

	"""
	
	Remove redundant features from a list of features based on a reference feature list.

	Parameters
	----------
	features : list
		The list of features to be filtered.
	reference_features : list
		The reference list of features.
	channel_names : list or None, optional
		The list of channel names. If provided, it is used to identify and remove redundant intensity features. 
		Default is None.

	Returns
	-------
	list
		The filtered list of features without redundant entries.

	Notes
	-----
	This function removes redundant features from the input list based on a reference list of features. Features that 
	appear in the reference list are removed from the input list. Additionally, if the channel_names parameter is provided,
	it is used to identify and remove redundant intensity features. Intensity features that have the same mode (e.g., 'mean',
	'min', 'max') as any of the channel names in the reference list are also removed.

	Examples
	--------
	>>> features = ['area', 'intensity_mean', 'intensity_max', 'eccentricity']
	>>> reference_features = ['area', 'eccentricity']
	>>> filtered_features = remove_redundant_features(features, reference_features)
	>>> filtered_features
	['intensity_mean', 'intensity_max']

	>>> channel_names = ['brightfield', 'channel1', 'channel2']
	>>> filtered_features = remove_redundant_features(features, reference_features, channel_names)
	>>> filtered_features
	['area', 'eccentricity']

	"""

	new_features = features.copy()

	for f in features:

		if f in reference_features:
			new_features.remove(f)

		if ('intensity' in f) and (channel_names is not None):

			mode = f.split('_')[-1]
			pattern = [a+'_'+mode for a in channel_names]

			for p in pattern:
				if p in reference_features:
					try:
						new_features.remove(f)
					except:
						pass
	return new_features

def _estimate_scale_factor(spatial_calibration, required_spatial_calibration):

	if (required_spatial_calibration is not None)*(spatial_calibration is not None):
		scale = spatial_calibration / required_spatial_calibration
	else:
		scale = None

	epsilon = 0.05
	if scale is not None:
		if not np.all([scale >= (1-epsilon), scale <= (1+epsilon)]):
			print(f"Each frame will be rescaled by a factor {scale} to match with the model training data...")
		else:
			scale = None
	return scale

def auto_find_gpu():

	gpus = list_physical_devices('GPU')
	if len(gpus)>0:
		use_gpu = True
	else:
		use_gpu = False

	return use_gpu

def _extract_channel_indices(channels, required_channels):

	if channels is not None:
		channel_indices = []
		for ch in required_channels:
			
			try:
				idx = channels.index(ch)
			except ValueError:
				print('Mismatch between the channels required by the model and the provided channels.')
				return None

			channel_indices.append(idx)
		channel_indices = np.array(channel_indices)
	else:
		channel_indices = np.arange(len(required_channels))

	return channel_indices

def ConfigSectionMap(path,section):

	"""
	Parse the config file to extract experiment parameters
	following https://wiki.python.org/moin/ConfigParserExamples

	Parameters
	----------

	path: str
			path to the config.ini file

	section: str
			name of the section that contains the parameter

	Returns
	-------

	dict1: dictionary

	"""

	Config = configparser.ConfigParser()
	Config.read(path)
	dict1 = {}
	options = Config.options(section)
	for option in options:
		try:
			dict1[option] = Config.get(section, option)
			if dict1[option] == -1:
				DebugPrint("skip: %s" % option)
		except:
			print("exception on %s!" % option)
			dict1[option] = None
	return dict1

def _extract_channel_indices_from_config(config, channels_to_extract):

	# V2
	channels = []
	for c in channels_to_extract:
		if c!='None':
			try:
				c1 = int(ConfigSectionMap(config,"Channels")[c])
				channels.append(c1)
			except Exception as e:
				print(f"Error {e}. The channel required by the model is not available in your data... Check the configuration file.")
				channels = None
				break
		else:
			channels.append(None)

	# LEGACY
	if channels is None:
		channels = []
		for c in channels_to_extract:
			try:
				c1 = int(ConfigSectionMap(config,"MovieSettings")[c])
				channels.append(c1)
			except Exception as e:
				print(f"Error {e}. The channel required by the model is not available in your data... Check the configuration file.")
				channels = None
				break
	return channels

def _extract_nbr_channels_from_config(config, return_names=False):

	# V2
	nbr_channels = 0
	channels = []
	try:
		fields = ConfigSectionMap(config,"Channels")
		for c in fields:
			try:
				channel = int(ConfigSectionMap(config, "Channels")[c])
				nbr_channels += 1
				channels.append(c)
			except:
				pass
	except:
		pass

	if nbr_channels==0:	

		# Read channels LEGACY
		nbr_channels = 0
		channels = []
		try:
			brightfield_channel = int(ConfigSectionMap(config,"MovieSettings")["brightfield_channel"])
			nbr_channels += 1
			channels.append('brightfield_channel')
		except:
			brightfield_channel = None

		try:
			live_nuclei_channel = int(ConfigSectionMap(config,"MovieSettings")["live_nuclei_channel"])
			nbr_channels += 1
			channels.append('live_nuclei_channel')
		except:
			live_nuclei_channel = None

		try:
			dead_nuclei_channel = int(ConfigSectionMap(config,"MovieSettings")["dead_nuclei_channel"])
			nbr_channels +=1
			channels.append('dead_nuclei_channel')
		except:
			dead_nuclei_channel = None

		try:
			effector_fluo_channel = int(ConfigSectionMap(config,"MovieSettings")["effector_fluo_channel"])
			nbr_channels +=1
			channels.append('effector_fluo_channel')
		except:
			effector_fluo_channel = None

		try:
			adhesion_channel = int(ConfigSectionMap(config,"MovieSettings")["adhesion_channel"])
			nbr_channels += 1
			channels.append('adhesion_channel')
		except:
			adhesion_channel = None

		try:
			fluo_channel_1 = int(ConfigSectionMap(config,"MovieSettings")["fluo_channel_1"])
			nbr_channels += 1
			channels.append('fluo_channel_1')
		except:
			fluo_channel_1 = None	

		try:
			fluo_channel_2 = int(ConfigSectionMap(config,"MovieSettings")["fluo_channel_2"])
			nbr_channels += 1
			channels.append('fluo_channel_2')
		except:
			fluo_channel_2 = None

	if return_names:
		return nbr_channels,channels
	else:
		return nbr_channels

def _get_img_num_per_channel(channels_indices, len_movie, nbr_channels):

	len_movie = int(len_movie)
	nbr_channels = int(nbr_channels)
	
	img_num_all_channels = []
	for c in channels_indices:
		if c is not None:
			indices = np.arange(len_movie*nbr_channels)[c::nbr_channels]
		else:
			indices = [-1]*len_movie
		img_num_all_channels.append(indices)
	img_num_all_channels = np.array(img_num_all_channels, dtype=int)	
	return img_num_all_channels

def _extract_labels_from_config(config,number_of_wells):

	"""

	Extract each well's biological condition from the configuration file

	Parameters
	----------

	config: str,
			path to the configuration file

	number_of_wells: int,
			total number of wells in the experiment

	Returns
	-------

	labels: string of the biological condition for each well

	"""
	
	try:
		concentrations = ConfigSectionMap(config,"Labels")["concentrations"].split(",")
		cell_types = ConfigSectionMap(config,"Labels")["cell_types"].split(",")
		antibodies = ConfigSectionMap(config,"Labels")["antibodies"].split(",")
		pharmaceutical_agents = ConfigSectionMap(config,"Labels")["pharmaceutical_agents"].split(",")
		index = np.arange(len(concentrations)).astype(int) + 1
		if not np.all(pharmaceutical_agents=="None"):
			labels = [f"W{idx}: [CT] "+a+"; [Ab] "+b+" @ "+c+" pM "+d for idx,a,b,c,d in zip(index,cell_types,antibodies,concentrations,pharmaceutical_agents)]
		else:
			labels = [f"W{idx}: [CT] "+a+"; [Ab] "+b+" @ "+c+" pM " for idx,a,b,c in zip(index,cell_types,antibodies,concentrations)]


	except Exception as e:
		print(f"{e}: the well labels cannot be read from the concentration and cell_type fields")
		labels = np.linspace(0,number_of_wells-1,number_of_wells,dtype=str)

	return(labels)

def extract_experiment_channels(config):

	# V2
	channel_names = []
	channel_indices = []
	try:
		fields = ConfigSectionMap(config,"Channels")
		for c in fields:
			try:
				idx = int(ConfigSectionMap(config, "Channels")[c])
				channel_names.append(c)
				channel_indices.append(idx)
			except:
				pass
	except:
		pass


	if not channel_names:
		# LEGACY
		# Remap intensities to channel:
		channel_names = []
		channel_indices = []

		try:
			brightfield_channel = int(ConfigSectionMap(config,"MovieSettings")["brightfield_channel"])
			channel_names.append("brightfield_channel")
			channel_indices.append(brightfield_channel)
			#exp_channels.update({"brightfield_channel": brightfield_channel})
		except:
			pass
		try:
			live_nuclei_channel = int(ConfigSectionMap(config,"MovieSettings")["live_nuclei_channel"])
			channel_names.append("live_nuclei_channel")
			channel_indices.append(live_nuclei_channel)
			#exp_channels.update({"live_nuclei_channel": live_nuclei_channel})
		except:
			pass
		try:
			dead_nuclei_channel = int(ConfigSectionMap(config,"MovieSettings")["dead_nuclei_channel"])
			channel_names.append("dead_nuclei_channel")
			channel_indices.append(dead_nuclei_channel)
			#exp_channels.update({"dead_nuclei_channel": dead_nuclei_channel})
		except:
			pass
		try:
			effector_fluo_channel = int(ConfigSectionMap(config,"MovieSettings")["effector_fluo_channel"])
			channel_names.append("effector_fluo_channel")
			channel_indices.append(effector_fluo_channel)
			#exp_channels.update({"effector_fluo_channel": effector_fluo_channel})
		except:
			pass
		try:
			adhesion_channel = int(ConfigSectionMap(config,"MovieSettings")["adhesion_channel"])
			channel_names.append("adhesion_channel")
			channel_indices.append(adhesion_channel)
			#exp_channels.update({"adhesion_channel": adhesion_channel})
		except:
			pass
		try:
			fluo_channel_1 = int(ConfigSectionMap(config,"MovieSettings")["fluo_channel_1"])
			channel_names.append("fluo_channel_1")
			channel_indices.append(fluo_channel_1)
			#exp_channels.update({"fluo_channel_1": fluo_channel_1})
		except:
			pass
		try:
			fluo_channel_2 = int(ConfigSectionMap(config,"MovieSettings")["fluo_channel_2"])
			channel_names.append("fluo_channel_2")
			channel_indices.append(fluo_channel_2)
			#exp_channels.update({"fluo_channel_2": fluo_channel_2})
		except:
			pass

	channel_indices = np.array(channel_indices)
	channel_names = np.array(channel_names)
	reorder = np.argsort(channel_indices)
	channel_indices = channel_indices[reorder]
	channel_names = channel_names[reorder]

	return channel_names, channel_indices

def get_software_location():
	return rf"{os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]}"

def remove_trajectory_measurements(trajectories, column_labels):

	tracks = trajectories.copy()

	columns_to_keep = [column_labels['track'], column_labels['time'], column_labels['x'], column_labels['y'],column_labels['x']+'_um', column_labels['y']+'_um', 'class_id', 
					  't', 'state', 'generation', 'root', 'parent', 'ID', 't0', 'class', 'status', 'class_color', 'status_color', 'class_firstdetection', 't_firstdetection']
	cols = tracks.columns
	for c in columns_to_keep:
		if c not in cols:
			columns_to_keep.remove(c)

	keep = [x for x in columns_to_keep if x in cols]
	tracks = tracks[keep]	

	return tracks


def color_from_status(status, recently_modified=False):
	
	if not recently_modified:
		if status==0:
			return 'tab:blue'
		elif status==1:
			return 'tab:red'
		elif status==2:
			return 'yellow'
		else:
			return 'k'
	else:
		if status==0:
			return 'tab:cyan'
		elif status==1:
			return 'tab:orange'
		elif status==2:
			return 'tab:olive'
		else:
			return 'k'

def color_from_class(cclass, recently_modified=False):

	if not recently_modified:
		if cclass==0:
			return 'tab:red'
		elif cclass==1:
			return 'tab:blue'
		elif cclass==2:
			return 'yellow'
		else:
			return 'k'
	else:
		if cclass==0:
			return 'tab:orange'
		elif cclass==1:
			return 'tab:cyan'
		elif cclass==2:
			return 'tab:olive'
		else:
			return 'k'

def random_fliprot(img, mask):

	"""

	Perform random flipping of the image and the associated mask. 
	Needs YXC (channel last).

	"""
	assert img.ndim >= mask.ndim
	axes = tuple(range(mask.ndim))
	perm = tuple(np.random.permutation(axes))
	img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
	mask = mask.transpose(perm) 
	for ax in axes: 
		if np.random.rand() > 0.5:
			img = np.flip(img, axis=ax)
			mask = np.flip(mask, axis=ax)
	return img, mask 

# def random_intensity_change(img):
#     img[img!=0.] = img[img!=0.]*np.random.uniform(0.3,2)
#     img[img!=0.] += np.random.uniform(-0.2,0.2)
#     return img

def random_shift(image,mask, max_shift_amplitude=0.1):

	"""

	Perform random shift of the image in X and or Y. 
	Needs YXC (channel last).

	"""	
	
	input_shape = image.shape[0]
	max_shift = input_shape*max_shift_amplitude
	
	shift_value_x = random.choice(np.arange(max_shift))
	if np.random.random() > 0.5:
		shift_value_x*=-1

	shift_value_y = random.choice(np.arange(max_shift))
	if np.random.random() > 0.5:
		shift_value_y*=-1

	image = shift(image,[shift_value_x, shift_value_y, 0], output=np.float32, order=3, mode="constant",cval=0.0)
	mask = shift(mask,[shift_value_x,shift_value_y],order=0,mode="constant",cval=0.0)
	
	return image,mask


def blur(x,max_sigma=4.0):
	"""
	Random image blur
	"""
	sigma = np.random.random()*max_sigma
	loc_i,loc_j,loc_c = np.where(x==0.)
	x = gaussian(x, sigma, channel_axis=-1, preserve_range=True)
	x[loc_i,loc_j,loc_c] = 0.

	return x

def noise(x, apply_probability=0.5, clip_option=False):

	"""
	Apply random noise to a multichannel image

	"""

	x_noise = x.astype(float).copy()
	loc_i,loc_j,loc_c = np.where(x_noise==0.)
	options =  ['gaussian', 'localvar', 'poisson', 'speckle']

	for k in range(x_noise.shape[-1]):
		mode_order = random.sample(options, len(options))
		for m in mode_order:
			p = np.random.random()
			if p <= apply_probability:
				try:
					x_noise[:,:,k] = random_noise(x_noise[:,:,k], mode=m, clip=clip_option)
				except:
					pass

	x_noise[loc_i,loc_j,loc_c] = 0.

	return x_noise



def augmenter(x, y, flip=True, gauss_blur=True, noise_option=True, shift=True, 
	channel_extinction=False, extinction_probability=0.1, clip=False, max_sigma_blur=4, 
	apply_noise_probability=0.5, augment_probability=0.9):

	"""
	Augmentation routine for DL training.

	"""

	r = random.random()
	if r<= augment_probability:

		if flip:
			x, y = random_fliprot(x, y)

		if gauss_blur:
			x = blur(x, max_sigma=max_sigma_blur)

		if noise_option:
			x = noise(x, apply_probability=apply_noise_probability, clip_option=clip)

		if shift:
			x,y = random_shift(x,y)
	  
		if channel_extinction:
			assert extinction_probability <= 1.,'The extinction probability must be a number between 0 and 1.'
			for i in range(x.shape[-1]):
				if np.random.random() > (1 - extinction_probability):
					x[:,:,i] = 0.

	return x, y

def normalize_per_channel(X, normalization_percentile_mode=True, normalization_values=[0.1,99.99],normalization_clipping=False):
	
	assert X[0].ndim==3,'Channel axis does not exist. Abort.'
	n_channels = X[0].shape[-1]
	if isinstance(normalization_percentile_mode, bool):
		normalization_percentile_mode = [normalization_percentile_mode]*n_channels
	if isinstance(normalization_clipping, bool):
		normalization_clipping = [normalization_clipping]*n_channels
	if len(normalization_values)==2 and not isinstance(normalization_values[0], list):
		normalization_values = [normalization_values]*n_channels

	assert len(normalization_values)==n_channels
	assert len(normalization_clipping)==n_channels
	assert len(normalization_percentile_mode)==n_channels
	
	for i in range(len(X)):
		x = X[i]
		loc_i,loc_j,loc_c = np.where(x==0.)
		norm_x = np.zeros_like(x, dtype=np.float32)
		for k in range(x.shape[-1]):
			chan = x[:,:,k]
			if not np.all(chan.flatten()==0):
				if normalization_percentile_mode[k]:
					min_val = np.percentile(chan[chan!=0.].flatten(), normalization_values[k][0])
					max_val = np.percentile(chan[chan!=0.].flatten(), normalization_values[k][1])
				else:
					min_val = normalization_values[k][0]
					max_val = normalization_values[k][1]

				clip_option = normalization_clipping[k]
				norm_x[:,:,k] = normalize_mi_ma(chan.astype(np.float32), min_val, max_val, clip=clip_option, eps=1e-20, dtype=np.float32)
		
		X[i] = norm_x

	return X

def load_image_dataset(datasets, channels, train_spatial_calibration=None, mask_suffix='labelled'):

	if isinstance(channels, str):
		channels = [channels]
		
	assert isinstance(channels, list),'Please provide a list of channels. Abort.'

	X = []; Y = [];

	for ds in datasets:
		print(f'Loading data from dataset {ds}...')
		if not ds.endswith(os.sep):
			ds+=os.sep
		img_paths = list(set(glob(ds+'*.tif')) - set(glob(ds+f'*_{mask_suffix}.tif')))
		for im in img_paths:
			print(f'{im=}')
			mask_path = os.sep.join([os.path.split(im)[0],os.path.split(im)[-1].replace('.tif', f'_{mask_suffix}.tif')])
			if os.path.exists(mask_path):
				# load image and mask
				image = imread(im)
				if image.ndim==2:
					image = image[np.newaxis]
				if image.ndim>3:
					print('Invalid image shape, skipping')
					continue
				mask = imread(mask_path)
				config_path = im.replace('.tif','.json')
				if os.path.exists(config_path):
					# Load config
					with open(config_path, 'r') as f:
						config = json.load(f)
					try:
						ch_idx = []
						for c in channels:
							if c!='None':
								idx = config['channels'].index(c)
								ch_idx.append(idx)
							else:
								ch_idx.append(np.nan)
						im_calib = config['spatial_calibration']
					except Exception as e:
						print(e,' channels and/or spatial calibration could not be found in the config... Skipping image.')
						continue
				
				ch_idx = np.array(ch_idx)
				ch_idx_safe = np.copy(ch_idx)
				ch_idx_safe[ch_idx_safe!=ch_idx_safe] = 0
				ch_idx_safe = ch_idx_safe.astype(int)
				print(ch_idx_safe)
				image = image[ch_idx_safe]
				image[np.where(ch_idx!=ch_idx)[0],:,:] = 0

				image = np.moveaxis(image,0,-1)
				assert image.ndim==3,'The image has a wrong number of dimensions. Abort.'
	
				if im_calib != train_spatial_calibration:
					factor = im_calib / train_spatial_calibration
					print(f'{im_calib=}, {train_spatial_calibration=}, {factor=}')
					image = zoom(image, [factor,factor,1], order=3)
					mask = zoom(mask, [factor,factor], order=0)        
					
			X.append(image)
			Y.append(mask)

	assert len(X)==len(Y),'The number of images does not match with the number of masks... Abort.'
	return X,Y


def download_url_to_file(url, dst, progress=True):
	r"""Download object at the given URL to a local path.
			Thanks to torch, slightly modified, from Cellpose
	Args:
		url (string): URL of the object to download
		dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
		progress (bool, optional): whether or not to display a progress bar to stderr
			Default: True
	"""
	file_size = None
	import ssl
	ssl._create_default_https_context = ssl._create_unverified_context
	u = urlopen(url)
	meta = u.info()
	if hasattr(meta, 'getheaders'):
		content_length = meta.getheaders("Content-Length")
	else:
		content_length = meta.get_all("Content-Length")
	if content_length is not None and len(content_length) > 0:
		file_size = int(content_length[0])
	# We deliberately save it in a temp file and move it after
	dst = os.path.expanduser(dst)
	dst_dir = os.path.dirname(dst)
	f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
	try:
		with tqdm(total=file_size, disable=not progress,
				  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
			while True:
				buffer = u.read(8192) #8192
				if len(buffer) == 0:
					break
				f.write(buffer)
				pbar.update(len(buffer))
		f.close()
		shutil.move(f.name, dst)
	finally:
		f.close()
		if os.path.exists(f.name):
			os.remove(f.name)

def get_zenodo_files(cat=None):


	zenodo_json = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],"celldetective", "links", "zenodo.json"])
	with open(zenodo_json,"r") as f:
		zenodo_json = json.load(f)
	all_files = list(zenodo_json['files']['entries'].keys())
	all_files_short = [f.replace(".zip","") for f in all_files]
	
	categories = []
	for f in all_files_short:
		if f.startswith('CP') or f.startswith('SD'):
			category = os.sep.join(['models','segmentation_generic'])
		elif f.startswith('MCF7'):
			category = os.sep.join(['models','segmentation_targets'])
		elif f.startswith('primNK'):
			category = os.sep.join(['models','segmentation_effectors'])
		elif f.startswith('demo'):
			category = 'demos'
		elif f.startswith('db-si'):
			category = os.sep.join(['datasets','signal_annotations'])
		elif f.startswith('db'):
			category = os.sep.join(['datasets','segmentation_annotations'])
		else:
			category = os.sep.join(['models','signal_detection'])
		categories.append(category)
	
	if cat is not None:
		assert cat in [os.sep.join(['models','segmentation_generic']), os.sep.join(['models','segmentation_targets']), os.sep.join(['models','segmentation_effectors']), \
						   'demos', os.sep.join(['datasets','signal_annotations']), os.sep.join(['datasets','segmentation_annotations']),  os.sep.join(['models','signal_detection'])]
		categories = np.array(categories)
		all_files_short = np.array(all_files_short)
		return list(all_files_short[np.where(categories==cat)[0]])
	else:
		return all_files_short,categories

def download_zenodo_file(file, output_dir):
	
	zenodo_json = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],"celldetective", "links", "zenodo.json"])
	with open(zenodo_json,"r") as f:
		zenodo_json = json.load(f)
	all_files = list(zenodo_json['files']['entries'].keys())
	all_files_short = [f.replace(".zip","") for f in all_files]
	zenodo_url = zenodo_json['links']['files'].replace('api/','')
	full_links = ["/".join([zenodo_url, f]) for f in all_files]
	index = all_files_short.index(file)
	zip_url = full_links[index]
	
	path_to_zip_file = os.sep.join([output_dir, 'temp.zip'])
	download_url_to_file(fr"{zip_url}",path_to_zip_file)
	with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
		zip_ref.extractall(output_dir)
	
	file_to_rename = glob(os.sep.join([output_dir,file,"*[!.json][!.png][!.h5][!.csv][!.npy][!.tif][!.ini]"]))
	if len(file_to_rename)>0 and not file_to_rename[0].endswith(os.sep):
		os.rename(file_to_rename[0], os.sep.join([output_dir,file,file]))

	if file=="db_mcf7_nuclei_w_primary_NK":
		os.rename(os.sep.join([output_dir,file.replace('db_','')]), os.sep.join([output_dir,file]))
	if file=="db_primary_NK_w_mcf7":
		os.rename(os.sep.join([output_dir,file.replace('db_','')]), os.sep.join([output_dir,file]))
	if file=='db-si-NucPI':
		os.rename(os.sep.join([output_dir,'db2-NucPI']), os.sep.join([output_dir,file]))
	if file=='db-si-NucCondensation':
		os.rename(os.sep.join([output_dir,'db1-NucCondensation']), os.sep.join([output_dir,file]))

	os.remove(path_to_zip_file)