import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import regionprops_table
from scipy.ndimage.morphology import distance_transform_edt
from functools import reduce
from mahotas.features import haralick
from scipy.ndimage import zoom
import os
import subprocess
from celldetective.utils import rename_intensity_column, create_patch_mask, remove_redundant_features

abs_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]+'/celldetective'

def measure(stack=None, labels=None, trajectories=None, channel_names=None,
			features=None, intensity_measurement_radii=None, isotropic_operations=['mean'], border_distances=None,
			haralick_options=None, column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}, clear_previous=False):

	"""
	
	Perform measurements on a stack of images or labels.

	Parameters
	----------
	stack : numpy array, optional
		Stack of images with shape (T, Y, X, C), where T is the number of frames, Y and X are the spatial dimensions,
		and C is the number of channels. Default is None.
	labels : numpy array, optional
		Label stack with shape (T, Y, X) representing cell segmentations. Default is None.
	trajectories : pandas DataFrame, optional
		DataFrame of cell trajectories with columns specified in `column_labels`. Default is None.
	channel_names : list, optional
		List of channel names corresponding to the image stack. Default is None.
	features : list, optional
		List of features to measure using the `measure_features` function. Default is None.
	intensity_measurement_radii : int, float, or list, optional
		Radius or list of radii specifying the size of the isotropic measurement area for intensity measurements.
		If a single value is provided, a circular measurement area is used. If a list of values is provided, multiple
		measurements are performed using ring-shaped measurement areas. Default is None.
	isotropic_operations : list, optional
		List of operations to perform on the isotropic intensity values. Default is ['mean'].
	border_distances : int, float, or list, optional
		Distance or list of distances specifying the size of the border region for intensity measurements.
		If a single value is provided, measurements are performed at a fixed distance from the cell borders.
		If a list of values is provided, measurements are performed at multiple border distances. Default is None.
	haralick_options : dict, optional
		Dictionary of options for Haralick feature measurements. Default is None.
	column_labels : dict, optional
		Dictionary containing the column labels for the DataFrame. Default is {'track': "TRACK_ID",
		'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.

	Returns
	-------
	pandas DataFrame
		DataFrame containing the measured features and intensities.

	Notes
	-----
	This function performs measurements on a stack of images or labels. If both `stack` and `labels` are provided,
	measurements are performed on each frame of the stack. The measurements include isotropic intensity values, computed
	using the `measure_isotropic_intensity` function, and additional features, computed using the `measure_features` function.
	The intensity measurements are performed at the positions specified in the `trajectories` DataFrame, using the
	specified `intensity_measurement_radii` and `border_distances`. The resulting measurements are combined into a single
	DataFrame and returned.

	Examples
	--------
	>>> stack = np.random.rand(10, 100, 100, 3)
	>>> labels = np.random.randint(0, 2, (10, 100, 100))
	>>> trajectories = pd.DataFrame({'TRACK_ID': [1, 2, 3], 'FRAME': [1, 1, 1],
	...                              'POSITION_X': [10, 20, 30], 'POSITION_Y': [15, 25, 35]})
	>>> channel_names = ['channel1', 'channel2', 'channel3']
	>>> features = ['area', 'intensity_mean']
	>>> intensity_measurement_radii = [5, 10]
	>>> border_distances = 2
	>>> measurements = measure(stack=stack, labels=labels, trajectories=trajectories, channel_names=channel_names,
	...                        features=features, intensity_measurement_radii=intensity_measurement_radii,
	...                        border_distances=border_distances)
	# Perform measurements on the stack, labels, and trajectories, computing isotropic intensities and additional features.

	"""


	do_iso_intensities = True
	do_features = True


	# Check that conditions are satisfied to perform measurements
	assert (labels is not None) or (stack is not None),'Please pass a stack and/or labels... Abort.'
	if (labels is not None)*(stack is not None):
		assert labels.shape==stack.shape[:-1],f"Shape mismatch between the stack of shape {stack.shape} and the segmentation {labels.shape}..."
	
	# Condition to compute features
	if labels is None:
		do_features = False
		nbr_frames = len(stack)
		print('No labels were provided... Features will not be computed...')
	else:
		nbr_frames = len(labels)

	# Condition to compute isotropic intensities
	if (stack is None) or (trajectories is None) or (intensity_measurement_radii is None):
		do_iso_intensities = False
		print('Either no image, no positions or no radii were provided... Isotropic intensities will not be computed...')

	# Compensate for non provided channel names
	if (stack is not None)*(channel_names is None):
		nbr_channels = stack.shape[-1]
		channel_names = [f'intensity-{k}' for k in range(nbr_channels)]

	if isinstance(intensity_measurement_radii, int) or isinstance(intensity_measurement_radii, float):
		intensity_measurement_radii = [intensity_measurement_radii]

	if isinstance(border_distances, int) or isinstance(border_distances, float):
		border_distances = [border_distances]

	if features is not None:
		features = remove_redundant_features(features, trajectories.columns,
											channel_names=channel_names)

	if features is None:
		features = []

	# Prep for the case where no trajectory is provided but still want to measure isotropic intensities...
	if (trajectories is None):
		do_features = True
		features += ['centroid']
	else:
		if clear_previous:
			trajectories = remove_trajectory_measurements(trajectories, column_labels)

	timestep_dataframes = []

	for t in tqdm(range(nbr_frames),desc='frame'):

		if stack is not None:
			img = stack[t]
		else:
			img = None
		if labels is not None:
			lbl = labels[t]
		else:
			lbl = None

		if trajectories is not None:
			positions_at_t = trajectories.loc[trajectories[column_labels['time']]==t].copy()

		if do_features:
			feature_table = measure_features(img, lbl, features = features, border_dist=border_distances, 
											channels=channel_names, haralick_options=haralick_options, verbose=False)
			if trajectories is None:
				# Use the centroids as estimate for the location of the cells, to be passed to the measure_isotropic_intensity function.
				positions_at_t = feature_table[['centroid-1', 'centroid-0','class_id']].copy()
				positions_at_t['ID'] = np.arange(len(positions_at_t))	# temporary ID for the cells, that will be reset at the end since they are not tracked
				positions_at_t.rename(columns={'centroid-1': 'POSITION_X', 'centroid-0': 'POSITION_Y'},inplace=True)
				positions_at_t['FRAME'] = int(t)
				column_labels = {'track': "ID", 'time': column_labels['time'], 'x': column_labels['x'], 'y': column_labels['y']}

		# Isotropic measurements (circle, ring)
		if do_iso_intensities:
			iso_table = measure_isotropic_intensity(positions_at_t, img, channels=channel_names, intensity_measurement_radii=intensity_measurement_radii,
													column_labels=column_labels, operations=isotropic_operations, verbose=False)

		if do_iso_intensities*do_features:
			measurements_at_t = iso_table.merge(feature_table, how='outer', on='class_id')
		elif do_iso_intensities*(not do_features):
			measurements_at_t = iso_table
		elif do_features*(trajectories is not None):
			measurements_at_t = positions_at_t.merge(feature_table, how='outer', on='class_id')
		elif do_features*(trajectories is None):
			measurements_at_t = positions_at_t

		timestep_dataframes.append(measurements_at_t)

	measurements = pd.concat(timestep_dataframes)
	if trajectories is not None:
		measurements = measurements.sort_values(by=[column_labels['track'],column_labels['time']])
		measurements = measurements.dropna(subset=[column_labels['track']])
	else:
		measurements['ID'] = np.arange(len(df))

	measurements = measurements.reset_index(drop=True)

	return measurements

def contour_of_instance_segmentation(label, distance):

	"""

	Generate an instance mask containing the contour of the segmented objects.

	Parameters
	----------
	label : ndarray
		The instance segmentation labels.
	distance : int, float, list, or tuple
		The distance or range of distances from the edge of each instance to include in the contour.
		If a single value is provided, it represents the maximum distance. If a tuple or list is provided,
		it represents the minimum and maximum distances.

	Returns
	-------
	border_label : ndarray
		An instance mask containing the contour of the segmented objects.

	Notes
	-----
	This function generates an instance mask representing the contour of the segmented instances in the label image.
	It use the distance_transform_edt function from the scipy.ndimage module to compute the Euclidean distance transform.
	The contour is defined based on the specified distance(s) from the edge of each instance.
	The resulting mask, `border_label`, contains the contour regions, while the interior regions are set to zero.

	Examples
	--------
	>>> border_label = contour_of_instance_segmentation(label, distance=3)
	# Generate a binary mask containing the contour of the segmented instances with a maximum distance of 3 pixels.
	
	"""


	edt = distance_transform_edt(label)
	
	if isinstance(distance, list) or isinstance(distance, tuple):
		min_distance = distance[0]; max_distance = distance[1]
	elif isinstance(distance, int) or isinstance(distance, float):
		min_distance = 0.; max_distance = distance

	thresholded = (edt <= max_distance)*(edt>min_distance)
	border_label = np.copy(label)
	border_label[np.where(thresholded==0)] = 0

	return border_label

def drop_tonal_features(features):

	for f in features:
		if 'intensity' in f:
			features.remove(f)
	return features

def measure_features(img, label, features=['area', 'intensity_mean'], channels=None,
	border_dist=None, haralick_options=None, verbose=True):

	"""

	Measure features within segmented regions of an image.

	Parameters
	----------
	img : ndarray
		The input image as a NumPy array.
	label : ndarray
		The segmentation labels corresponding to the image regions.
	features : list, optional
		The list of features to measure within the segmented regions. The default is ['area', 'intensity_mean'].
	channels : list, optional
		The list of channel names in the image. The default is ["brightfield_channel", "dead_nuclei_channel", "live_nuclei_channel"].
	border_dist : int, float, or list, optional
		The distance(s) in pixels from the edge of each segmented region to measure features. The default is None.
	haralick_options : dict, optional
		The options for computing Haralick features. The default is None.

	Returns
	-------
	df_props : DataFrame
		A pandas DataFrame containing the measured features for each segmented region.

	Notes
	-----
	This function measures features within segmented regions of an image.
	It utilizes the regionprops_table function from the skimage.measure module for feature extraction.
	The features to measure can be specified using the 'features' parameter.
	Optional parameters such as 'channels' and 'border_dist' allow for additional measurements.
	If provided, Haralick features can be computed using the 'haralick_options' parameter.
	The results are returned as a pandas DataFrame.

	Examples
	--------
	>>> df_props = measure_features(img, label, features=['area', 'intensity_mean'], channels=["brightfield_channel", "dead_nuclei_channel", "live_nuclei_channel"])
	# Measure area and mean intensity within segmented regions of the image.
	
	"""


	if features is None:
		features = []

	# Add label to have identity of mask
	if 'label' not in features:
		features.append('label')

	if img is None:
		if verbose:
			print('No image was provided... Skip intensity measurements.')
		border_dist = None; haralick_options = None;
		features = drop_tonal_features(features)

	if img is not None:
		if img.ndim==2:
			img = img[:,:,np.newaxis]
		if channels is None:
			channels = [f'intensity-{k}' for k in range(img.shape[-1])]
		if (channels is not None)*(img.ndim==3):
			assert len(channels)==img.shape[-1],"Mismatch between the provided channel names and the shape of the image"

	props = regionprops_table(label, intensity_image=img, properties=features)
	df_props = pd.DataFrame(props)

	if border_dist is not None:

		# automatically drop all non intensity features
		intensity_features_test = ['intensity' in s for s in features]
		intensity_features = list(np.array(features)[np.array(intensity_features_test)])
		# If no intensity feature was passed still measure mean intensity
		if len(intensity_features)==0:
			if verbose:
				print('No intensity feature was passed... Adding mean intensity for edge measurement...')
			intensity_features = np.append(intensity_features, 'intensity_mean')
		intensity_features = np.append(intensity_features, 'label')

		if (isinstance(border_dist, int) or isinstance(border_dist, float)):
			
			border_label = contour_of_instance_segmentation(label, border_dist)
			props_border = regionprops_table(border_label, intensity_image=img, properties=intensity_features)
			df_props_border = pd.DataFrame(props_border)
			for c in df_props_border.columns:
				if 'intensity' in c:
					df_props_border = df_props_border.rename({c: c+f'_edge_{border_dist}px'},axis=1)

		if isinstance(border_dist, list):
			df_props_border_list = []
			for d in border_dist:
				border_label = contour_of_instance_segmentation(label, d)
				props_border = regionprops_table(border_label, intensity_image=img, properties=intensity_features)
				df_props_border_d = pd.DataFrame(props_border)
				for c in df_props_border_d.columns:
					if 'intensity' in c:
						df_props_border_d = df_props_border_d.rename({c: c+f'_edge_{d}px'},axis=1)
				df_props_border_list.append(df_props_border_d)

			df_props_border = reduce(lambda  left,right: pd.merge(left,right,on=['label'],
											how='outer'), df_props_border_list)

		df_props = df_props.merge(df_props_border, how='outer', on='label')

	if haralick_options is not None:
		try:
			df_haralick = compute_haralick_features(img, label, channels=channels, **haralick_options)
			df_props = df_props.merge(df_haralick, left_on='label',right_on='cell_id')
			#df_props = df_props.drop(columns=['cell_label'])
		except Exception as e:
			print(e)
			pass

	if channels is not None:
		df_props = rename_intensity_column(df_props, channels)
	df_props.rename(columns={"label": "class_id"},inplace=True)
	df_props['class_id'] = df_props['class_id'].astype(float)
	df_props.to_csv('test.csv',index=False)

	return df_props

def compute_haralick_features(img, labels, channels=None, target_channel=0, scale_factor=1, percentiles=(0.01,99.99), clip_values=None,
								n_intensity_bins=256, ignore_zero=True, return_mean=True, return_mean_ptp=False, distance=1, disable_progress_bar=False, return_norm_image_only=False, return_digit_image_only=False):

	"""

	Compute Haralick texture features on each segmented region of an image.

	Parameters
	----------
	img : ndarray
		The input image as a NumPy array.
	labels : ndarray
		The segmentation labels corresponding to the image regions.
	target_channel : int, optional
		The target channel index of the image. The default is 0.
	modality : str, optional
		The modality or channel type of the image. The default is 'brightfield_channel'.
	scale_factor : float, optional
		The scale factor for resampling the image and labels. The default is 1.
	percentiles : tuple of float, optional
		The percentiles to use for image normalization. The default is (0.01, 99.99).
	clip_values : tuple of float, optional
		The minimum and maximum values to clip the image. If None, percentiles are used. The default is None.
	n_intensity_bins : int, optional
		The number of intensity bins for image normalization. The default is 255.
	ignore_zero : bool, optional
		Flag indicating whether to ignore zero values during feature computation. The default is True.
	return_mean : bool, optional
		Flag indicating whether to return the mean value of each Haralick feature. The default is True.
	return_mean_ptp : bool, optional
		Flag indicating whether to return the mean and peak-to-peak values of each Haralick feature. The default is False.
	distance : int, optional
		The distance parameter for Haralick feature computation. The default is 1.

	Returns
	-------
	features : DataFrame
		A pandas DataFrame containing the computed Haralick features for each segmented region.

	Notes
	-----
	This function computes Haralick features on an image within segmented regions.
	It uses the mahotas library for feature extraction and pandas DataFrame for storage.
	The image is rescaled, normalized and digitized based on the specified parameters.
	Haralick features are computed for each segmented region, and the results are returned as a DataFrame.

	Examples
	--------
	>>> features = compute_haralick_features(img, labels, target_channel=0, modality="brightfield_channel")
	# Compute Haralick features on the image within segmented regions.
	
	"""

	assert ((img.ndim==2)|(img.ndim==3)),f'Invalid image shape to compute the Haralick features. Expected YXC, got {img.shape}...'
	assert img.shape[:2]==labels.shape,f'Mismatch between image shape {img.shape} and labels shape {labels.shape}'

	if img.ndim==2:
		img = img[:,:,np.newaxis]
		target_channel = 0
		if isinstance(channels, list):
			modality = channels[0]
		elif isinstance(channels, str):
			modality = channels
		else:
			print('Channel name unrecognized...')
			modality=''
	elif img.ndim==3:
		assert target_channel is not None,"The image is multichannel. Please provide a target channel to compute the Haralick features. Abort."
		modality = channels[target_channel]

	haralick_labels = ["angular_second_moment",
					   "contrast",
					   "correlation",
					   "sum_of_square_variance",
					   "inverse_difference_moment",
					   "sum_average",
					   "sum_variance",
					   "sum_entropy",
					   "entropy",
					   "difference_variance",
					   "difference_entropy",
					   "information_measure_of_correlation_1",
					   "information_measure_of_correlation_2",
					   "maximal_correlation_coefficient"]

	haralick_labels = ['haralick_'+h+"_"+modality for h in haralick_labels]
	if len(img.shape)==3:
		img = img[:,:,target_channel]

	# Rescale image and mask
	img = zoom(img,[scale_factor,scale_factor],order=3).astype(float)
	labels = zoom(labels, [scale_factor,scale_factor],order=0)

	# Normalize image
	if clip_values is None:
		min_value = np.nanpercentile(img[img!=0.].flatten(), percentiles[0])
		max_value = np.nanpercentile(img[img!=0.].flatten(), percentiles[1])
	else:
		min_value = clip_values[0]; max_value = clip_values[1]

	img -= min_value
	img /= (max_value-min_value) / n_intensity_bins
	img[img<=0.] = 0.
	img[img>=n_intensity_bins] = n_intensity_bins

	if return_norm_image_only:
		return img

	hist,bins = np.histogram(img.flatten(),bins=n_intensity_bins)
	centered_bins = [bins[0]] + [bins[i] + (bins[i+1] - bins[i])/2. for i in range(len(bins)-1)]

	digitized = np.digitize(img, bins)
	img_binned = np.zeros_like(img)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			img_binned[i,j] = centered_bins[digitized[i,j] - 1]

	img = img_binned.astype(int)
	if return_digit_image_only:
		return img

	haralick_properties = []

	for cell in tqdm(np.unique(labels)[1:],disable=disable_progress_bar):

		mask = labels==cell
		f = img*mask
		features = haralick(f, ignore_zeros=ignore_zero,return_mean=return_mean,distance=distance)

		dictionary = {'cell_id': cell}
		for k in range(len(features)):
			dictionary.update({haralick_labels[k]: features[k]})
		haralick_properties.append(dictionary)

	assert len(haralick_properties)==(len(np.unique(labels))-1),'Some cells have not been measured...'

	return pd.DataFrame(haralick_properties)


def measure_isotropic_intensity(positions, # Dataframe of cell positions @ t
								 img,  # multichannel frame (YXC) @ t
								 channels=None, #channels, need labels to name measurements
								 intensity_measurement_radii=None, #list of radii, single value is circle, tuple is ring?
								 operations = ['mean'],
								 measurement_kernel = None,
								 pbar=None,
								 column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'},
								 verbose=True,
								):
	
	"""
	
	Measure isotropic intensity values around cell positions in an image.

	Parameters
	----------
	positions : pandas DataFrame
		DataFrame of cell positions at time 't' containing columns specified in `column_labels`.
	img : numpy array
		Multichannel frame (YXC) at time 't' used for intensity measurement.
	channels : list or str, optional
		List of channel names corresponding to the image channels. Default is None.
	intensity_measurement_radii : int, list, or tuple
		Radius or list of radii specifying the size of the isotropic measurement area.
		If a single value is provided, a circular measurement area is used. If a list or tuple of two values
		is provided, a ring-shaped measurement area is used. Default is None.
	operations : list, optional
		List of operations to perform on the intensity values. Default is ['mean'].
	measurement_kernel : numpy array, optional
		Kernel used for intensity measurement. If None, a circular or ring-shaped kernel is generated
		based on the provided `intensity_measurement_radii`. Default is None.
	pbar : tqdm progress bar, optional
		Progress bar for tracking the measurement process. Default is None.
	column_labels : dict, optional
		Dictionary containing the column labels for the DataFrame. Default is {'track': "TRACK_ID",
		'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.
	verbose : bool, optional
		If True, enables verbose output. Default is True.

	Returns
	-------
	pandas DataFrame
		The updated DataFrame `positions` with additional columns representing the measured intensity values.

	Notes
	-----
	This function measures the isotropic intensity values around the cell positions specified in the `positions`
	DataFrame using the provided image `img`. The intensity measurements are performed using circular or ring-shaped
	measurement areas defined by the `intensity_measurement_radii`. The measurements are calculated for each channel
	specified in the `channels` list. The resulting intensity values are stored in additional columns of the `positions`
	DataFrame. The `operations` parameter allows specifying different operations to be performed on the intensity
	values, such as 'mean', 'median', etc. The measurement kernel can be customized by providing the `measurement_kernel`
	parameter. If not provided, the measurement kernel is automatically generated based on the `intensity_measurement_radii`.
	The progress bar `pbar` can be used to track the measurement process. The `column_labels` dictionary is used to
	specify the column labels for the DataFrame.

	Examples
	--------
	>>> positions = pd.DataFrame({'TRACK_ID': [1, 2, 3], 'FRAME': [1, 1, 1],
	...                           'POSITION_X': [10, 20, 30], 'POSITION_Y': [15, 25, 35]})
	>>> img = np.random.rand(100, 100, 3)
	>>> channels = ['channel1', 'channel2', 'channel3']
	>>> intensity_measurement_radii = 5
	>>> positions = measure_isotropic_intensity(positions, img, channels=channels,
	...                                         intensity_measurement_radii=intensity_measurement_radii)
	# Measure isotropic intensity values around cell positions in the image.

	"""


	assert ((img.ndim==2)|(img.ndim==3)),f'Invalid image shape to compute the Haralick features. Expected YXC, got {img.shape}...'

	if img.ndim==2:
		img = img[:,:,np.newaxis]
		if isinstance(channels, str):
			channels = [channels]
		else:
			if verbose:
				print('Channel name unrecognized...')
			channels=['intensity']
	elif img.ndim==3:
		assert channels is not None,"The image is multichannel. Please provide the list of channel names. Abort."

	if isinstance(intensity_measurement_radii, int) or isinstance(intensity_measurement_radii, float):
		intensity_measurement_radii = [intensity_measurement_radii]

	if (measurement_kernel is None)*(intensity_measurement_radii is not None):

		for r in intensity_measurement_radii:

			if isinstance(r,list):
				mask = create_patch_mask(2*max(r)+1,2*max(r)+1,((2*max(r))//2,(2*max(r))//2),radius=r)
			else:
				mask = create_patch_mask(2*r+1,2*r+1,((2*r)//2,(2*r)//2),r)
			
			pad_value_x = mask.shape[0]//2 + 1
			pad_value_y = mask.shape[1]//2 + 1
			frame_padded = np.pad(img, [(pad_value_x,pad_value_x),(pad_value_y,pad_value_y),(0,0)])
			
			# Find a way to measure intensity in mask
			for tid,group in positions.groupby(column_labels['track']):

				x = group[column_labels['x']].to_numpy()[0]
				y = group[column_labels['y']].to_numpy()[0]

				xmin = int(x)
				xmax = int(x) + 2*pad_value_y - 1
				ymin = int(y)
				ymax = int(y) + 2*pad_value_x - 1

				assert frame_padded[ymin:ymax,xmin:xmax,0].shape == mask.shape,"Shape mismatch between the measurement kernel and the image..."
				
				expanded_mask = np.expand_dims(mask, axis=-1)  # shape: (X, Y, 1)
				crop = frame_padded[ymin:ymax,xmin:xmax]
				projection = np.multiply(crop, expanded_mask)

				for op in operations:
					func = eval('np.'+op)
					intensity_values = func(projection, axis=(0,1), where=projection!=0.)
					for k in range(crop.shape[-1]):
						if isinstance(r,list):
							positions.loc[group.index, f'{channels[k]}_ring_{min(r)}_{max(r)}_{op}'] = intensity_values[k]
						else:
							positions.loc[group.index, f'{channels[k]}_circle_{r}_{op}'] = intensity_values[k]

	elif (measurement_kernel is not None):
		# do something like this
		mask = measurement_kernel
		pad_value_x = mask.shape[0]//2 + 1
		pad_value_y = mask.shape[1]//2 + 1
		frame_padded = np.pad(img, [(pad_value_x,pad_value_x),(pad_value_y,pad_value_y),(0,0)])
		
		for tid,group in positions.groupby(column_labels['track']):

			x = group[column_labels['x']].to_numpy()[0]
			y = group[column_labels['y']].to_numpy()[0]

			xmin = int(x)
			xmax = int(x) + 2*pad_value_y - 1
			ymin = int(y)
			ymax = int(y) + 2*pad_value_x - 1

			assert frame_padded[ymin:ymax,xmin:xmax,0].shape == mask.shape,"Shape mismatch between the measurement kernel and the image..."
			
			expanded_mask = np.expand_dims(mask, axis=-1)  # shape: (X, Y, 1)
			crop = frame_padded[ymin:ymax,xmin:xmax]
			projection = np.multiply(crop, expanded_mask)

			for op in operations:
				func = eval('np.'+op)
				intensity_values = func(projection, axis=(0,1), where=projection!=0.)
				for k in range(crop.shape[-1]):
					positions.loc[group.index, f'{channels[k]}_custom_kernel_{op}'] = intensity_values[k]

	if pbar is not None:
		pbar.update(1)
	positions['class_id'] = positions['class_id'].astype(float)
	return positions

def measure_at_position(pos, mode, return_measurements=False):
	
	pos = pos.replace('\\','/')
	pos = pos.replace(' ','\\')
	assert os.path.exists(pos),f'Position {pos} is not a valid path.'
	if not pos.endswith('/'):
		pos += '/'
	subprocess.call(f"python {abs_path}/scripts/measure_cells.py --pos {pos} --mode {mode}", shell=True)
	
	return None