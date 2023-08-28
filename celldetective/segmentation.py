"""
Segmentation module
"""
import json
import os
from .io import locate_segmentation_model, get_stack_normalization_values, normalize_multichannel
from .utils import _estimate_scale_factor, _extract_channel_indices
from pathlib import Path
from tqdm import tqdm
import numpy as np
from stardist.models import StarDist2D
from cellpose.models import CellposeModel
from skimage.transform import resize
from celldetective.io import _view_on_napari, locate_labels, locate_stack, _view_on_napari
from celldetective.filters import * #rework this to give a name
from celldetective.utils import rename_intensity_column
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops_table
from skimage.exposure import match_histograms
import pandas as pd
import subprocess


abs_path = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],'celldetective'])

def segment(stack, model_name, channels=None, spatial_calibration=None, view_on_napari=False,
			use_gpu=True, time_flat_normalization=False, time_flat_percentiles=(0.0,99.99)):
	
	"""
	
	Segment objects in a stack using a pre-trained segmentation model.

	Parameters
	----------
	stack : ndarray
		The input stack to be segmented, with shape (frames, height, width, channels).
	model_name : str
		The name of the pre-trained segmentation model to use.
	channels : list or None, optional
		The names of the channels in the stack. If None, assumes the channels are indexed from 0 to `stack.shape[-1] - 1`.
		Default is None.
	spatial_calibration : float or None, optional
		The spatial calibration factor of the stack. If None, the calibration factor from the model configuration will be used.
		Default is None.
	view_on_napari : bool, optional
		Whether to visualize the segmentation results using Napari. Default is False.
	use_gpu : bool, optional
		Whether to use GPU acceleration if available. Default is True.
	time_flat_normalization : bool, optional
		Whether to perform time-flat normalization on the stack before segmentation. Default is False.
	time_flat_percentiles : tuple, optional
		The percentiles used for time-flat normalization. Default is (0.0, 99.99).

	Returns
	-------
	ndarray
		The segmented labels with shape (frames, height, width).

	Notes
	-----
	This function applies object segmentation to a stack of images using a pre-trained segmentation model. The stack is first
	preprocessed by normalizing the intensity values, rescaling the spatial dimensions, and applying the segmentation model.
	The resulting labels are returned as an ndarray with the same number of frames as the input stack.

	Examples
	--------
	>>> stack = np.random.rand(10, 256, 256, 3)
	>>> labels = segment(stack, 'model_name', channels=['channel_1', 'channel_2', 'channel_3'], spatial_calibration=0.5)

	"""

	model_path = locate_segmentation_model(model_name)
	input_config = model_path+'config_input.json'
	if os.path.exists(input_config):
		with open(input_config) as config:
			print("Loading input configuration from 'config_input.json'.")
			input_config = json.load(config)
	else:
		print('Model input configuration could not be located...')
		return None

	if not use_gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	else:
		os.environ['CUDA_VISIBLE_DEVICES'] = '0'		

	if channels is not None:
		assert len(channels)==stack.shape[-1],f'The channel names provided do not match with the expected number of channels in the stack: {stack.shape[-1]}.'

	required_channels = input_config['channels']
	channel_indices = _extract_channel_indices(channels, required_channels)

	required_spatial_calibration = input_config['spatial_calibration']
	model_type = input_config['model_type']

	if 'normalize' in input_config:
		normalize = input_config['normalize']
	else:
		normalize = True

	if model_type=='cellpose':
		diameter = input_config['diameter']
		if diameter!=30:
			required_spatial_calibration = None
		cellprob_threshold = input_config['cellprob_threshold']
		flow_threshold = input_config['flow_threshold']

	scale = _estimate_scale_factor(spatial_calibration, required_spatial_calibration)

	if model_type=='stardist':
		model = StarDist2D(None, name=model_name, basedir=Path(model_path).parent)
		print(f"StarDist model {model_name} successfully loaded")

	elif model_type=='cellpose':
		model = CellposeModel(gpu=use_gpu, pretrained_model=model_path+model_path.split('/')[-2], diam_mean=30.0)

	labels = []
	if (time_flat_normalization)*normalize:
		normalization_values = get_stack_normalization_values(stack[:,:,:,channel_indices], percentiles=time_flat_percentiles)
	else:
		normalization_values = [None]*len(channel_indices)

	for t in tqdm(range(len(stack)),desc="frame"):

		# normalize
		frame = stack[t,:,:,np.array(channel_indices)]
		if np.argmin(frame.shape)!=(frame.ndim-1):
			frame = np.moveaxis(frame,np.argmin(frame.shape),-1)
		if normalize:
			frame = normalize_multichannel(frame, values=normalization_values)

		if scale is not None:
			frame = ndi.zoom(frame, [scale,scale,1], order=3)		

		if model_type=="stardist":

			Y_pred, details = model.predict_instances(frame, n_tiles=model._guess_n_tiles(frame), show_tile_progress=False, verbose=False)
			Y_pred = Y_pred.astype(np.uint16)

		elif model_type=="cellpose":

			if stack.ndim==3:
				channels_cp = [[0,0]]
			else:
				channels_cp = [[0,1]]

			Y_pred, _, _ = model.eval([frame], diameter = diameter, flow_threshold=flow_threshold, channels=channels_cp, normalize=normalize)
			Y_pred = Y_pred[0].astype(np.uint16)

		if scale is not None:
			Y_pred = ndi.zoom(Y_pred, [1./scale,1./scale],order=0)


		if Y_pred.shape != stack[0].shape[:2]:
			Y_pred = resize(Y_pred, stack[0].shape, order=0)

		labels.append(Y_pred)

	labels = np.array(labels,dtype=int)

	if view_on_napari:
		_view_on_napari(tracks=None, stack=stack, labels=labels)

	return labels


def segment_from_thresholds(stack, target_channel=0, thresholds=None, view_on_napari=False, equalize_reference=None,
							filters=None, marker_min_distance=30, marker_footprint_size=20, marker_footprint=None, feature_queries=None):

	masks = []
	for t in tqdm(range(len(stack))):
		instance_seg = segment_frame_from_thresholds(stack[t], target_channel=target_channel, thresholds=thresholds, equalize_reference=equalize_reference,
													filters=filters, marker_min_distance=marker_min_distance, marker_footprint_size=marker_footprint_size,
													marker_footprint=marker_footprint, feature_queries=feature_queries)
		masks.append(instance_seg)

	masks = np.array(masks, dtype=np.int16)
	if view_on_napari:
		_view_on_napari(tracks=None, stack=stack, labels=masks)
	return masks

def segment_frame_from_thresholds(frame, target_channel=0, thresholds=None, equalize_reference=None,
								  filters=None, marker_min_distance=30, marker_footprint_size=20, marker_footprint=None, feature_queries=None, channel_names=None):
	
	img = frame[:,:,target_channel]
	if equalize_reference is not None:
		img = match_histograms(img, equalize_reference)
	img_mc = frame.copy()
	img = filter_image(img, filters=filters)
	binary_image = threshold_image(img, thresholds[0], thresholds[1])
	coords,distance = identify_markers_from_binary(binary_image, marker_min_distance, footprint_size=marker_footprint_size, footprint=marker_footprint, return_edt=True)
	instance_seg = apply_watershed(binary_image, coords, distance)
	instance_seg = filter_on_property(instance_seg, intensity_image=img_mc, queries=feature_queries, channel_names=channel_names)

	return instance_seg


def filter_on_property(labels, intensity_image=None, queries=None, channel_names=None):

	if queries is None:
		return labels
	else:
		if isinstance(queries, str):
			queries = [queries]

	props = ['label','area', 'area_bbox', 'area_convex', 'area_filled', 'axis_major_length',
						'axis_minor_length', 'eccentricity', 'equivalent_diameter_area', 
						'euler_number', 'feret_diameter_max', 'orientation', 'perimeter',
						'perimeter_crofton', 'solidity']

	intensity_props = ['intensity_mean', 'intensity_max', 'intensity_min']

	if intensity_image is not None:
		props.extend(intensity_props)

	properties = pd.DataFrame(regionprops_table(labels, intensity_image=intensity_image, properties=props))
	if channel_names is not None:
		properties = rename_intensity_column(properties, channel_names)
	for query in queries:
		try:
			properties = properties.query(f'not ({query})')
		except Exception as e:
			print(f'Query {query} could not be applied. Ensure that the feature exists. {e}')

	cell_ids = list(np.unique(labels)[1:])
	leftover_cells = list(properties['label'].unique())
	to_remove = [value for value in cell_ids if value not in leftover_cells]

	for c in to_remove:
		labels[np.where(labels==c)] = 0.

	return labels


def apply_watershed(binary_image, coords, distance):

	mask = np.zeros(binary_image.shape, dtype=bool)
	mask[tuple(coords.T)] = True
	markers, _ = ndi.label(mask)
	labels = watershed(-distance, markers, mask=binary_image)

	return labels

def identify_markers_from_binary(binary_image, min_distance, footprint_size=20, footprint=None, return_edt=False):

	"""

	Identify markers from a binary image using distance transform and peak detection.

	Parameters
	----------
	binary_image : ndarray
		The binary image from which to identify markers.
	min_distance : int
		The minimum distance between markers. Only the markers with a minimum distance greater than or equal to
		`min_distance` will be identified.
	footprint_size : int, optional
		The size of the footprint or structuring element used for peak detection. Default is 20.
	footprint : ndarray, optional
		The footprint or structuring element used for peak detection. If None, a square footprint of size
		`footprint_size` will be used. Default is None.
	return_edt : bool, optional
		Whether to return the Euclidean distance transform image along with the identified marker coordinates.
		If True, the function will return the marker coordinates and the distance transform image as a tuple.
		If False, only the marker coordinates will be returned. Default is False.

	Returns
	-------
	ndarray or tuple
		If `return_edt` is False, returns the identified marker coordinates as an ndarray of shape (N, 2), where N is
		the number of identified markers. If `return_edt` is True, returns a tuple containing the marker coordinates
		and the distance transform image.

	Notes
	-----
	This function uses the distance transform of the binary image to identify markers by detecting local maxima. The
	distance transform assigns each pixel a value representing the Euclidean distance to the nearest background pixel.
	By finding peaks in the distance transform, we can identify the markers in the original binary image. The `min_distance`
	parameter controls the minimum distance between markers to avoid clustering.

	"""
	
	distance = ndi.distance_transform_edt(binary_image.astype(float))
	if footprint is None:
		footprint = np.ones((footprint_size, footprint_size))
	coords = peak_local_max(distance, footprint=footprint,
		labels=binary_image.astype(int), min_distance=min_distance)
	if return_edt:
		return coords, distance
	else:
		return coords


def threshold_image(img, min_threshold, max_threshold, foreground_value=255., fill_holes=True):

	"""
	
	Threshold the input image to create a binary mask.

	Parameters
	----------
	img : ndarray
		The input image to be thresholded.
	min_threshold : float
		The minimum threshold value.
	max_threshold : float
		The maximum threshold value.
	foreground_value : float, optional
		The value assigned to foreground pixels in the binary mask. Default is 255.
	fill_holes : bool, optional
		Whether to fill holes in the binary mask. If True, the binary mask will be processed to fill any holes.
		If False, the binary mask will not be modified. Default is True.

	Returns
	-------
	ndarray
		The binary mask after thresholding.

	Notes
	-----
	This function applies a threshold to the input image to create a binary mask. Pixels with values within the specified
	threshold range are considered as foreground and assigned the `foreground_value`, while pixels outside the range are
	considered as background and assigned 0. If `fill_holes` is True, the binary mask will be processed to fill any holes
	using morphological operations.

	Examples
	--------
	>>> image = np.random.rand(256, 256)
	>>> binary_mask = threshold_image(image, 0.2, 0.8, foreground_value=1., fill_holes=True)

	"""


	binary = (img>=min_threshold)*(img<=max_threshold) * foreground_value
	if fill_holes:
		binary = ndi.binary_fill_holes(binary)
	return binary

def filter_image(img, filters=None):

	"""

	Apply one or more image filters to the input image.

	Parameters
	----------
	img : ndarray
		The input image to be filtered.
	filters : list or None, optional
		A list of filters to be applied to the image. Each filter is represented as a tuple or list with the first element being
		the filter function name (minus the '_filter' extension, as listed in software.filters) and the subsequent elements being 
		the arguments for that filter function. If None, the original image is returned without any filtering applied. Default is None.

	Returns
	-------
	ndarray
		The filtered image.

	Notes
	-----
	This function applies a series of image filters to the input image. The filters are specified as a list of tuples,
	where each tuple contains the name of the filter function and its corresponding arguments. The filters are applied
	sequentially to the image. If no filters are provided, the original image is returned unchanged.

	Examples
	--------
	>>> image = np.random.rand(256, 256)
	>>> filtered_image = filter_image(image, filters=[('gaussian', 3), ('median', 5)])

	"""

	if filters is None:
		return img

	if img.ndim==3:
		img = np.squeeze(img)

	for f in filters:
		func = eval(f[0]+'_filter')
		print(f, f[0], f[1:])
		img = func(img, *f[1:])	
	return img


def segment_at_position(pos, mode, model_name, stack_prefix=None, use_gpu=True, return_labels=False, view_on_napari=False):

	"""
	Perform image segmentation at the specified position using a pre-trained model.

	Parameters
	----------
	pos : str
		The path to the position directory containing the input images to be segmented.
	mode : str
		The segmentation mode. This determines the type of objects to be segmented ('target' or 'effector').
	model_name : str
		The name of the pre-trained segmentation model to be used.
	stack_prefix : str or None, optional
		The prefix of the stack file name. Defaults to None.
	use_gpu : bool, optional
		Whether to use the GPU for segmentation if available. Defaults to True.
	return_labels : bool, optional
		If True, the function returns the segmentation labels as an output. Defaults to False.
	view_on_napari : bool, optional
		If True, the segmented labels are displayed in a Napari viewer. Defaults to False.

	Returns
	-------
	numpy.ndarray or None
		If `return_labels` is True, the function returns the segmentation labels as a NumPy array. Otherwise, it returns None. The subprocess writes the
		segmentation labels in the position directory.

	Examples
	--------
	>>> labels = segment_at_position('ExperimentFolder/W1/100/', 'effector', 'mice_t_cell_RICM', return_labels=True)

	"""
	
	pos = pos.replace('\\','/')
	pos = pos.replace(' ','\\')
	assert os.path.exists(pos),f'Position {pos} is not a valid path.'
	
	script_path = os.sep.join([abs_path, 'scripts', 'segment_cells.py'])
	subprocess.call(rf"python {script_path} --pos {pos} --model {model_name} --mode {mode} --use_gpu {use_gpu}", shell=True)

	if return_labels or view_on_napari:
		labels = locate_labels(pos, population=mode)
	if view_on_napari:
		if stack_prefix is None:
			stack_prefix = ''
		stack = locate_stack(pos, prefix=stack_prefix)
		_view_on_napari(tracks=None, stack=stack, labels=labels)
	if return_labels:
		return labels
	else:
		return None

def segment_from_threshold_at_position(pos, mode, config):

	pos = pos.replace('\\','/')
	pos = pos.replace(' ','\\ ')
	assert os.path.exists(pos),f'Position {pos} is not a valid path.'

	config = config.replace('\\','/')
	config = config.replace(' ','\\')
	assert os.path.exists(config),f'Config {config} is not a valid path.'

	script_path = os.sep.join([abs_path, 'scripts', 'segment_cells_thresholds.py'])
	subprocess.call(rf"python {script_path} --pos {pos} --config {config} --mode {mode}", shell=True)


if __name__ == "__main__":
	print(segment(None,'test'))
