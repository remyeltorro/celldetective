"""
Copright © 2024 Laboratoire Adhesion et Inflammation, Authored by Remy Torro & Ksenija Dervanova.
"""

from tqdm import tqdm
import numpy as np
import os
from celldetective.io import get_config, get_experiment_wells, interpret_wells_and_positions, extract_well_name_and_number, get_positions_in_well, extract_position_name, get_position_movie_path, load_frames, auto_load_number_of_frames
from celldetective.utils import ConfigSectionMap, _extract_channel_indices_from_config, _extract_nbr_channels_from_config, _get_img_num_per_channel
from celldetective.filters import std_filter, gauss_filter
from stardist import fill_label_holes
from csbdeep.io import save_tiff_imagej_compatible
from gc import collect
from lmfit import Parameters, Model, models
import matplotlib.pyplot as plt

def estimate_background_per_condition(experiment, threshold_on_std=1, well_option='*', target_channel="channel_name", frame_range=[0,5], mode="timeseries", show_progress_per_pos=False, show_progress_per_well=True):
	
	"""
	Estimate the background per condition in an experiment.

	This function calculates the background of each well in the given experiment
	by analyzing frames from the specified range. It supports two modes: "timeseries"
	and "tiles". The function applies Gaussian and standard deviation filters to
	identify and mask out high-variance areas, and computes the median background
	across positions within each well.

	Parameters
	----------
	experiment : object
		The experiment object containing well and position information.
	threshold_on_std : float, optional
		The threshold for the standard deviation filter to identify high-variance areas. Defaults to 1.
	well_option : str, int, or list of int, optional
		The well selection option:
		- '*' : Select all wells.
		- int : Select a specific well by its index.
		- list of int : Select multiple wells by their indices. Defaults to '*'.
	target_channel : str
		The specific channel to be analyzed.
	frame_range : list of int, optional
		The range of frames to be analyzed, specified as [start, end]. Defaults to [0, 5].
	mode : {'timeseries', 'tiles'}, optional
		The mode of analysis. "timeseries" averages frames before filtering, while "tiles" filters each frame individually. Defaults to "timeseries".
	show_progress_per_pos : bool, optional
		If True, display a progress bar for position processing. Defaults to False.
	show_progress_per_well : bool, optional
		If True, display a progress bar for well processing. Defaults to True.

	Returns
	-------
	backgrounds : list of dict
		A list of dictionaries, each containing:
		- 'bg' : numpy.ndarray
			The computed background for the well.
		- 'well' : str
			The path to the well.

	Examples
	--------
	>>> experiment = ...  # Some experiment object
	>>> backgrounds = estimate_background_per_condition(experiment, threshold_on_std=1.5, well_option=[0, 1, 2], target_channel='DAPI', frame_range=[0, 10], mode="tiles")
	>>> print(backgrounds[0]['bg'])  # The background array for the first well
	>>> print(backgrounds[0]['well'])  # The path to the first well
	"""


	config = get_config(experiment)
	wells = get_experiment_wells(experiment)
	len_movie = float(ConfigSectionMap(config,"MovieSettings")["len_movie"])
	movie_prefix = ConfigSectionMap(config,"MovieSettings")["movie_prefix"]	

	well_indices, position_indices = interpret_wells_and_positions(experiment, well_option, "*")

	channel_indices = _extract_channel_indices_from_config(config, [target_channel])
	nbr_channels = _extract_nbr_channels_from_config(config)
	img_num_channels = _get_img_num_per_channel(channel_indices, int(len_movie), nbr_channels)
	
	backgrounds = []

	for k, well_path in enumerate(tqdm(wells[well_indices], disable=not show_progress_per_well)):
		
		well_name, _ = extract_well_name_and_number(well_path)
		well_idx = well_indices[k]
		
		positions = get_positions_in_well(well_path)
		print(f"Reconstruct a background in well {well_name} from positions: {[extract_position_name(p) for p in positions]}...")

		frame_mean_per_position = []

		for l,pos_path in enumerate(tqdm(positions, disable=not show_progress_per_pos)):
			
			stack_path = get_position_movie_path(pos_path, prefix=movie_prefix)

			if mode=="timeseries":

				frames = load_frames(img_num_channels[0,frame_range[0]:frame_range[1]], stack_path, normalize_input=False)
				frames = np.moveaxis(frames, -1, 0).astype(float)

				for i in range(len(frames)):
					if np.all(frames[i].flatten()==0):
						frames[i,:,:] = np.nan

				frame_mean = np.nanmean(frames, axis=0)
				
				frame = frame_mean.copy().astype(float)
				frame = gauss_filter(frame, 2)
				std_frame = std_filter(frame, 4)
				
				mask = std_frame > threshold_on_std
				mask = fill_label_holes(mask)
				frame[np.where(mask==1)] = np.nan

			elif mode=="tiles":

				frames = load_frames(img_num_channels[0,:], stack_path, normalize_input=False).astype(float)
				frames = np.moveaxis(frames, -1, 0).astype(float)

				for i in range(len(frames)):

					if np.all(frames[i].flatten()==0):
						frames[i,:,:] = np.nan
						continue

					f = frames[i].copy()
					f = gauss_filter(f, 2)
					std_frame = std_filter(f, 4)

					mask = std_frame > threshold_on_std
					mask = fill_label_holes(mask)
					f[np.where(mask==1)] = np.nan

					frames[i,:,:] = f
				
				frame = np.nanmedian(frames, axis=0)

			# store
			frame_mean_per_position.append(frame)

		background = np.nanmedian(frame_mean_per_position,axis=0)
		backgrounds.append({"bg": background, "well": well_path})
		print(f"Background successfully computed for well {well_name}...")

	return backgrounds

def correct_background_model_free(
					   experiment, 
					   well_option='*',
					   position_option='*',
					   target_channel="channel_name",
					   mode = "timeseries",
					   threshold_on_std = 1,
					   frame_range = [0,5],
					   optimize_option = False,
					   opt_coef_range = [0.95,1.05],
					   opt_coef_nbr = 100,
					   operation = 'divide',
					   clip = False,
					   show_progress_per_well = True,
					   show_progress_per_pos = False,
					   export = False,
					   return_stacks = False,
					   movie_prefix=None,
					   export_prefix='Corrected',
					   **kwargs,
					   ):

	"""
	Correct the background of image stacks for a given experiment.

	This function processes image stacks by estimating and correcting the background 
	for each well and position in the experiment. It supports different modes, such 
	as timeseries or tiles, and offers options for optimization and exporting the results.

	Parameters
	----------
	experiment : str
		Path to the experiment configuration.
	well_option : str, int, or list of int, optional
		Selection of wells to process. '*' indicates all wells. Defaults to '*'.
	position_option : str, int, or list of int, optional
		Selection of positions to process within each well. '*' indicates all positions. Defaults to '*'.
	target_channel : str, optional
		The name of the target channel to be corrected. Defaults to "channel_name".
	mode : {'timeseries', 'tiles'}, optional
		The mode of processing. Defaults to "timeseries".
	threshold_on_std : float, optional
		The threshold for the standard deviation filter to identify high-variance areas. Defaults to 1.
	frame_range : list of int, optional
		The range of frames to consider for background estimation. Defaults to [0, 5].
	optimize_option : bool, optional
		If True, optimize the correction coefficient. Defaults to False.
	opt_coef_range : list of float, optional
		The range of coefficients to try for optimization. Defaults to [0.95, 1.05].
	opt_coef_nbr : int, optional
		The number of coefficients to test within the optimization range. Defaults to 100.
	operation : {'divide', 'subtract'}, optional
		The operation to apply for background correction. Defaults to 'divide'.
	clip : bool, optional
		If True, clip the corrected values to be non-negative when using subtraction. Defaults to False.
	show_progress_per_well : bool, optional
		If True, show progress bar for each well. Defaults to True.
	show_progress_per_pos : bool, optional
		If True, show progress bar for each position. Defaults to False.
	export : bool, optional
		If True, export the corrected stacks to files. Defaults to False.
	return_stacks : bool, optional
		If True, return the corrected stacks as a list of numpy arrays. Defaults to False.

	Returns
	-------
	list of numpy.ndarray, optional
		A list of corrected image stacks if `return_stacks` is True.

	Notes
	-----
	The function uses several helper functions, including `interpret_wells_and_positions`, 
	`estimate_background_per_condition`, and `apply_background_to_stack`.

	Examples
	--------
	>>> experiment = "path/to/experiment/config"
	>>> corrected_stacks = correct_background(experiment, well_option=[0, 1], position_option='*', target_channel="DAPI", mode="timeseries", threshold_on_std=2, frame_range=[0, 10], optimize_option=True, operation='subtract', clip=True, return_stacks=True)
	>>> print(len(corrected_stacks))
	2

	"""

	config = get_config(experiment)
	wells = get_experiment_wells(experiment)
	len_movie = float(ConfigSectionMap(config,"MovieSettings")["len_movie"])
	if movie_prefix is None:
		movie_prefix = ConfigSectionMap(config,"MovieSettings")["movie_prefix"]	

	well_indices, position_indices = interpret_wells_and_positions(experiment, well_option, position_option)
	channel_indices = _extract_channel_indices_from_config(config, [target_channel])
	nbr_channels = _extract_nbr_channels_from_config(config)
	img_num_channels = _get_img_num_per_channel(channel_indices, int(len_movie), nbr_channels)
	
	stacks = []

	for k, well_path in enumerate(tqdm(wells[well_indices], disable=not show_progress_per_well)):
		
		well_name, _ = extract_well_name_and_number(well_path)

		try:
			background = estimate_background_per_condition(experiment, threshold_on_std=threshold_on_std, well_option=int(well_indices[k]), target_channel=target_channel, frame_range=frame_range, mode=mode, show_progress_per_pos=True, show_progress_per_well=False)
			background = background[0]
			background = background['bg']
		except Exception as e:
			print(f'Background could not be estimated due to error "{e}"... Skipping well {well_name}...')
			continue

		positions = get_positions_in_well(well_path)
		selection = positions[position_indices]
		if isinstance(selection[0],np.ndarray):
			selection = selection[0]

		for pidx,pos_path in enumerate(tqdm(selection, disable=not show_progress_per_pos)):
			
			stack_path = get_position_movie_path(pos_path, prefix=movie_prefix)
			print(f'Applying the correction to position {extract_position_name(pos_path)}...')

			corrected_stack = apply_background_to_stack(stack_path, 
														background,
														target_channel_index=channel_indices[0],
														nbr_channels=nbr_channels,
														stack_length=len_movie,
														threshold_on_std=threshold_on_std,
														optimize_option=optimize_option,
														opt_coef_range=opt_coef_range,
														opt_coef_nbr=opt_coef_nbr,
														operation=operation,
														clip=clip,
														export=export,
														prefix=export_prefix,
													  )
			print('Correction successful.')
			if return_stacks:
				stacks.append(corrected_stack)
			else:
				del corrected_stack
			collect()

	if return_stacks:
		return stacks



def apply_background_to_stack(stack_path, background, target_channel_index=0, nbr_channels=1, stack_length=45, threshold_on_std=1, optimize_option=True, opt_coef_range=(0.95,1.05), opt_coef_nbr=100, operation='divide', clip=False, export=False, prefix="Corrected"):

	"""
	Apply background correction to an image stack.

	This function corrects the background of an image stack by applying a specified operation 
	(either division or subtraction) between the image stack and the background. It also supports
	optimization of the correction coefficient through brute-force regression.

	Parameters
	----------
	stack_path : str
		The path to the image stack file.
	background : numpy.ndarray
		The background image to be applied for correction.
	target_channel_index : int, optional
		The index of the target channel to be corrected. Defaults to 0.
	nbr_channels : int, optional
		The number of channels in the image stack. Defaults to 1.
	stack_length : int, optional
		The length of the image stack (number of frames). If None, the length is auto-detected. Defaults to 45.
	threshold_on_std : float, optional
		The threshold for the standard deviation filter to identify high-variance areas. Defaults to 1.
	optimize_option : bool, optional
		If True, optimize the correction coefficient using a range of values. Defaults to True.
	opt_coef_range : tuple of float, optional
		The range of coefficients to try for optimization. Defaults to (0.95, 1.05).
	opt_coef_nbr : int, optional
		The number of coefficients to test within the optimization range. Defaults to 100.
	operation : {'divide', 'subtract'}, optional
		The operation to apply for background correction. Defaults to 'divide'.
	clip : bool, optional
		If True, clip the corrected values to be non-negative when using subtraction. Defaults to False.
	export : bool, optional
		If True, export the corrected stack to a file. Defaults to False.
	prefix : str, optional
		The prefix for the exported file name. Defaults to "Corrected".

	Returns
	-------
	corrected_stack : numpy.ndarray
		The background-corrected image stack.

	Examples
	--------
	>>> stack_path = "path/to/stack.tif"
	>>> background = np.zeros((512, 512))  # Example background
	>>> corrected_stack = apply_background_to_stack(stack_path, background, target_channel_index=0, nbr_channels=3, stack_length=45, optimize_option=False, operation='subtract', clip=True)
	>>> print(corrected_stack.shape)
	(44, 512, 512, 3)

	"""

	if stack_length is None:
		stack_length = auto_load_number_of_frames(stack_path)
		if stack_length is None:
			print('stack length not provided')
			return None

	if optimize_option:
		coefficients = np.linspace(opt_coef_range[0], opt_coef_range[1], int(opt_coef_nbr))
		coefficients = np.append(coefficients, [1.0])
	if export:
		path,file = os.path.split(stack_path)
		if prefix is None:
			newfile = file
		else:
			newfile = '_'.join([prefix,file])

	corrected_stack = []

	for i in range(0,int(stack_length*nbr_channels) - nbr_channels,nbr_channels):
		
		frames = load_frames(list(np.arange(i,(i+nbr_channels))), stack_path, normalize_input=False).astype(float)
		target_img = frames[:,:,target_channel_index].copy()

		if optimize_option:
			
			target_copy = target_img.copy()
			f = gauss_filter(target_img.copy(), 2)
			std_frame = std_filter(f, 4)
			mask = std_frame > threshold_on_std
			mask = fill_label_holes(mask)
			target_copy[np.where(mask==1)] = np.nan
			loss = []
			
			# brute-force regression, could do gradient descent instead
			for c in coefficients:
				diff = np.subtract(target_copy, c*background, where=target_copy==target_copy)
				s = np.sum(np.abs(diff, where=diff==diff), where=diff==diff)
				loss.append(s)
			c = coefficients[np.argmin(loss)]
			print(f"Frame: {i}; optimal coefficient: {c}...")
			# if c==min(coefficients) or c==max(coefficients):
			# 	print('Warning... The optimal coefficient is beyond the range provided... Please adjust your coefficient range...')	
		else:
			c=1

		if operation=="divide":
			correction = np.divide(target_img, background*c, where=background==background)
			correction[background!=background] = np.nan
			correction[target_img!=target_img] = np.nan
			fill_val = 1.0

		elif operation=="subtract":
			correction = np.subtract(target_img, background*c, where=background==background)
			correction[background!=background] = np.nan
			correction[target_img!=target_img] = np.nan
			fill_val = 0.0
			if clip:
				correction[correction<=0.] = 0.

		frames[:,:,target_channel_index] = correction
		corrected_stack.append(frames)

	corrected_stack = np.array(corrected_stack)

	if export:
		save_tiff_imagej_compatible(os.sep.join([path,newfile]), corrected_stack, axes='TYXC')

	return corrected_stack

def paraboloid(x, y, a, b, c, d, e, g):
	return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + g


def plane(x, y, a, b, c):
	return a * x + b * y + c


def fit_plane(image, cell_masks=None):
	"""
	Fit a plane to the given image data.

	Parameters:
	- image (numpy.ndarray): The input image data.
	- cell_masks (numpy.ndarray, optional): An array specifying cell masks. If provided, areas covered by
											cell masks will be excluded from the fitting process.

	Returns:
	- numpy.ndarray: The fitted plane.

	This function fits a plane to the given image data using least squares regression. It constructs a mesh
	grid based on the dimensions of the image and fits a plane model to the data points. If cell masks are
	provided, areas covered by cell masks will be excluded from the fitting process.

	Example:
	>>> image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	>>> result = fit_plane(image)
	>>> print(result)
	[[1. 2. 3.]
	 [4. 5. 6.]
	 [7. 8. 9.]]

	Note:
	- The 'cell_masks' parameter allows excluding areas covered by cell masks from the fitting process.
	"""
	data = np.empty(image.shape)
	x = np.arange(0, image.shape[1])
	y = np.arange(0, image.shape[0])
	xx, yy = np.meshgrid(x, y)

	params = Parameters()
	params.add('a', value=1)
	params.add('b', value=1)
	params.add('c', value=1)

	model = Model(plane, independent_vars=['x', 'y'])

	weights = np.ones_like(xx, dtype=float)
	weights[np.where(cell_masks > 0)] = 0.

	result = model.fit(image,
					   x=xx,
					   y=yy,
					   weights=weights,
					   params=params, max_nfev=3000)
	del model
	collect()

	return result.best_fit


def fit_paraboloid(image, cell_masks=None):
	"""
	Fit a paraboloid to the given image data.

	Parameters:
	- image (numpy.ndarray): The input image data.
	- cell_masks (numpy.ndarray, optional): An array specifying cell masks. If provided, areas covered by
											cell masks will be excluded from the fitting process.

	Returns:
	- numpy.ndarray: The fitted paraboloid.

	This function fits a paraboloid to the given image data using least squares regression. It constructs
	a mesh grid based on the dimensions of the image and fits a paraboloid model to the data points. If cell
	masks are provided, areas covered by cell masks will be excluded from the fitting process.

	Example:
	>>> image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	>>> result = fit_paraboloid(image)
	>>> print(result)
	[[1. 2. 3.]
	 [4. 5. 6.]
	 [7. 8. 9.]]

	Note:
	- The 'cell_masks' parameter allows excluding areas covered by cell masks from the fitting process.
	"""
	data = np.empty(image.shape)
	x = np.arange(0, image.shape[1])
	y = np.arange(0, image.shape[0])
	xx, yy = np.meshgrid(x, y)

	params = Parameters()
	params.add('a', value=1.0E-05)
	params.add('b', value=1.0E-05)
	params.add('c', value=1.0E-06)
	params.add('d', value=0.01)
	params.add('e', value=0.01)
	params.add('g', value=100)

	model = Model(paraboloid, independent_vars=['x', 'y'])

	weights = np.ones_like(xx, dtype=float)
	weights[np.where(cell_masks > 0)] = 0.

	result = model.fit(image,
					   x=xx,
					   y=yy,
					   weights=weights,
					   params=params, max_nfev=3000)

	#print(result.fit_report())
	del model
	collect()
	return result.best_fit


def correct_background_model(
						   experiment, 
						   well_option='*',
						   position_option='*',
						   target_channel="channel_name",
						   threshold_on_std = 1,
						   model = 'paraboloid',
						   operation = 'divide',
						   clip = False,
						   show_progress_per_well = True,
						   show_progress_per_pos = False,
						   export = False,
						   return_stacks = False,
						   movie_prefix=None,
						   export_prefix='Corrected',
						   **kwargs,
						   ):

	config = get_config(experiment)
	wells = get_experiment_wells(experiment)
	len_movie = float(ConfigSectionMap(config,"MovieSettings")["len_movie"])
	if movie_prefix is None:
		movie_prefix = ConfigSectionMap(config,"MovieSettings")["movie_prefix"]	

	well_indices, position_indices = interpret_wells_and_positions(experiment, well_option, position_option)
	channel_indices = _extract_channel_indices_from_config(config, [target_channel])
	nbr_channels = _extract_nbr_channels_from_config(config)
	img_num_channels = _get_img_num_per_channel(channel_indices, int(len_movie), nbr_channels)
	
	stacks = []

	for k, well_path in enumerate(tqdm(wells[well_indices], disable=not show_progress_per_well)):
		
		well_name, _ = extract_well_name_and_number(well_path)
		positions = get_positions_in_well(well_path)
		selection = positions[position_indices]
		if isinstance(selection[0],np.ndarray):
			selection = selection[0]

		for pidx,pos_path in enumerate(tqdm(selection, disable=not show_progress_per_pos)):
			
			stack_path = get_position_movie_path(pos_path, prefix=movie_prefix)
			print(f'Applying the correction to position {extract_position_name(pos_path)}...')
			print(stack_path)

			corrected_stack = fit_and_apply_model_background_to_stack(stack_path, 
														target_channel_index=channel_indices[0],
														model = model,
														nbr_channels=nbr_channels,
														stack_length=len_movie,
														threshold_on_std=threshold_on_std,
														operation=operation,
														clip=clip,
														export=export,
														prefix=export_prefix,
													  )
			print('Correction successful.')
			if return_stacks:
				stacks.append(corrected_stack)
			else:
				del corrected_stack
			collect()

	if return_stacks:
		return stacks

def fit_and_apply_model_background_to_stack(stack_path,
											target_channel_index=0,
											nbr_channels=1,
											stack_length=45,
											threshold_on_std=1, 
											operation='divide',
											model='paraboloid',
											clip=False, 
											export=False,
											prefix="Corrected"
											):

	stack_length_auto = auto_load_number_of_frames(stack_path)
	if stack_length_auto is None and stack_length is None:
		print('stack length not provided')
		return None
	if stack_length_auto is not None:
		stack_length = stack_length_auto

	if export:
		path,file = os.path.split(stack_path)
		if prefix is None:
			newfile = file
		else:
			newfile = '_'.join([prefix,file])

	corrected_stack = []

	for i in tqdm(range(0,int(stack_length*nbr_channels) - nbr_channels,nbr_channels)):
		
		frames = load_frames(list(np.arange(i,(i+nbr_channels))), stack_path, normalize_input=False).astype(float)
		target_img = frames[:,:,target_channel_index].copy()

		target_copy = target_img.copy()
		f = gauss_filter(target_img.copy(), 2)
		std_frame = std_filter(f, 4)
		masks = std_frame > threshold_on_std
		masks = fill_label_holes(masks).astype(int)

		background = fit_background_model(target_img, cell_masks=masks, model=model)	

		if operation=="divide":
			correction = np.divide(target_img, background, where=background==background)
			correction[background!=background] = np.nan
			correction[target_img!=target_img] = np.nan
			fill_val = 1.0

		elif operation=="subtract":
			correction = np.subtract(target_img, background, where=background==background)
			correction[background!=background] = np.nan
			correction[target_img!=target_img] = np.nan
			fill_val = 0.0
			if clip:
				correction[correction<=0.] = 0.

		frames[:,:,target_channel_index] = correction.copy()
		corrected_stack.append(frames)

		del frames
		del correction
		del background
		del masks
		del std_frame
		del target_img
		del target_copy
		del f
		collect()

	corrected_stack = np.array(corrected_stack)

	if export:
		save_tiff_imagej_compatible(os.sep.join([path,newfile]), corrected_stack, axes='TYXC')

	return corrected_stack

def fit_background_model(img, cell_masks=None, model='paraboloid'):
	
	if model == "paraboloid":
		bg = fit_paraboloid(img.astype(float), cell_masks=cell_masks).astype(float)
	elif model == "plane":
		bg = fit_plane(img.astype(float), cell_masks=cell_masks).astype(float)

	if bg is not None:
		bg = np.array(bg)

	return bg