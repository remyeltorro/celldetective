from natsort import natsorted
from glob import glob
from tifffile import imread, TiffFile
import numpy as np
import os
import pandas as pd
import napari
import gc
from tqdm import tqdm
from csbdeep.utils import normalize_mi_ma
import skimage.io as skio
from scipy.ndimage import zoom
from btrack.datasets import cell_config
from magicgui import magicgui
from csbdeep.io import save_tiff_imagej_compatible
from pathlib import Path, PurePath
from celldetective.utils import ConfigSectionMap, extract_experiment_channels, _extract_labels_from_config
import json

def get_experiment_wells(experiment):
	
	if not experiment.endswith(os.sep):
		experiment += os.sep
		
	wells = natsorted(glob(experiment + "W*" + os.sep))
	return np.array(wells,dtype=str)

def get_config(experiment):

	if not experiment.endswith(os.sep):
		experiment += os.sep
	
	config = experiment + 'config.ini'
	config = rf"{config}"
	assert os.path.exists(config),'The experiment configuration could not be located...'
	return config	


def get_spatial_calibration(experiment):
	
	
	config = get_config(experiment)
	PxToUm = float(ConfigSectionMap(self.exp_config,"MovieSettings")["pxtoum"])
	
	return PxToUm

def get_temporal_calibration(experiment):
	
	config = get_config(experiment)
	FrameToMin = float(ConfigSectionMap(config,"MovieSettings")["frametomin"])
	
	return FrameToMin

def get_experiment_concentrations(experiment, dtype=str):
	
	
	config = get_config(experiment)
	wells = get_experiment_wells(experiment)
	nbr_of_wells = len(wells)
	
	concentrations = ConfigSectionMap(config,"Labels")["concentrations"].split(",")
	if nbr_of_wells != len(concentrations):
		concentrations = [str(s) for s in np.linspace(0,nbr_of_wells-1,nbr_of_wells)]
	
	return np.array([dtype(c) for c in concentrations])

def get_experiment_cell_types(experiment, dtype=str):
	
	config = get_config(experiment)
	wells = get_experiment_wells(experiment)
	nbr_of_wells = len(wells)
	
	cell_types = ConfigSectionMap(config,"Labels")["cell_types"].split(",")
	if nbr_of_wells != len(cell_types):
		cell_types = [str(s) for s in np.linspace(0,nbr_of_wells-1,nbr_of_wells)]
	
	return np.array([dtype(c) for c in cell_types])

def get_experiment_antibodies(experiment, dtype=str):
	
	config = get_config(experiment)
	wells = get_experiment_wells(experiment)
	nbr_of_wells = len(wells)
	
	antibodies = ConfigSectionMap(config,"Labels")["antibodies"].split(",")
	if nbr_of_wells != len(antibodies):
		antibodies = [str(s) for s in np.linspace(0,nbr_of_wells-1,nbr_of_wells)]
	
	return np.array([dtype(c) for c in antibodies])

def get_experiment_pharmaceutical_agents(experiment, dtype=str):
	
	config = get_config(experiment)
	wells = get_experiment_wells(experiment)
	nbr_of_wells = len(wells)
	
	pharmaceutical_agents = ConfigSectionMap(config,"Labels")["pharmaceutical_agents"].split(",")
	if nbr_of_wells != len(pharmaceutical_agents):
		pharmaceutical_agents = [str(s) for s in np.linspace(0,nbr_of_wells-1,nbr_of_wells)]
	
	return np.array([dtype(c) for c in pharmaceutical_agents])


def _interpret_wells_and_positions(experiment, well_option, position_option):
	
	wells = get_experiment_wells(experiment)
	nbr_of_wells = len(wells)    
	
	if well_option=='*':
		well_indices = np.arange(len(wells))
	elif isinstance(well_option, int):
		well_indices = np.array([well_option], dtype=int)
	elif isinstance(well_option, list):
		well_indices = well_option
		
	if position_option=='*':
		position_indices = None
	elif isinstance(position_option, int):
		position_indices = np.array([position_option], dtype=int)
	elif isinstance(position_option, list):
		position_indices = position_option
	
	return well_indices, position_indices
		
def extract_well_name_and_number(well):
	
	split_well_path = well.split(os.sep)
	split_well_path = list(filter(None, split_well_path))
	well_name = split_well_path[-1]
	well_number = int(split_well_path[-1].replace('W',''))  
	
	return well_name, well_number

def extract_position_name(pos):
	
	split_pos_path = pos.split(os.sep)
	split_pos_path = list(filter(None, split_pos_path))
	pos_name = split_pos_path[-1]
	
	return pos_name

def get_position_table(pos, population, return_path=False):
	
	if not pos.endswith(os.sep):
		table = os.sep.join([pos,'output','tables',f'trajectories_{population}.csv'])
	else:
		table = pos + os.sep.join(['output','tables',f'trajectories_{population}.csv'])		

	if os.path.exists(table):
		df_pos = pd.read_csv(table, low_memory=False)
	else:
		df_pos = None
	
	if return_path:
		return df_pos, table
	else:
		return df_pos

def get_position_movie_path(pos, prefix=''):

	movies = glob(pos+os.sep.join(['movie',prefix+'*.tif']))
	if len(movies)>0:
		stack_path = movies[0]
	else:
		stack_path = np.nan
	
	return stack_path
					
def load_experiment_tables(experiment, population='targets', well_option='*',position_option='*', return_pos_info=False):
	

	config = get_config(experiment)
	wells = get_experiment_wells(experiment)

	movie_prefix = ConfigSectionMap(config,"MovieSettings")["movie_prefix"]
	concentrations = get_experiment_concentrations(experiment, dtype=float)
	cell_types = get_experiment_cell_types(experiment)
	antibodies = get_experiment_antibodies(experiment)
	pharmaceutical_agents = get_experiment_pharmaceutical_agents(experiment)
	well_labels = _extract_labels_from_config(config,len(wells))
	
	well_indices, position_indices = _interpret_wells_and_positions(experiment, well_option, position_option)

	df = []
	df_pos_info = []
	real_well_index = 0
	
	for widx, well_path in enumerate(tqdm(wells[well_indices])):
		
		any_table = False # assume no table
		
		well_index = widx
		well_name, well_number = extract_well_name_and_number(well_path)
		well_alias = well_labels[widx]
		
		well_concentration = concentrations[widx]
		well_antibody = antibodies[widx]
		well_cell_type = cell_types[widx]
		well_pharmaceutical_agent = pharmaceutical_agents[widx]
		
		positions = np.array(natsorted(glob(well_path+'*'+os.sep)),dtype=str)
		if position_indices is not None:
			try:
				positions = positions[position_indices]
			except Exception as e:
				print(e)
				continue

		real_pos_index = 0
		for pidx,pos_path in enumerate(positions):
			
			pos_name = extract_position_name(pos_path)
			
			stack_path = get_position_movie_path(pos_path, prefix=movie_prefix)
			
			df_pos,table = get_position_table(pos_path, population=population, return_path=True)
			if df_pos is not None:
			
				df_pos['position'] = pos_path
				df_pos['well'] = well_path
				df_pos['well_index'] = well_number
				df_pos['well_name'] = well_name
				df_pos['pos_name'] = pos_name

				df_pos['concentration'] = well_concentration
				df_pos['antibody'] = well_antibody
				df_pos['cell_type'] = well_cell_type
				df_pos['pharmaceutical_agent'] = well_pharmaceutical_agent
				
				df.append(df_pos)
				any_table = True
				
				df_pos_info.append({'pos_path': pos_path, 'pos_index': real_pos_index, 'pos_name': pos_name, 'table_path': table, 'stack_path': stack_path,
									'well_path': well_path, 'well_index': real_well_index, 'well_name': well_name, 'well_number': well_number, 'well_alias': well_alias})
				
				real_pos_index+=1
		
		if any_table:
			real_well_index += 1
	
	df_pos_info = pd.DataFrame(df_pos_info)
	if len(df)>0:
		df = pd.concat(df)
		df = df.reset_index(drop=True)
	else:
		df = None
		
	if return_pos_info:
		return df, df_pos_info
	else:
		return df



def locate_stack(position, prefix='Aligned'):

	"""

	Locate and load a stack of images.

	Parameters
	----------
	position : str
		The position folder within the well where the stack is located.
	prefix : str, optional
		The prefix used to identify the stack. The default is 'Aligned'.

	Returns
	-------
	stack : ndarray
		The loaded stack as a NumPy array.

	Raises
	------
	AssertionError
		If no stack with the specified prefix is found.

	Notes
	-----
	This function locates and loads a stack of images based on the specified position and prefix.
	It assumes that the stack is stored in a directory named 'movie' within the specified position.
	The function loads the stack as a NumPy array and performs shape manipulation to have the channels
	at the end.

	Examples
	--------
	>>> stack = locate_stack(position, prefix='Aligned')
	# Locate and load a stack of images for further processing.

	"""

	stack_path = glob(position+os.sep.join(['movie', f'{prefix}*.tif']))
	assert len(stack_path)>0,f"No movie with prefix {prefix} found..."
	stack = imread(stack_path[0].replace('\\','/'))
	if stack.ndim==4:
		stack = np.moveaxis(stack, 1, -1)
	elif stack.ndim==3:
		stack = stack[:,:,:,np.newaxis]

	return stack

def locate_labels(position, population='target'):

	"""

	Locate and load labels for a specific population.

	Parameters
	----------
	position : str
		The position folder within the well where the stack is located.
	population : str, optional
		The population for which to locate the labels. 
		Valid options are 'target' and 'effector'.
		The default is 'target'.

	Returns
	-------
	labels : ndarray
		The loaded labels as a NumPy array.

	Notes
	-----
	This function locates and loads the labels for a specific population based on the specified position.
	It assumes that the labels are stored in a directory named 'labels' or 'labels_effectors'
	within the specified position, depending on the population.
	The function loads the labels as a NumPy array.

	Examples
	--------
	>>> labels = locate_labels(position, population='target')
	# Locate and load labels for the target population.

	"""


	if population.lower()=="target" or population.lower()=="targets":
		label_path = natsorted(glob(position+os.sep.join(["labels_targets", "*.tif"])))
	elif population.lower()=="effector" or population.lower()=="effectors":
		label_path = natsorted(glob(position+os.sep.join(["labels_effectors", "*.tif"])))
	labels = np.array([imread(i.replace('\\','/')) for i in label_path])

	return labels



def locate_stack_and_labels(position, prefix='Aligned', population="target"):

	"""

	Locate and load the stack and corresponding segmentation labels.

	Parameters
	----------
	position : str
		The position or directory path where the stack and labels are located.
	prefix : str, optional
		The prefix used to identify the stack. The default is 'Aligned'.
	population : str, optional
		The population for which the segmentation must be located. The default is 'target'.

	Returns
	-------
	stack : ndarray
		The loaded stack as a NumPy array.
	labels : ndarray
		The loaded segmentation labels as a NumPy array.

	Raises
	------
	AssertionError
		If no stack with the specified prefix is found or if the shape of the stack and labels do not match.

	Notes
	-----
	This function locates the stack and corresponding segmentation labels based on the specified position and population.
	It assumes that the stack and labels are stored in separate directories: 'movie' for the stack and 'labels' or 'labels_effectors' for the labels.
	The function loads the stack and labels as NumPy arrays and performs shape validation.

	Examples
	--------
	>>> stack, labels = locate_stack_and_labels(position, prefix='Aligned', population="target")
	# Locate and load the stack and segmentation labels for further processing.
	
	"""

	position = position.replace('\\','/')
	labels = locate_labels(position, population=population)
	stack = locate_stack(position, prefix=prefix)
	assert len(stack)==len(labels),f"The shape of the stack {stack.shape} does not match with the shape of the labels {labels.shape}"

	return stack,labels

def load_tracking_data(position, prefix="Aligned", population="target"):

	"""
	
	Load the tracking data, labels, and stack for a given position and population.

	Parameters
	----------
	position : str
		The position or directory where the data is located.
	prefix : str, optional
		The prefix used in the filenames of the stack images (default is "Aligned").
	population : str, optional
		The population to load the data for. Options are "target" or "effector" (default is "target").

	Returns
	-------
	trajectories : DataFrame
		The tracking data loaded as a pandas DataFrame.
	labels : ndarray
		The segmentation labels loaded as a numpy ndarray.
	stack : ndarray
		The image stack loaded as a numpy ndarray.

	Notes
	-----
	This function loads the tracking data, labels, and stack for a given position and population.
	It reads the trajectories from the appropriate CSV file based on the specified population.
	The stack and labels are located using the `locate_stack_and_labels` function.
	The resulting tracking data is returned as a pandas DataFrame, and the labels and stack are returned as numpy ndarrays.

	Examples
	--------
	>>> trajectories, labels, stack = load_tracking_data(position, population="target")
	# Load the tracking data, labels, and stack for the specified position and target population.

	"""

	position = position.replace('\\','/')
	if population.lower()=="target" or population.lower()=="targets":
		trajectories = pd.read_csv(position+os.sep.join(['output', 'tables', 'trajectories_targets.csv']))
	elif population.lower()=="effector" or population.lower()=="effectors":
		trajectories = pd.read_csv(position+os.sep.join(['output', 'tables', 'trajectories_effectors.csv']))

	stack,labels = locate_stack_and_labels(position, prefix=prefix, population=population)

	return trajectories,labels,stack


def auto_load_number_of_frames(stack_path):

	"""

	Automatically estimate the number of frames in a stack.

	Parameters
	----------
	stack_path : str
		The file path to the stack.

	Returns
	-------
	int or None
		The estimated number of frames in the stack. Returns None if the number of frames cannot be determined.

	Notes
	-----
	This function attempts to estimate the number of frames in a stack by parsing the image description metadata.
	It reads the stack file using the TiffFile from the tifffile library.
	It searches for metadata fields containing information about the number of slices or frames.
	If the number of slices or frames is found, it returns the estimated length of the movie.
	If the number of slices or frames cannot be determined, it returns None.

	Examples
	--------
	>>> len_movie = auto_load_number_of_frames(stack_path)
	# Automatically estimate the number of frames in the stack.
	
	"""

	# Try to estimate automatically # frames
	stack_path = stack_path.replace('\\','/')

	with TiffFile(stack_path) as tif:
		try:
			tif_tags = {}
			for tag in tif.pages[0].tags.values():
				name, value = tag.name, tag.value
				tif_tags[name] = value
			img_desc = tif_tags["ImageDescription"]
			attr = img_desc.split("\n")
		except:
			pass
		try:
			# Try nframes
			nslices = int(attr[np.argmax([s.startswith("frames") for s in attr])].split("=")[-1])
			if nslices>1:
				len_movie = nslices
				print(f"Auto-detected movie length movie: {len_movie}")
			else:
				break_the_code()
		except:
			try:
				# try nslices
				frames = int(attr[np.argmax([s.startswith("slices") for s in attr])].split("=")[-1])
				len_movie = frames
				print(f"Auto-detected movie length movie: {len_movie}")
			except:
				pass

	try:
		del tif;
		del tif_tags;
		del img_desc;
	except:
		pass
	gc.collect()

	return len_movie if 'len_movie' in locals() else None

def parse_isotropic_radii(string):
	sections = re.split(',| ', string)
	radii = []
	for k,s in enumerate(sections):
		if s.isdigit():
			radii.append(int(s))
		if '[' in s:
			ring = [int(s.replace('[','')), int(sections[k+1].replace(']',''))]
			radii.append(ring)
		else:
			pass
	return radii

def get_tracking_configs_list(return_path=False):

	"""
	
	Retrieve a list of available tracking configurations.

	Parameters
	----------
	return_path : bool, optional
		If True, also returns the path to the models. Default is False.

	Returns
	-------
	list or tuple
		If return_path is False, returns a list of available tracking configurations.
		If return_path is True, returns a tuple containing the list of models and the path to the models.

	Notes
	-----
	This function retrieves the list of available tracking configurations by searching for model directories
	in the predefined model path. The model path is derived from the parent directory of the current script
	location and the path to the model directory. By default, it returns only the names of the models.
	If return_path is set to True, it also returns the path to the models.

	Examples
	--------
	>>> models = get_tracking_configs_list()
	# Retrieve a list of available tracking configurations.

	>>> models, path = get_tracking_configs_list(return_path=True)
	# Retrieve a list of available tracking configurations.

	"""

	modelpath = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],"celldetective", "models", "tracking_configs", os.sep])
	available_models = glob(modelpath+'*.json')
	available_models = [m.replace('\\','/').split('/')[-1] for m in available_models]
	available_models = [m.replace('\\','/').split('.')[0] for m in available_models]


	if not return_path:
		return available_models
	else:
		return available_models, modelpath

def interpret_tracking_configuration(config):
	
	if isinstance(config, str):
		if os.path.exists(config):
			return config
		else:
			modelpath = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],"celldetective", "models", "tracking_configs", os.sep])
			if os.path.exists(modelpath+config+'.json'):
				return modelpath+config+'.json'
			else:
				config = cell_config()
	elif config is None:
		config = cell_config()

	return config

def get_signal_models_list(return_path=False):

	"""
	
	Retrieve a list of available signal detection models.

	Parameters
	----------
	return_path : bool, optional
		If True, also returns the path to the models. Default is False.

	Returns
	-------
	list or tuple
		If return_path is False, returns a list of available signal detection models.
		If return_path is True, returns a tuple containing the list of models and the path to the models.

	Notes
	-----
	This function retrieves the list of available signal detection models by searching for model directories
	in the predefined model path. The model path is derived from the parent directory of the current script
	location and the path to the model directory. By default, it returns only the names of the models.
	If return_path is set to True, it also returns the path to the models.

	Examples
	--------
	>>> models = get_signal_models_list()
	# Retrieve a list of available signal detection models.

	>>> models, path = get_signal_models_list(return_path=True)
	# Retrieve a list of available signal detection models and the path to the models.

	"""

	modelpath = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],"celldetective", "models", "signal_detection", os.sep])
	available_models = glob(modelpath+f'*{os.sep}')
	available_models = [m.replace('\\','/').split('/')[-2] for m in available_models]

	if not return_path:
		return available_models
	else:
		return available_models, modelpath


def relabel_segmentation(labels, data, properties, column_labels={'track': "track", 'frame': 'frame', 'y': 'y', 'x': 'x', 'label': 'class_id'}, threads=1):

	"""

	Relabel the segmentation labels based on the provided tracking data and properties.

	Parameters
	----------
	labels : ndarray
		The original segmentation labels.
	data : ndarray
		The tracking data containing information about tracks, frames, y-coordinates, and x-coordinates.
	properties : ndarray
		The properties associated with the tracking data.
	column_labels : dict, optional
		A dictionary specifying the column labels for the tracking data. The default is {'track': "track",
		'frame': 'frame', 'y': 'y', 'x': 'x', 'label': 'class_id'}.

	Returns
	-------
	ndarray
		The relabeled segmentation labels.

	Notes
	-----
	This function relabels the segmentation labels based on the provided tracking data and properties.
	It creates a DataFrame from the tracking data and properties, merges them based on the indices, and sorts them by track and frame.
	Then, it iterates over unique frames in the DataFrame, retrieves the tracks and identities at each frame,
	and updates the corresponding labels with the new track values.

	Examples
	--------
	>>> relabeled = relabel_segmentation(labels, data, properties, column_labels={'track': "track", 'frame': 'frame',
	...                                                                 'y': 'y', 'x': 'x', 'label': 'class_id'})
	# Relabel the segmentation labels based on the provided tracking data and properties.
	
	"""

	df = pd.DataFrame(data,columns=[column_labels['track'],column_labels['frame'],column_labels['y'],column_labels['x']])
	df = df.merge(pd.DataFrame(properties),left_index=True, right_index=True)
	df = df.sort_values(by=[column_labels['track'],column_labels['frame']])

	new_labels = np.zeros_like(labels)

	for t in tqdm(df[column_labels['frame']].unique()):
		f = int(t)
		tracks_at_t = df.loc[df[column_labels['frame']]==f, column_labels['track']].to_numpy()
		identities = df.loc[df[column_labels['frame']]==f, column_labels['label']].to_numpy()

		tracks_at_t = tracks_at_t[identities==identities]
		identities = identities[identities==identities]

		for k in range(len(identities)):
			loc_i,loc_j = np.where(labels[f]==identities[k])
			new_labels[f,loc_i,loc_j] = int(tracks_at_t[k])

	return new_labels

def control_tracking_btrack(position, prefix="Aligned", population="target", relabel=True, flush_memory=True, threads=1):

	"""
	Load the necessary data for visualization of bTrack trajectories in napari.

	Parameters
	----------
	position : str
		The path to the position directory.
	prefix : str, optional
		The prefix used to identify the movie file. The default is "Aligned".
	population : str, optional
		The population type to load, either "target" or "effector". The default is "target".

	Returns
	-------
	None
		This function displays the data in Napari for visualization and analysis.

	Examples
	--------
	>>> control_tracking_btrack("path/to/position", population="target")
	# Executes napari for visualization of target trajectories.

	"""

	data,properties,graph,labels,stack = load_napari_data(position, prefix=prefix, population=population)
	view_on_napari_btrack(data,properties,graph,labels=labels, stack=stack, relabel=relabel, flush_memory=flush_memory, threads=threads)

def view_on_napari_btrack(data,properties,graph,stack=None,labels=None,relabel=True, flush_memory=True, position=None, threads=1):
	
	"""

	Visualize btrack data, including stack, labels, points, and tracks, using the napari viewer.

	Parameters
	----------
	data : ndarray
		The btrack data containing information about tracks.
	properties : ndarray
		The properties associated with the btrack data.
	graph : Graph
		The btrack graph containing information about track connections.
	stack : ndarray, optional
		The stack of images to visualize. The default is None.
	labels : ndarray, optional
		The segmentation labels to visualize. The default is None.
	relabel : bool, optional
		Specify whether to relabel the segmentation labels using the provided data. The default is True.

	Notes
	-----
	This function visualizes btrack data using the napari viewer. It adds the stack, labels, points,
	and tracks to the viewer for visualization. If `relabel` is True and labels are provided, it calls
	the `relabel_segmentation` function to relabel the segmentation labels based on the provided data.

	Examples
	--------
	>>> view_on_napari_btrack(data, properties, graph, stack=stack, labels=labels, relabel=True)
	# Visualize btrack data, including stack, labels, points, and tracks, using the napari viewer.

	"""

	if (labels is not None)*relabel:
		print('Relabeling the cell masks with the track ID.')
		labels = relabel_segmentation(labels, data, properties, threads=threads)

	vertices = data[:, 1:]
	viewer = napari.Viewer()
	if stack is not None:
		viewer.add_image(stack,channel_axis=-1,colormap=["gray"]*stack.shape[-1])
	if labels is not None:
		viewer.add_labels(labels, name='segmentation',opacity=0.4)
	viewer.add_points(vertices, size=4, name='points', opacity=0.3)
	viewer.add_tracks(data, properties=properties, graph=graph, name='tracks')
	viewer.show(block=True)
	
	if flush_memory:
		# temporary fix for slight napari memory leak
		for i in range(10000):
			try:
				viewer.layers.pop()
			except:
				pass

		del viewer
		del stack
		del labels
		gc.collect()

def load_napari_data(position, prefix="Aligned", population="target"):

	"""
	Load the necessary data for visualization in napari.

	Parameters
	----------
	position : str
		The path to the position or experiment directory.
	prefix : str, optional
		The prefix used to identify the the movie file. The default is "Aligned".
	population : str, optional
		The population type to load, either "target" or "effector". The default is "target".

	Returns
	-------
	tuple
		A tuple containing the loaded data, properties, graph, labels, and stack.

	Examples
	--------
	>>> data, properties, graph, labels, stack = load_napari_data("path/to/position")
	# Load the necessary data for visualization of target trajectories.
	
	"""
	position = position.replace('\\','/')
	if population.lower()=="target" or population.lower()=="targets":
		napari_data = np.load(position+os.sep.join(['output','tables','napari_target_trajectories.npy']), allow_pickle=True)
	elif population.lower()=="effector" or population.lower()=="effectors":
		napari_data = np.load(position+os.sep.join(['output', 'tables', 'napari_effector_trajectories.npy']), allow_pickle=True)
	data = napari_data.item()['data']
	properties = napari_data.item()['properties']
	graph = napari_data.item()['graph']

	stack,labels = locate_stack_and_labels(position, prefix=prefix, population=population)

	return data,properties,graph,labels,stack


def control_segmentation_napari(position, prefix='Aligned', population="target", flush_memory=False):

	"""
	
	Control the visualization of segmentation labels using the napari viewer.

	Parameters
	----------
	position : str
		The position or directory path where the segmentation labels and stack are located.
	prefix : str, optional
		The prefix used to identify the stack. The default is 'Aligned'.
	population : str, optional
		The population type for which the segmentation is performed. The default is 'target'.

	Notes
	-----
	This function loads the segmentation labels and stack corresponding to the specified position and population.
	It then creates a napari viewer and adds the stack and labels as layers for visualization.

	Examples
	--------
	>>> control_segmentation_napari(position, prefix='Aligned', population="target")
	# Control the visualization of segmentation labels using the napari viewer.

	"""

	def export_labels():
		labels_layer = viewer.layers['segmentation'].data
		for t,im in enumerate(tqdm(labels_layer)):
			save_tiff_imagej_compatible(output_folder+f"{str(t).zfill(4)}.tif", im, axes='YX')
		print("The labels have been successfully rewritten.")

	def export_annotation():
		
		# Locate experiment config
		parent1 = Path(position).parent
		expfolder = parent1.parent
		config = PurePath(expfolder,Path("config.ini"))
		expfolder = str(expfolder)
		exp_name = os.path.split(expfolder)[-1]
		print(exp_name)

		spatial_calibration = float(ConfigSectionMap(config,"MovieSettings")["pxtoum"])
		channel_names, channel_indices = extract_experiment_channels(config)

		annotation_folder = expfolder + os.sep + f'annotations_{population}' + os.sep
		if not os.path.exists(annotation_folder):
			os.mkdir(annotation_folder)

		print('exporting!')
		t = viewer.dims.current_step[0]
		labels_layer = viewer.layers['segmentation'].data[t] # at current time

		frame = viewer.layers['Image'].data[t]
		# if frame.ndim==2:
		# 	frame = frame[np.newaxis,:,:]

		multichannel = [frame]
		for i in range(len(channel_indices)-1):
			try:
				frame = viewer.layers[f'Image [{i+1}]'].data[t]
				multichannel.append(frame)
			except:
				pass
		multichannel = np.array(multichannel)
		print(multichannel.shape)

		save_tiff_imagej_compatible(annotation_folder + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}_labelled.tif", labels_layer, axes='YX')
		save_tiff_imagej_compatible(annotation_folder + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}.tif", multichannel, axes='CYX')
		info = {"spatial_calibration": spatial_calibration, "channels": list(channel_names)}
		info_name = annotation_folder + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}.json"
		with open(info_name, 'w') as f:
			json.dump(info, f, indent=4)
		print('Done.')

	@magicgui(call_button='Save the modified labels')
	def save_widget():
		return export_labels()

	@magicgui(call_button='Export the annotation\nof the current frame')
	def export_widget():
		return export_annotation()

	stack,labels = locate_stack_and_labels(position, prefix=prefix, population=population)

	if not population.endswith('s'):
		population+='s'
	output_folder = position+f'labels_{population}{os.sep}'

	viewer = napari.Viewer()
	viewer.add_image(stack,channel_axis=-1,colormap=["gray"]*stack.shape[-1])
	viewer.add_labels(labels.astype(int), name='segmentation',opacity=0.4)
	viewer.window.add_dock_widget(save_widget, area='right')
	viewer.window.add_dock_widget(export_widget, area='right')
	viewer.show(block=True)

	if flush_memory:
		# temporary fix for slight napari memory leak
		for i in range(10000):
			try:
				viewer.layers.pop()
			except:
				pass

		del viewer
		del stack
		del labels
		gc.collect()


def _view_on_napari(tracks=None, stack=None, labels=None):
	
	"""

	Visualize tracks, stack, and labels using Napari.

	Parameters
	----------
	tracks : pandas DataFrame
		DataFrame containing track information.
	stack : numpy array, optional
		Stack of images with shape (T, Y, X, C), where T is the number of frames, Y and X are the spatial dimensions,
		and C is the number of channels. Default is None.
	labels : numpy array, optional
		Label stack with shape (T, Y, X) representing cell segmentations. Default is None.

	Returns
	-------
	None

	Notes
	-----
	This function visualizes tracks, stack, and labels using Napari, an interactive multi-dimensional image viewer.
	The tracks are represented as line segments on the viewer. If a stack is provided, it is displayed as an image.
	If labels are provided, they are displayed as a segmentation overlay on the stack.

	Examples
	--------
	>>> tracks = pd.DataFrame({'track': [1, 2, 3], 'time': [1, 1, 1],
	...                        'x': [10, 20, 30], 'y': [15, 25, 35]})
	>>> stack = np.random.rand(100, 100, 3)
	>>> labels = np.random.randint(0, 2, (100, 100))
	>>> view_on_napari(tracks, stack=stack, labels=labels)
	# Visualize tracks, stack, and labels using Napari.
	
	"""

	viewer = napari.Viewer()
	if stack is not None:
		viewer.add_image(stack,channel_axis=-1,colormap=["gray"]*stack.shape[-1])
	if labels is not None:
		viewer.add_labels(labels, name='segmentation',opacity=0.4)
	if tracks is not None:
		viewer.add_tracks(tracks, name='tracks')
	viewer.show(block=True)

def control_tracking_table(position, calibration=1, prefix="Aligned", population="target",
	column_labels={'track': "TRACK_ID", 'frame': 'FRAME', 'y': 'POSITION_Y', 'x': 'POSITION_X', 'label': 'class_id'}):

	"""
	
	Control the tracking table and visualize tracks using Napari.

	Parameters
	----------
	position : str
		The position or directory of the tracking data.
	calibration : float, optional
		Calibration factor for converting pixel coordinates to physical units. Default is 1.
	prefix : str, optional
		Prefix used for the tracking data file. Default is "Aligned".
	population : str, optional
		Population type, either "target" or "effector". Default is "target".
	column_labels : dict, optional
		Dictionary containing the column labels for the tracking table. Default is
		{'track': "TRACK_ID", 'frame': 'FRAME', 'y': 'POSITION_Y', 'x': 'POSITION_X', 'label': 'class_id'}.

	Returns
	-------
	None

	Notes
	-----
	This function loads the tracking data, applies calibration to the spatial coordinates, and visualizes the tracks
	using Napari. The tracking data is loaded from the specified `position` directory with the given `prefix` and
	`population`. The spatial coordinates (x, y) in the tracking table are divided by the `calibration` factor to
	convert them from pixel units to the specified physical units. The tracks are then visualized using Napari.

	Examples
	--------
	>>> control_tracking_table('path/to/tracking_data', calibration=0.1, prefix='Aligned', population='target')
	# Control the tracking table and visualize tracks using Napari.

	"""

	position = position.replace('\\','/')
	tracks,labels,stack = load_tracking_data(position, prefix=prefix, population=population)
	tracks = tracks.loc[:, [column_labels['track'], column_labels['frame'], column_labels['y'], column_labels['x']]].to_numpy()
	tracks[:,-2:] /= calibration
	_view_on_napari(tracks,labels=labels, stack=stack)


def get_segmentation_models_list(mode='targets', return_path=False):
	
	if mode=='targets':
		modelpath = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],"celldetective", "models", "segmentation_targets", os.sep])
	elif mode=='effectors':
		modelpath = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],"celldetective", "models", "segmentation_effectors", os.sep])

	available_models = glob(modelpath+'*/')
	available_models = [m.replace('\\','/').split('/')[-2] for m in available_models]

	if not return_path:
		return available_models
	else:
		return available_models, modelpath

def locate_segmentation_model(name):
	
	modelpath = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],"celldetective", "models", "segmentation*", os.sep])
	print(f'Looking for {name} in {modelpath}')
	models = glob(modelpath+'*/')

	match=None
	for m in models:
		if name==m.replace('\\','/').split('/')[-2]:
			match = m
			return match
	return match

def normalize(frame, percentiles=(0.0,99.99), values=None, ignore_gray_value=0., clip=False, amplification=None, dtype=float):

	"""
	
	Normalize the intensity values of a frame.

	Parameters
	----------
	frame : ndarray
		The input frame to be normalized.
	percentiles : tuple, optional
		The percentiles used to determine the minimum and maximum values for normalization. Default is (0.0, 99.99).
	values : tuple or None, optional
		The specific minimum and maximum values to use for normalization. If None, percentiles are used. Default is None.
	ignore_gray_value : float or None, optional
		The gray value to ignore during normalization. If specified, the pixels with this value will not be normalized. Default is 0.0.

	Returns
	-------
	ndarray
		The normalized frame.

	Notes
	-----
	This function performs intensity normalization on a frame. It computes the minimum and maximum values for normalization either
	using the specified values or by calculating percentiles from the frame. The frame is then normalized between the minimum and
	maximum values using the `normalize_mi_ma` function. If `ignore_gray_value` is specified, the pixels with this value will be
	left unmodified during normalization.

	Examples
	--------
	>>> frame = np.array([[10, 20, 30],
						  [40, 50, 60],
						  [70, 80, 90]])
	>>> normalized = normalize(frame)
	>>> normalized

	array([[0. , 0.2, 0.4],
		   [0.6, 0.8, 1. ],
		   [1.2, 1.4, 1.6]], dtype=float32)

	>>> normalized = normalize(frame, percentiles=(10.0, 90.0))
	>>> normalized

	array([[0.33333334, 0.44444445, 0.5555556 ],
		   [0.6666667 , 0.7777778 , 0.8888889 ],
		   [1.        , 1.1111112 , 1.2222222 ]], dtype=float32)

	"""

	frame = frame.astype(float)

	if ignore_gray_value is not None:
		subframe = frame[frame!=ignore_gray_value]
	else:
		subframe = frame.copy()

	if values is not None:
		mi = values[0]; ma = values[1]
	else:
		mi = np.nanpercentile(subframe.flatten(),percentiles[0],keepdims=True)
		ma = np.nanpercentile(subframe.flatten(),percentiles[1],keepdims=True)

	frame0 = frame.copy()
	frame = normalize_mi_ma(frame0, mi, ma, clip=False, eps=1e-20, dtype=np.float32)
	if amplification is not None:
		frame *= amplification
	if clip:
		if amplification is None:
			amplification = 1.
		frame[frame>=amplification] = amplification
		frame[frame<=0.] = 0.
	if ignore_gray_value is not None:
		frame[np.where(frame0)==ignore_gray_value] = ignore_gray_value

	return frame.copy().astype(dtype)

def normalize_multichannel(multichannel_frame, percentiles=None,
						   values=None, ignore_gray_value=0., clip=False,
						   amplification=None, dtype=float):
	
	mf = multichannel_frame.copy().astype(float)
	assert mf.ndim==3,f'Wrong shape for the multichannel frame: {mf.shape}.'
	if percentiles is None:
		percentiles = [(0.,99.99)]*mf.shape[-1]
	elif isinstance(percentiles,tuple):
		percentiles = [percentiles]*mf.shape[-1]
	if values is not None:
		if isinstance(values, tuple):
			values = [values]*mf.shape[-1]
		assert len(values)==mf.shape[-1],'Mismatch between the normalization values provided and the number of channels.'

	for c in range(mf.shape[-1]):
		if values is not None:
			v = values[c]
		else:
			v = None
		mf[:,:,c] = normalize(mf[:,:,c].copy(),
							  percentiles=percentiles[c],
							  values=v,
							  ignore_gray_value=ignore_gray_value,
							  clip=clip,
							  amplification=amplification,
							  dtype=dtype,
							  )
	return mf

def load_frames(img_nums, stack_path, scale=None, normalize_input=True, dtype=float, normalize_kwargs={"percentiles": (0.,99.99)}):

	try:
		frames = skio.imread(stack_path, img_num=img_nums, plugin="tifffile")
	except Exception as e:
		print(f'Error in loading the frame {img_nums} {e}. Please check that the experiment channel information is consistent with the movie being read.')
		return None

	if frames.ndim==3:
		# Systematically move channel axis to the end
		channel_axis = np.argmin(frames.shape)
		frames = np.moveaxis(frames, channel_axis, -1)
	if frames.ndim==2:
		frames = frames[:,:,np.newaxis]
	if normalize_input:
		frames = normalize_multichannel(frames, **normalize_kwargs)
	if scale is not None:
		frames = zoom(frames, [scale,scale,1], order=3)
	
	# add a fake pixel to prevent auto normalization errors on images that are uniform
	# to revisit
	for k in range(frames.shape[2]):
		unique_values = np.unique(frames[:,:,k])
		if len(unique_values)==1:
			frames[0,0,k] += 1

	return frames.astype(dtype)


def get_stack_normalization_values(stack, percentiles=None, ignore_gray_value=0.):

	assert stack.ndim==4,f'Wrong number of dimensions for the stack, expect TYXC (4) got {stack.ndim}.'
	if percentiles is None:
		percentiles = [(0.,99.99)]*stack.shape[-1]
	elif isinstance(percentiles,tuple):
		percentiles = [percentiles]*stack.shape[-1]
	elif isinstance(percentiles,list):
		assert len(percentiles)==stack.shape[-1],f'Mismatch between the provided percentiles and the number of channels {stack.shape[-1]}. If you meant to apply the same percentiles to all channels, please provide a single tuple.'

	values = []
	for c in range(stack.shape[-1]):
		perc = percentiles[c]
		mi = np.nanpercentile(stack[:,:,:,c].flatten(),perc[0],keepdims=True)[0]
		ma = np.nanpercentile(stack[:,:,:,c].flatten(),perc[1],keepdims=True)[0]
		values.append(tuple((mi,ma)))
		gc.collect()

	return values


if __name__ == '__main__':
	control_segmentation_napari("/home/limozin/Documents/Experiments/MinimumJan/W4/401/", prefix='Aligned', population="target", flush_memory=False)
