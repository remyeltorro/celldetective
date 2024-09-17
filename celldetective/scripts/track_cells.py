"""
Copright © 2022 Laboratoire Adhesion et Inflammation, Authored by Remy Torro.
"""

import argparse
import datetime
import os
import json
from celldetective.io import auto_load_number_of_frames, load_frames, interpret_tracking_configuration
from celldetective.utils import extract_experiment_channels, ConfigSectionMap, _get_img_num_per_channel, extract_experiment_channels
from celldetective.measure import drop_tonal_features, measure_features
from celldetective.tracking import track
from pathlib import Path, PurePath
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc
import os
from natsort import natsorted
from art import tprint
from tifffile import imread

tprint("Track")

parser = argparse.ArgumentParser(description="Segment a movie in position with the selected model",
								formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p',"--position", required=True, help="Path to the position")
parser.add_argument("--mode", default="target", choices=["target","effector","targets","effectors"],help="Cell population of interest")
parser.add_argument("--threads", default="1",help="Number of parallel threads")

args = parser.parse_args()
process_arguments = vars(args)
pos = str(process_arguments['position'])
mode = str(process_arguments['mode'])
n_threads = int(process_arguments['threads'])

if not os.path.exists(pos+"output"):
	os.mkdir(pos+"output")

if not os.path.exists(pos+os.sep.join(["output","tables"])):
	os.mkdir(pos+os.sep.join(["output","tables"]))

if mode.lower()=="target" or mode.lower()=="targets":
	label_folder = "labels_targets"
	instruction_file = os.sep.join(["configs", "tracking_instructions_targets.json"])
	napari_name = "napari_target_trajectories.npy"
	table_name = "trajectories_targets.csv"

elif mode.lower()=="effector" or mode.lower()=="effectors":
	label_folder = "labels_effectors"
	instruction_file = os.sep.join(["configs","tracking_instructions_effectors.json"])
	napari_name = "napari_effector_trajectories.npy"
	table_name = "trajectories_effectors.csv"

# Locate experiment config
parent1 = Path(pos).parent
expfolder = parent1.parent
config = PurePath(expfolder,Path("config.ini"))
assert os.path.exists(config),'The configuration file for the experiment could not be located. Abort.'

# from exp config fetch spatial calib, channel names
movie_prefix = ConfigSectionMap(config,"MovieSettings")["movie_prefix"]
spatial_calibration = float(ConfigSectionMap(config,"MovieSettings")["pxtoum"])
time_calibration = float(ConfigSectionMap(config,"MovieSettings")["frametomin"])
len_movie = float(ConfigSectionMap(config,"MovieSettings")["len_movie"])
shape_x = int(ConfigSectionMap(config,"MovieSettings")["shape_x"])
shape_y = int(ConfigSectionMap(config,"MovieSettings")["shape_y"])

channel_names, channel_indices = extract_experiment_channels(config)
nbr_channels = len(channel_names)

# from tracking instructions, fetch btrack config, features, haralick, clean_traj, idea: fetch custom timeline?
instr_path = PurePath(expfolder,Path(f"{instruction_file}"))
if os.path.exists(instr_path):
	print(f"Tracking instructions for the {mode} population have been successfully loaded...")
	with open(instr_path, 'r') as f:
		instructions = json.load(f)
	btrack_config = interpret_tracking_configuration(instructions['btrack_config_path'])

	if 'features' in instructions:
		features = instructions['features']
	else:
		features = None
	
	if 'mask_channels' in instructions:
		mask_channels = instructions['mask_channels']
	else:
		mask_channels = None
	
	if 'haralick_options' in instructions:
		haralick_options = instructions['haralick_options']
	else:
		haralick_options = None

	if 'post_processing_options' in instructions:
		post_processing_options = instructions['post_processing_options']
	else:
		post_processing_options = None
else:
	print('Tracking instructions could not be located... Using a standard bTrack motion model instead...')
	btrack_config = interpret_tracking_configuration(None)
	features = None
	mask_channels = None
	haralick_options = None
	post_processing_options = None

if features is None:
	features = []

# from pos fetch labels
label_path = natsorted(glob(pos+f"{label_folder}"+os.sep+"*.tif"))
if len(label_path)>0:
	print(f"Found {len(label_path)} segmented frames...")
else:
	print(f"No segmented frames have been found. Please run segmentation first. Abort...")
	os.abort()

# Do this if features or Haralick is not None, else don't need stack
try:
	file = glob(pos+os.sep.join(["movie", f"{movie_prefix}*.tif"]))[0]
except IndexError:
	print('Movie could not be found. Check the prefix. If you intended to measure texture or tone, this will not be performed.')
	file = None
	haralick_option = None
	features = drop_tonal_features(features)

len_movie_auto = auto_load_number_of_frames(file)
if len_movie_auto is not None:
	len_movie = len_movie_auto

img_num_channels = _get_img_num_per_channel(channel_indices, len_movie, nbr_channels)

#######################################
# Loop over all frames and find objects
#######################################

timestep_dataframes = []
features_log=f'features: {features}'
mask_channels_log=f'mask_channels: {mask_channels}'
haralick_option_log=f'haralick_options: {haralick_options}'
post_processing_option_log=f'post_processing_options: {post_processing_options}'
log_list=[features_log, mask_channels_log, haralick_option_log, post_processing_option_log]
log='\n'.join(log_list)

with open(pos+f'log_{mode}.json', 'a') as f:
	f.write(f'{datetime.datetime.now()} TRACK \n')
	f.write(log+"\n")

def measure_index(indices):
	for t in tqdm(indices,desc="frame"):
		
		# Load channels at time t
		img = load_frames(img_num_channels[:,t], file, scale=None, normalize_input=False)
		lbl = imread(label_path[t])

		df_props = measure_features(img, lbl, features = features+['centroid'], border_dist=None, 
										channels=channel_names, haralick_options=haralick_options, verbose=False, 
									)
		df_props.rename(columns={'centroid-1': 'x', 'centroid-0': 'y'},inplace=True)
		df_props['t'] = int(t)
		timestep_dataframes.append(df_props)

# Multithreading
indices = list(range(img_num_channels.shape[1]))
chunks = np.array_split(indices, n_threads)

import concurrent.futures

with concurrent.futures.ThreadPoolExecutor() as executor:
	executor.map(measure_index, chunks)

# threads = []
# for i in range(n_threads):
# 	thread_i = threading.Thread(target=measure_index, args=[chunks[i]])
# 	threads.append(thread_i)
# for th in threads:
# 	th.start()
# for th in threads:
# 	th.join()

df = pd.concat(timestep_dataframes)	
df.reset_index(inplace=True, drop=True)

if mask_channels is not None:
	cols_to_drop = []
	for mc in mask_channels:
		columns = df.columns
		col_contains = [mc in c for c in columns]
		to_remove = np.array(columns)[np.array(col_contains)]
		cols_to_drop.extend(to_remove)
	if len(cols_to_drop)>0:
		df = df.drop(cols_to_drop, axis=1)

# do tracking
trajectories, napari_data = track(None,
					configuration=btrack_config,
					objects=df, 
					spatial_calibration=spatial_calibration, 
					channel_names=channel_names,
					return_napari_data=True,
		  			optimizer_options = {'tm_lim': int(12e4)}, 
		  			track_kwargs={'step_size': 100}, 
		  			clean_trajectories_kwargs=post_processing_options, 
		  			volume=(shape_x, shape_y),
		  			)

# out trajectory table, create POSITION_X_um, POSITION_Y_um, TIME_min (new ones)
# Save napari data
np.save(pos+os.sep.join(['output', 'tables', napari_name]), napari_data, allow_pickle=True)
print(f"napari data successfully saved in {pos+os.sep.join(['output', 'tables'])}")

trajectories.to_csv(pos+os.sep.join(['output', 'tables', table_name]), index=False)
print(f"Table {table_name} successfully saved in {os.sep.join(['output', 'tables'])}")

if os.path.exists(pos+os.sep.join(['output', 'tables', table_name.replace('.csv','.pkl')])):
	os.remove(pos+os.sep.join(['output', 'tables', table_name.replace('.csv','.pkl')]))

del trajectories; del napari_data;
gc.collect()