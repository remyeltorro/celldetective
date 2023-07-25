"""
Copright © 2022 Laboratoire Adhesion et Inflammation, Authored by Remy Torro.
"""

import argparse
import os
import json
from celldetective.io import auto_load_number_of_frames, load_frames, interpret_tracking_configuration
from celldetective.utils import extract_experiment_channels, _extract_channel_indices_from_config, _extract_channel_indices, ConfigSectionMap, _extract_nbr_channels_from_config, _get_img_num_per_channel, extract_experiment_channels
from celldetective.measure import drop_tonal_features
from pathlib import Path, PurePath
from glob import glob
from shutil import rmtree
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc
import os
from natsort import natsorted
from art import tprint

tprint("Track")

parser = argparse.ArgumentParser(description="Segment a movie in position with the selected model",
								formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p',"--position", required=True, help="Path to the position")
parser.add_argument("--mode", default="target", choices=["target","effector","targets","effectors"],help="Cell population of interest")

args = parser.parse_args()
process_arguments = vars(args)
pos = str(process_arguments['position'])
mode = str(process_arguments['mode'])

if mode.lower()=="target" or mode.lower()=="targets":
	label_folder = "labels_targets"
	instruction_file = "tracking_instructions_targets.json"
elif mode.lower()=="effector" or mode.lower()=="effectors":
	label_folder = "labels_effectors"
	instruction_file = "tracking_instructions_effectors.json"

# Locate experiment config
parent1 = Path(pos).parent
expfolder = parent1.parent
config = PurePath(expfolder,Path("config.ini"))
assert os.path.exists(config),'The configuration file for the experiment could not be located. Abort.'
print("Configuration file: ",config)

# from exp config fetch spatial calib, channel names
movie_prefix = ConfigSectionMap(config,"MovieSettings")["movie_prefix"]
spatial_calibration = float(ConfigSectionMap(config,"MovieSettings")["pxtoum"])
time_calibration = float(ConfigSectionMap(config,"MovieSettings")["frametomin"])
len_movie = float(ConfigSectionMap(config,"MovieSettings")["len_movie"])
channel_names, channel_indices = extract_experiment_channels(config)
nbr_channels = len(channel_names)


# from tracking instructions, fetch btrack config, features, haralick, clean_traj, idea: fetch custom timeline?
instr_path = PurePath(expfolder,Path(f"{instruction_file}"))
if os.path.exists(instr_path):
	print(f"Tracking instructions for the {mode} population has been successfully located.")
	with open(instr_path, 'r') as f:
		instructions = json.load(f)
		print(instructions)
	btrack_config = interpret_tracking_configuration(instructions['btrack_config_path'])
	features = instructions['features']
	haralick_option = instructions['haralick_options']
	post_processing_options = instructions['post_processing_options']
else:
	print('No tracking instructions found. Use standard bTrack motion model.')
	btrack_config = interpret_tracking_configuration(None)
	features = None
	haralick_option = None
	post_processing_options = None

# from pos fetch labels
label_path = natsorted(glob(pos+f"{label_folder}/*.tif"))
if len(label_path)>0:
	print(f"Found {len(label_path)} segmented frames...")
else:
	print(f"No segmented frames have been found. Please run segmentation first, skipping...")
	os.abort()

# Do this if features or Haralick is not None, else don't need stack
try:
	file = glob(pos+f"movie/{movie_prefix}*.tif")[0]
except IndexError:
	print('Movie could not be found. Check the prefix. If you intended to measure texture or tone, this will not be performed.')
	file = None
	haralick_option = None
	features = drop_tonal_features(features)

len_movie_auto = auto_load_number_of_frames(file)
if len_movie_auto is not None:
	len_movie = len_movie_auto


img_num_channels = _get_img_num_per_channel(channel_indices, len_movie, nbr_channels)
for c,cn in zip(channel_names, img_num_channels):
	print(c, cn)

# do tracking

# out trajectory table, create POSITION_X_um, POSITION_Y_um, TIME_min (new ones)