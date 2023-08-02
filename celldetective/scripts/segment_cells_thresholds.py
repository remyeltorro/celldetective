"""
Copright © 2022 Laboratoire Adhesion et Inflammation, Authored by Remy Torro.
"""

import argparse
import os
import json
from celldetective.io import auto_load_number_of_frames, load_frames
from celldetective.segmentation import segment_frame_from_thresholds
from celldetective.utils import _estimate_scale_factor, _extract_channel_indices_from_config, _extract_channel_indices, ConfigSectionMap, _extract_nbr_channels_from_config, _get_img_num_per_channel, extract_experiment_channels
from pathlib import Path, PurePath
from glob import glob
from shutil import rmtree
from tqdm import tqdm
import numpy as np
from csbdeep.io import save_tiff_imagej_compatible
import gc
from art import tprint

tprint("Segment")

parser = argparse.ArgumentParser(description="Segment a movie in position with a threshold pipeline",
								formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p',"--position", required=True, help="Path to the position")
parser.add_argument('-c',"--config", required=True,help="Threshold instructions")
parser.add_argument("--mode", default="target", choices=["target","effector","targets","effectors"],help="Cell population of interest")

args = parser.parse_args()
process_arguments = vars(args)
pos = str(process_arguments['position'])
mode = str(process_arguments['mode'])
threshold_instructions = str(process_arguments['config'])


if os.path.exists(threshold_instructions):
	with open(threshold_instructions, 'r') as f:
		threshold_instructions = json.load(f)
		required_channels = [threshold_instructions['target_channel']]
else:
	print('The configuration path is not valid. Abort.')
	os.abort()

print('The following instructions were successfully loaded: ', threshold_instructions)

if mode.lower()=="target" or mode.lower()=="targets":
	label_folder = "labels_targets/"
elif mode.lower()=="effector" or mode.lower()=="effectors":
	label_folder = "labels_effectors/"

# Locate experiment config
parent1 = Path(pos).parent
expfolder = parent1.parent
config = PurePath(expfolder,Path("config.ini"))
assert os.path.exists(config),'The configuration file for the experiment could not be located. Abort.'
print("Configuration file: ",config)


channel_indices = _extract_channel_indices_from_config(config, required_channels)
# need to abort if channel not found
print(f'Required channels: {required_channels} located at channel indices {channel_indices}.')

threshold_instructions.update({'target_channel': channel_indices[0]})

movie_prefix = ConfigSectionMap(config,"MovieSettings")["movie_prefix"]
len_movie = float(ConfigSectionMap(config,"MovieSettings")["len_movie"])
channel_names, channel_indices = extract_experiment_channels(config)
threshold_instructions.update({'channel_names': channel_names})

# Try to find the file
try:
	file = glob(pos+f"movie/{movie_prefix}*.tif")[0]
except IndexError:
	print('Movie could not be found. Check the prefix.')
	os.abort()

len_movie_auto = auto_load_number_of_frames(file)
if len_movie_auto is not None:
	len_movie = len_movie_auto

nbr_channels = _extract_nbr_channels_from_config(config)
print(f'Number of channels in the input movie: {nbr_channels}')
img_num_channels = _get_img_num_per_channel(np.arange(nbr_channels), len_movie, nbr_channels)

# If everything OK, prepare output, load models
print('Erasing previous segmentation folder.')
if os.path.exists(pos+label_folder):
	rmtree(pos+label_folder)
os.mkdir(pos+label_folder)
print(f'Folder {pos+label_folder} successfully generated.')


# Loop over all frames and segment
for t in tqdm(range(img_num_channels.shape[1]),desc="frame"):
	
	# Load channels at time t
	f = load_frames(img_num_channels[:,t], file, scale=None, normalize_input=False)
	mask = segment_frame_from_thresholds(f, **threshold_instructions)
	save_tiff_imagej_compatible(pos+label_folder+f"{str(t).zfill(4)}.tif", mask, axes='YX')

	del f;
	del mask;
	gc.collect()

print('Done.')
gc.collect()




