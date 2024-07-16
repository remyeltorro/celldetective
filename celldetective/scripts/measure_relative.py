import argparse
import os
import json
from celldetective.relative_measurements import measure_pair_signals_at_position, update_effector_table, extract_neighborhoods_from_pickles
from celldetective.utils import ConfigSectionMap, extract_experiment_channels

from pathlib import Path, PurePath

import pandas as pd

from art import tprint


tprint("Measure pairs")

parser = argparse.ArgumentParser(description="Measure features and intensities in a multichannel timeseries.",
								 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', "--position", required=True, help="Path to the position")

args = parser.parse_args()
process_arguments = vars(args)
pos = str(process_arguments['position'])

instruction_file = os.sep.join(['configs', "neighborhood_instructions.json"])

# Locate experiment config
parent1 = Path(pos).parent
expfolder = parent1.parent
config = PurePath(expfolder, Path("config.ini"))
assert os.path.exists(config), 'The configuration file for the experiment could not be located. Abort.'
print("Configuration file: ", config)

# from exp config fetch spatial calib, channel names
movie_prefix = ConfigSectionMap(config, "MovieSettings")["movie_prefix"]
spatial_calibration = float(ConfigSectionMap(config, "MovieSettings")["pxtoum"])
time_calibration = float(ConfigSectionMap(config, "MovieSettings")["frametomin"])
len_movie = float(ConfigSectionMap(config, "MovieSettings")["len_movie"])
channel_names, channel_inneigh_protocoles = extract_experiment_channels(config)
nbr_channels = len(channel_names)

# from tracking instructions, fetch btrack config, features, haralick, clean_traj, idea: fetch custom timeline?
instr_path = PurePath(expfolder, Path(f"{instruction_file}"))
if os.path.exists(instr_path):
	print(f"Neighborhood instructions has been successfully located.")
	with open(instr_path, 'r') as f:
		instructions = json.load(f)
		print("Reading the following instructions: ", instructions)

	if 'distance' in instructions:
		distance = instructions['distance'][0]
	else:
		distance = None

else:
	print('No measurement instructions found')
	os.abort()
if distance is None:
	print('No measurement could be performed. Check your inputs.')
	print('Done.')
	os.abort()
	#distance = 0
else:
	neighborhoods_to_measure = extract_neighborhoods_from_pickles(pos)
	all_df_pairs = []
	for k,neigh_protocol in enumerate(neighborhoods_to_measure):

		df_pairs = measure_pair_signals_at_position(pos, neigh_protocol)
		if 'REFERENCE_ID' in list(df_pairs.columns):
			all_df_pairs.append(df_pairs)

print(f'{len(all_df_pairs)} neighborhood measurements sets were computed...')
if len(all_df_pairs)>1:
	print('Merging...')
	df_pairs = all_df_pairs[0]
	for i in range(1,len(all_df_pairs)):
		df_pairs = pd.merge(df_pairs, all_df_pairs[i], how="outer", on=['REFERENCE_ID','NEIGHBOR_ID','reference_population', 'neighbor_population', 'FRAME', 'distance', 'velocity', 'angle', 'angular_velocity'])
elif len(all_df_pairs)==1:
	df_pairs = all_df_pairs[0]

print('Writing table...')
path = pos + os.sep.join(['output', 'tables', 'cell_pair_measurements.csv']) 
df_pairs.to_csv(path, index=False)
print('Done.')

