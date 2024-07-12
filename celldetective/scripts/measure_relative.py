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

		df_pairs = measure_pair_signals_at_position(pos,reference=neigh_protocol['reference'],neighbor=neigh_protocol['neighbor'],neigh_dist=neigh_protocol['distance'], target_classes=[0,1,2],description=neigh_protocol['description'])
		df_pairs['reference_population']=neigh_protocol['reference']
		df_pairs[f"{neigh_protocol['description']}"] = 1
		all_df_pairs.append(df_pairs)

		# Figure this out
		# # Check if REFERENCE_ID, NEIGHBOR_ID, and POPULATION are the same
		# if neigh_protocol['reference']!=neigh_protocol['neighbor']:
		# 	common_cols = ['REFERENCE_ID', 'NEIGHBOR_ID', 'ref_population']
		# 	matching_rows = df_test.merge(rel[common_cols], on=common_cols, how='inner')
		# 	if not matching_rows.empty:
		# 		# Update description columns for matching rows
		# 		for desc_col in [col for col in rel.columns if col.startswith('neighborhood') or col.startswith('status') or col.startswith('class')]:

		# 			df_test.loc[df_test.set_index(common_cols).index.isin(
		# 				matching_rows.set_index(common_cols).index), desc_col] = 1
		# 	else:
		# 		# Append rel to df_test to add new information
		# 		df_test = pd.concat([df_test, rel], ignore_index=True)
		# else:
		# 	df_test = pd.concat([df_test, rel], ignore_index=True)

		# # Fill NaN values in description columns with 'No'
		# description_cols = [col for col in df_test.columns if col.startswith('neighborhood')]
		# for col in description_cols:
		# 	#df_test[col].fillna(0, inplace=True)
		# 	df_test.fillna({col: 0}, inplace=True)

	path = pos + os.sep.join(['output', 'tables', 'cell_pair_measurements.csv']) 
	df_test.to_csv(path, index=False)

print('Done.')

