import argparse
import os
import json
from celldetective.relative_measurements import relative_quantities_per_pos2, update_effector_table, check_tables
from celldetective.utils import ConfigSectionMap, extract_experiment_channels

from pathlib import Path, PurePath

import pandas as pd

from art import tprint


tprint("Measure")

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
channel_names, channel_indices = extract_experiment_channels(config)
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
    neighbors_to_measure=check_tables(pos)
    df_test=pd.DataFrame()
    for ind,dic in enumerate(neighbors_to_measure):

        rel=relative_quantities_per_pos2(pos,reference=dic['reference'],neighbor=dic['neighbor'],neigh_dist=dic['distance'], target_classes=[0,1,2],description=dic['description'])
        print(rel)
        rel['ref_population']=dic['reference']
        rel[f"{dic['description']}"] = 1
        rel=pd.DataFrame(rel)
        if ind==0:
            df_test=pd.DataFrame(rel)
        else:
            # Check if REFERENCE_ID, NEIGHBOR_ID, and POPULATION are the same
            if dic['reference']!=dic['neighbor']:
                common_cols = ['REFERENCE_ID', 'NEIGHBOR_ID', 'ref_population']
                matching_rows = df_test.merge(rel[common_cols], on=common_cols, how='inner')
                if not matching_rows.empty:
                    # Update description columns for matching rows
                    for desc_col in [col for col in rel.columns if col.startswith('neighborhood') or col.startswith('status') or col.startswith('class')]:

                        df_test.loc[df_test.set_index(common_cols).index.isin(
                            matching_rows.set_index(common_cols).index), desc_col] = 1
                else:
                    # Append rel to df_test to add new information
                    df_test = pd.concat([df_test, rel], ignore_index=True)
            else:
                df_test = pd.concat([df_test, rel], ignore_index=True)

        # Fill NaN values in description columns with 'No'
        description_cols = [col for col in df_test.columns if col.startswith('neighborhood')]
        for col in description_cols:
            #df_test[col].fillna(0, inplace=True)
            df_test.fillna({col: 0}, inplace=True)

    # Fill NaN values in description columns with 'No'
    # description_cols = [col for col in df_test.columns if 'neighborhood' in col]
    # for col in description_cols:
    #     df_test[col].fillna('No', inplace=True)
    #print(df_test)
    #for row in df_test.iterrows():
        #print(row)

    # rel = pd.DataFrame(relative_quantities_per_pos2(pos, [0,2], neigh_dist=distance))
    path = pos + 'output/tables/relative_measurements_neighborhood.csv'
    df_test.to_csv(path, index=False)
    # tab_eff = pos + os.sep.join(['output', 'tables', 'trajectories_effectors.csv'])
    # df_effectors = pd.read_csv(tab_eff)
    # updated_eff=update_effector_table(rel,df_effectors)
    # updated_eff.to_csv(tab_eff, index=False)
    # print(f'Measurements successfully written in table {pos + os.sep.join(["output", "tables", "relative_measurements_neighborhood.csv"])}')
    #print('Done.')

