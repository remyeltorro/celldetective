import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from celldetective.signals import derivative
from glob import glob
from natsort import natsorted
import os
import subprocess
from celldetective.signals import velocity, magnitude_velocity, sliding_msd
import seaborn as sns
from math import ceil
from skimage.io import imread
abs_path = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], 'celldetective'])

def relative_quantities_per_pos2(pos, target_classes, neigh_dist=200, target_lysis_class='class_custom',
                                 target_lysis_time='t_custom', pre_lysis_time_window=5,
                                 velocity_kwargs={'window': 1, 'mode': 'bi'},
                                 neighborhood_kwargs={'status': None, 'include_dead_weight': True,
                                                      "compute_cum_sum": False, "attention_weight": False}):
    """
    pos: position to process
    target_classes [list]: target classes to keep
    neigh_dist: neighborhood cut distance
    theta_dist: distance to edge threshold
    target_lysis_class: name of class to filter targets on
    target_lysis_time: name of time col to find lysis times
    pre_lysis_time_window: number of frames before lysis time to average relative quantities
    velocity_kwargs: params for derivative of relative position
    neighborhood_kwargs: params for neigh
    """

    df_rel = []

    # Load tables
    tab_tc = pos + os.sep.join(['output', 'tables', 'trajectories_targets.csv'])
    if not os.path.exists(tab_tc):
        return None
    neigh_trajectories_path = pos + f'output/tables/trajectories_targets.pkl'
    df_targets = pd.read_csv(tab_tc)
    df_effectors = pd.read_csv(tab_tc.replace('targets', 'effectors'))
    target_pickle = pd.read_pickle(neigh_trajectories_path)
    df_targets.loc[:, f'neighborhood_2_circle_{neigh_dist}_px'] = target_pickle[f'neighborhood_2_circle_{neigh_dist}_px']

    for tid, group in df_targets.loc[df_targets[target_lysis_class].isin(target_classes), :].groupby('TRACK_ID'):
        # loop over targets in lysis class of interest
        print(tid)
        t0 = ceil(group[target_lysis_time].to_numpy()[0])
        if t0<=0:
           t0 = 5
        #print(group)
        neighbours = group.loc[group['FRAME'] <=t0 , f'neighborhood_2_circle_{neigh_dist}_px'].values  # all neighbours
        #print(neighbours)
        timeline_til_lysis = group.loc[group['FRAME'] <= t0, 'FRAME'].to_numpy()
        timeline = group['FRAME'].to_numpy()

        pi = group['dead_nuclei_channel_mean'].to_numpy()
        coords = group[['POSITION_X', 'POSITION_Y']].to_numpy()
        target_class = group[target_lysis_class].values[0]

        # all NK neighbours until target death
        nk_ids = []
        for t in range(len(timeline_til_lysis)):
            n = neighbours[t]
            if not n:
                pass
            if isinstance(n, float):
                pass
            elif t > t0:
                continue
            else:
                for nn in n:
                    nk_ids.append(nn['id'])

        unique_nks = list(np.unique(nk_ids))
        print(f'TC {tid} with t_lysis {t0}: found {len(unique_nks)} NK neighs: {unique_nks}...')
        nk_neighs = df_effectors.query(f"TRACK_ID.isin({unique_nks})")  # locate the NKs of interest in NK table

        for nk, group_nk in nk_neighs.groupby('TRACK_ID'):

            coords_nk = group_nk[['POSITION_X', 'POSITION_Y']].to_numpy()
            lamp = group_nk['fluo_channel_1_mean'].to_numpy()
            timeline_nk = group_nk['FRAME'].to_numpy()

            # Perform timeline matching to have same start-end points and no gaps
            full_timeline, index_tc, index_nk = timeline_matching(timeline, timeline_nk)
            relative_distance = np.zeros(len(full_timeline))
            relative_distance[:] = np.nan
            relative_distance_xy1 = np.zeros((len(full_timeline), 2))
            relative_distance_xy1[:, :] = np.nan
            relative_angle1 = np.zeros(len(full_timeline))
            relative_angle1[:] = np.nan
            relative_distance_xy2 = np.zeros((len(full_timeline), 2))
            relative_distance_xy2[:, :] = np.nan
            relative_angle2 = np.zeros(len(full_timeline))
            relative_angle2[:] = np.nan

            # Relative distance
            for t in range(len(relative_distance)):
                if t in timeline and t in timeline_nk:
                    idx1 = np.where(timeline == t)[0][0]
                    idx2 = np.where(timeline_nk == t)[0][0]
                    relative_distance[t] = np.sqrt(
                        (coords[idx1, 0] - coords_nk[idx2, 0]) ** 2 + (coords[idx1, 1] - coords_nk[idx2, 1]) ** 2)
                    relative_distance_xy1[t, 0] = coords[idx1, 0] - coords_nk[idx2, 0]
                    relative_distance_xy1[t, 1] = coords[idx1, 1] - coords_nk[idx2, 1]
                    relative_angle1[t] = np.arctan2(relative_distance_xy1[t, 1],
                                                    relative_distance_xy1[t, 0]) * 180 / np.pi
                    relative_distance_xy2[t, 0] = coords_nk[idx2, 0] - coords[idx1, 0]
                    relative_distance_xy2[t, 1] = coords_nk[idx2, 1] - coords[idx1, 1]
                    relative_angle2[t] = np.arctan2(relative_distance_xy2[t, 1],
                                                    relative_distance_xy2[t, 0]) * 180 / np.pi

            dddt = derivative(relative_distance, full_timeline, **velocity_kwargs)
            # Determine if NK is annotated as synapse before target death
            # print("RELATIVE VELOCITY")
            # print(len(dddt))

            nk_synapse = group_nk.loc[group_nk['FRAME'] <= ceil(t0), 'live_status'].to_numpy()
            if len(nk_synapse) > 0:
                nk_synapse = int(np.any(nk_synapse.astype(bool)))
            else:
                nk_synapse = 0

            for t in range(len(relative_distance)):
                df_rel.append({'target': tid, 'effector': nk, 'frame': t, 'relative_distance': relative_distance[t],
                               'relative_velocity': dddt[t], 't0_lysis': t0, 'angle_tc-eff': relative_angle1[t],
                               'angle-eff-tc': relative_angle2[t]})


    return df_rel


def timeline_matching(timeline1, timeline2):
    min_t = np.amin(np.concatenate((timeline1, timeline2)))
    max_t = np.amax(np.concatenate((timeline1, timeline2)))
    full_timeline = np.arange(min_t, max_t + 1)
    index1 = [list(np.where(full_timeline == int(t))[0])[0] for t in timeline1]
    index2 = [list(np.where(full_timeline == int(t))[0])[0] for t in timeline2]
    return full_timeline, index1, index2



def rel_measure_at_position(pos):

    pos = pos.replace('\\', '/')
    pos = rf"{pos}"
    assert os.path.exists(pos), f'Position {pos} is not a valid path.'
    if not pos.endswith('/'):
        pos += '/'
    script_path = os.sep.join([abs_path, 'scripts', 'measure_relative.py'])
    cmd = f'python "{script_path}" --pos "{pos}"'
    subprocess.call(cmd, shell=True)
