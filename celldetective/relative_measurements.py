import pandas as pd
import numpy as np
from celldetective.signals import derivative
import os
import subprocess
from math import ceil
abs_path = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], 'celldetective'])
import random
from tqdm import tqdm

def relative_quantities_per_pos2(pos, target_classes, neigh_dist, target_lysis_class='class_custom',
                                 target_lysis_time='t_custom', pre_lysis_time_window=5,
                                 velocity_kwargs={'window': 1, 'mode': 'bi'},):
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
    pts = []

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
        try:
            nk_neighs = df_effectors.query(f"TRACK_ID.isin({unique_nks})")  # locate the NKs of interest in NK table
        except:
            nk_neighs = df_effectors.query(f"ID.isin({unique_nks})")
        if 'TRACK_ID' in nk_neighs.columns:
            id_type='TRACK_ID'
        else:
            id_type='ID'

        for nk, group_nk in nk_neighs.groupby(id_type):

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

            nk_synapse = group_nk.loc[group_nk['FRAME'] <= ceil(t0), 'live_status'].to_numpy()
            if len(nk_synapse) > 0:
                nk_synapse = int(np.any(nk_synapse.astype(bool)))
            else:
                nk_synapse = 0

            rel_dist_til_lysis = relative_distance[:ceil(t0) + 1]
            duration_in_neigh = len(rel_dist_til_lysis[rel_dist_til_lysis <= neigh_dist]) / (ceil(t0) + 1)
            if target_classes[0] != 1.:

                t_low = max(ceil(t0) - pre_lysis_time_window, 0)
                t_high = ceil(t0) + 1

                rel_dist_crop = relative_distance[t_low:t_high]
                rel_v_crop = dddt[t_low:t_high]

                t_high_lamp = min(ceil(t0) + 1 + pre_lysis_time_window, df_targets['FRAME'].max())
                nk_lamp_crop = lamp[t_low:t_high_lamp]

                if len(rel_dist_crop[rel_dist_crop == rel_dist_crop]) > 0:
                    pre_lysis_d_rel = np.nanmean(rel_dist_crop)
                else:
                    pre_lysis_d_rel = np.nanmean(relative_distance[:])
                if len(rel_v_crop[rel_v_crop == rel_v_crop]) > 0:
                    pre_lysis_v_rel = np.nanmean(rel_v_crop)
                else:
                    pre_lysis_v_rel = np.nanmean(dddt[:])

                if len(nk_lamp_crop[nk_lamp_crop == nk_lamp_crop]) > 0:
                    nk_lamp = np.nanmean(nk_lamp_crop)
                else:
                    nk_lamp=np.nanmean(lamp[:])

                syn_class = nk_synapse

            else:
                pre_lysis_d_rel = np.nanmean(relative_distance[:])
                pre_lysis_v_rel = np.nanmean(dddt[:])
                syn_class = np.amax(nk_synapse[:])
                nk_lamp = np.nanmean(lamp[:])
            pts.append({'tc': tid, 'lysis_time': t0, 'nk': nk, 'drel': pre_lysis_d_rel, 'vrel': pre_lysis_v_rel,
                        'syn_class': syn_class, 'lamp1': nk_lamp, 'relxy': relative_distance_xy1,
                        't_residence_rel': duration_in_neigh})

            for t in range(len(relative_distance)):
                df_rel.append({'TARGET_ID': tid, 'EFFECTOR_ID': nk, 'FRAME': t, 'distance': relative_distance[t],
                               'velocity': dddt[t], 't0_lysis': t0, 'angle_tc-eff': relative_angle1[t],
                               'angle-eff-tc': relative_angle2[t],'probability':0})
    pts = pd.DataFrame(pts)
    probs = probabilities(pts)
    df_rel = pd.DataFrame(df_rel)
    for index,row in pts.iterrows():
        df_rel.loc[(df_rel['TARGET_ID'] == row['tc']) & (df_rel['EFFECTOR_ID'] == row['nk']), 'distance_mean'] = row[
            'drel']
        df_rel.loc[(df_rel['TARGET_ID'] == row['tc']) & (df_rel['EFFECTOR_ID'] == row['nk']), 'velocity_mean'] = row[
            'vrel']
        df_rel.loc[(df_rel['TARGET_ID'] == row['tc']) & (df_rel['EFFECTOR_ID'] == row['nk']), 't_residence_rel'] = row[
            't_residence_rel']
    for prob in probs:
        for index,row in prob.iterrows():
            df_rel.loc[(df_rel['TARGET_ID'] == row['tc']) & (df_rel['EFFECTOR_ID'] == row['nk']),'probability']=row['total_prob']


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


def mcf7_size_model(x,x0,x2):
    return np.piecewise(x, [x<= x0, (x > x0)*(x<=x2), x > x2], [lambda x: 1, lambda x: -1/(x2-x0)*x + (1+x0/(x2-x0)), 0])
def sigmoid(x,x0,k):
    return 1/(1 + np.exp(-(x-x0)/k))
def velocity_law(x):
    return np.piecewise(x, [x<=-10, x > -10],[lambda x: 0., lambda x: (1*x+10)*(1-sigmoid(x, 1,1))/10])
def probabilities(pairs,radius_critical=80,radius_max=150):
    scores = []
    pair_dico=[]
    print(f'Found {len(pairs)} TC-NK pairs...')
    if len(pairs) > 0:
        unique_tcs = np.unique(pairs['tc'].to_numpy())
        unique_nks = np.unique(pairs['nk'].to_numpy())
        matrix = np.zeros((len(unique_tcs), len(unique_nks)))
        for index, row in pairs.iterrows():

            i = np.where(unique_tcs == row['tc'])[0]
            j = np.where(unique_nks == row['nk'])[0]

            d_prob = mcf7_size_model(row['drel'], radius_critical, radius_max)
            lamp_prob = sigmoid(row['lamp1'], 1.05, 0.01)
            synapse_prob = row['syn_class']
            velocity_prob = velocity_law(row['vrel'])  # 1-sigmoid(row['vrel'], 1,1)
            time_prob = row['t_residence_rel']

            hypotheses = [d_prob, velocity_prob, lamp_prob, synapse_prob,
                          time_prob]  # lamp_prob d_prob, synapse_prob, velocity_prob, lamp_prob
            s = np.sum(hypotheses) / len(hypotheses)

            matrix[i, j] = s  # synapse_prob': synapse_prob,
            pair_dico.append(
                { 'tc': row['tc'], 'nk': row['nk'], 'synapse_prob': synapse_prob,
                 'd_prob': d_prob, 'lamp_prob': lamp_prob, 'velocity_prob': velocity_prob, 'time_prob': time_prob})
        pair_dico = pd.DataFrame(pair_dico)

        hypotheses = ['velocity_prob', 'd_prob', 'time_prob', 'lamp_prob', 'synapse_prob']

        for i in tqdm(range(2000)):
            sample = np.array(random.choices(np.linspace(0, 1, 100), k=len(hypotheses)))
            weights = sample / np.sum(sample)

            score_i = {}
            for k, hyp in enumerate(hypotheses):
                score_i.update({'w_' + hyp: weights[k]})
            probs=[]
            for cells, group in pair_dico.groupby(['tc']):

                group['total_prob'] = 0
                for hyp in hypotheses:
                    group['total_prob'] += group[hyp] * score_i['w_' + hyp]
                    probs.append(group)
    return probs

def update_effector_table(df_relative, df_effector):
    df_effector['group_neighborhood']=1
    effectors = np.unique(df_relative['EFFECTOR_ID'].to_numpy())
    for effector in effectors:
        if effector in df_effector['TRACK_ID'].values:
            # Set group_neighborhood to 0 where TRACK_ID matches effector
            df_effector.loc[df_effector['TRACK_ID'] == effector, 'group_neighborhood'] = 0
    return df_effector


