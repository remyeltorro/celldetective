import pandas as pd
import numpy as np
from celldetective.signals import derivative
import os
import subprocess
from math import ceil
abs_path = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], 'celldetective'])
import random
from tqdm import tqdm

def relative_quantities_per_pos2(pos, reference, neighbor,target_classes, neigh_dist, description, target_lysis_class='class_custom',
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
    tab_tc = pos + os.sep.join(['output', 'tables', f'trajectories_{reference}.csv'])
    if not os.path.exists(tab_tc):
        return None
    neigh_trajectories_path = pos + f'output/tables/trajectories_{reference}.pkl'
    df_reference = pd.read_csv(tab_tc)
    if reference != neighbor:
        df_neighbor = pd.read_csv(tab_tc.replace(f'{reference}', f'{neighbor}'))
    else:
        df_neighbor = df_reference
    #df_effectors = pd.read_csv(tab_tc.replace('targets', 'effectors'))
    reference_pickle = pd.read_pickle(neigh_trajectories_path)
    #try:
    df_reference.loc[:, f'{description}'] = reference_pickle[f'{description}']
    neigh_col=reference_pickle[f'{description}']
    #for tid, group in df_reference.loc[df_reference[f'{description}'].apply(lambda x: x != [])].groupby('TRACK_ID'):

    try:
        for tid, group in df_reference.groupby('TRACK_ID'):
            # loop over targets in lysis class of interest
            # t0 = ceil(group[target_lysis_time].to_numpy()[0])
            # if t0<=0:
            #     t0 = 5
            t0=0
            # print(type(group))
            # print(group)
            neighbours = group.loc[group['FRAME'] >=t0 , f'{description}'].values  # all neighbours

            #timeline_til_lysis = group.loc[group['FRAME'] <= t0, 'FRAME'].to_numpy()
            timeline = group['FRAME'].to_numpy()

    #
    #     pi = group['dead_nuclei_channel_mean'].to_numpy()
            coords = group[['POSITION_X', 'POSITION_Y']].to_numpy()
    #     target_class = group[target_lysis_class].values[0]
    #
    #     # all NK neighbours until target death
            neigh_ids = []
            t0_arrival={}
            t_departure={}
            for t in range(len(timeline)):
                n = neighbours[t]
                all_ids_at_t=[]
                if isinstance(n, float):
                        pass
                else:
                    for nn in n:
                        if nn['id'] not in neigh_ids:
                            t0_arrival[nn['id']]=t
                        neigh_ids.append(nn['id'])
                        all_ids_at_t.append(nn['id'])
                    for id in neigh_ids:
                        if id not in all_ids_at_t:
                            if id not in t_departure.keys():
                                t_departure[id]=t

                #print(neigh_ids)
                #for n in neighbours:
                # if isinstance(n, float):
                #     pass
                #     else:
                #         for i in range(0,len(n)):
                #             print(n[i]['id'])
                #             neigh_ids.append(n[i]['id'])
    #     for t in range(len(timeline_til_lysis)):
    #         n = neighbours[t]
    #         if not n:
    #             pass
    #         if isinstance(n, float):
    #             pass
    #         elif t > t0:
    #             continue
    #         else:
    #             for nn in n:
    #                 nk_ids.append(nn['id'])
    #
            unique_neigh = list(np.unique(neigh_ids))
            print(f'reference cell {tid} : found {len(unique_neigh)} neighbour cells: {unique_neigh}...')
            try:
                cells_neighs = df_neighbor.query(f"TRACK_ID.isin({unique_neigh})")  # locate the NKs of interest in NK table
            except:
                cells_neighs = df_neighbor.query(f"ID.isin({unique_neigh})")
            if 'TRACK_ID' in cells_neighs.columns:
                id_type='TRACK_ID'
            else:
                id_type='ID'

            for nc, group_nc in cells_neighs.groupby(id_type):
                coords_nc = group_nc[['POSITION_X', 'POSITION_Y']].to_numpy()
                # lamp = group_nk['fluo_channel_1_mean'].to_numpy()
                timeline_nc = group_nc['FRAME'].to_numpy()
                #
                # # Perform timeline matching to have same start-end points and no gaps
                full_timeline, index_tc, index_nk = timeline_matching(timeline, timeline_nc)
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
    #         # Relative distance
                for t in range(len(relative_distance)):

                    if t in timeline and t in timeline_nc:
                        idx1 = np.where(timeline == t)[0][0]
                        idx2 = np.where(timeline_nc == t)[0][0]
                        relative_distance[t] = np.sqrt(
                            (coords[idx1, 0] - coords_nc[idx2, 0]) ** 2 + (coords[idx1, 1] - coords_nc[idx2, 1]) ** 2)

                        relative_distance_xy1[t, 0] = coords[idx1, 0] - coords_nc[idx2, 0]
                        relative_distance_xy1[t, 1] = coords[idx1, 1] - coords_nc[idx2, 1]
                        angle1 = np.arctan2(relative_distance_xy1[t, 1], relative_distance_xy1[t, 0]) * 180 / np.pi
                        if angle1 < 0:
                            angle1 += 360
                        relative_angle1[t] = angle1
                        relative_distance_xy2[t, 0] = coords_nc[idx2, 0] - coords[idx1, 0]
                        relative_distance_xy2[t, 1] = coords_nc[idx2, 1] - coords[idx1, 1]
                        angle2 = np.arctan2(relative_distance_xy2[t, 1], relative_distance_xy2[t, 0]) * 180 / np.pi
                        if angle2 < 0:
                            angle2 += 360
                        relative_angle2[t] = angle2
                dddt = derivative(relative_distance, full_timeline, **velocity_kwargs)
                angular_velocity = np.zeros(len(full_timeline))
                angular_velocity[:] = np.nan

                for t in range(1, len(relative_angle1)):
                    if not np.isnan(relative_angle1[t]) and not np.isnan(relative_angle1[t - 1]):
                        delta_angle = relative_angle1[t] - relative_angle1[t - 1]
                        delta_time = full_timeline[t] - full_timeline[t - 1]
                        if delta_time != 0:
                            angular_velocity[t] = delta_angle / delta_time
    #         nk_synapse = group_nk.loc[group_nk['FRAME'] <= ceil(t0), 'live_status'].to_numpy()
    #         if len(nk_synapse) > 0:
    #             nk_synapse = int(np.any(nk_synapse.astype(bool)))
    #         else:
    #             nk_synapse = 0
    #
                neighb_dist=float(neigh_dist.split('_')[0])
                rel_dist_til_lysis = relative_distance[:ceil(t0) + 1]
                duration_in_neigh = len(rel_dist_til_lysis[rel_dist_til_lysis <= neighb_dist]) / (ceil(t0) + 1)
    #         if target_classes[0] != 1.:
    #
                t_low = max(ceil(t0) - pre_lysis_time_window, 0)
                t_high = ceil(t0) + 1
    #
                rel_dist_crop = relative_distance[t_low:t_high]
                rel_v_crop = dddt[t_low:t_high]
    #
    #             t_high_lamp = min(ceil(t0) + 1 + pre_lysis_time_window, df_targets['FRAME'].max())
    #             nk_lamp_crop = lamp[t_low:t_high_lamp]

                if len(rel_dist_crop[rel_dist_crop == rel_dist_crop]) > 0:
                    pre_lysis_d_rel = np.nanmean(rel_dist_crop)
                else:
                    pre_lysis_d_rel = np.nanmean(relative_distance[:])
                if len(rel_v_crop[rel_v_crop == rel_v_crop]) > 0:
                    pre_lysis_v_rel = np.nanmean(rel_v_crop)
                else:
                    pre_lysis_v_rel = np.nanmean(dddt[:])
    #
    #             if len(nk_lamp_crop[nk_lamp_crop == nk_lamp_crop]) > 0:
    #                 nk_lamp = np.nanmean(nk_lamp_crop)
    #             else:
    #                 nk_lamp=np.nanmean(lamp[:])
    #
    #             syn_class = nk_synapse
    #
            # else:
            #     pre_lysis_d_rel = np.nanmean(relative_distance[:])
            #     pre_lysis_v_rel = np.nanmean(dddt[:])
            #     #syn_class = np.amax(nk_synapse[:])
            #     #nk_lamp = np.nanmean(lamp[:])
                pts.append({'rc': tid, 'lysis_time': t0, 'nc': nc, 'drel': pre_lysis_d_rel, 'vrel': pre_lysis_v_rel,
                         'relxy': relative_distance_xy1,
                        't_residence_rel': duration_in_neigh})
        # pts.append({'rc': tid, 'lysis_time': t0, 'nc': nc, 'drel': pre_lysis_d_rel, 'vrel': pre_lysis_v_rel,
        #             'syn_class': syn_class, 'lamp1': nk_lamp, 'relxy': relative_distance_xy1,
        #             't_residence_rel': duration_in_neigh})
                for t in range(len(relative_distance)):
                    if nc in t_departure:
                        if t_departure[nc] > t >= t0_arrival[nc]:
                            df_rel.append(
                                {'REFERENCE_ID': tid, 'NEIGHBOR_ID': nc, 'FRAME': t, 'distance': relative_distance[t],
                                 'velocity': dddt[t], f't0_{description}': t0_arrival[nc],
                                 f't1_{description}': t_departure[nc], 'angle_tc-eff': relative_angle1[t],
                                 'angle-eff-tc': relative_angle2[t], 'angular_velocity': angular_velocity[t],
                                 f'status_{description}': 1})
                        else:

                            df_rel.append(
                                {'REFERENCE_ID': tid, 'NEIGHBOR_ID': nc, 'FRAME': t, 'distance': relative_distance[t],
                                 'velocity': dddt[t], f't0_{description}': t0_arrival[nc],
                                 f't1_{description}': t_departure[nc], 'angle_tc-eff': relative_angle1[t],
                                 'angle-eff-tc': relative_angle2[t], 'angular_velocity': angular_velocity[t],
                                 f'status_{description}': 0})
                    else:
                        if t >= t0_arrival[nc]:

                            df_rel.append(
                                {'REFERENCE_ID': tid, 'NEIGHBOR_ID': nc, 'FRAME': t, 'distance': relative_distance[t],
                                 'velocity': dddt[t], f't0_{description}': t0_arrival[nc],
                                 f't1_{description}': -1, 'angle_tc-eff': relative_angle1[t],
                                 'angle-eff-tc': relative_angle2[t], 'angular_velocity': angular_velocity[t],
                                 f'status_{description}': 1})
                        else:
                            df_rel.append(
                                {'REFERENCE_ID': tid, 'NEIGHBOR_ID': nc, 'FRAME': t, 'distance': relative_distance[t],
                                 'velocity': dddt[t], f't0_{description}': t0_arrival[nc],
                                 f't1_{description}': -1, 'angle_tc-eff': relative_angle1[t],
                                 'angle-eff-tc': relative_angle2[t], 'angular_velocity': angular_velocity[t],
                                 f'status_{description}': 0})

                # for t in range(len(relative_distance)):
                #     df_rel.append({'TARGET_ID': tid, 'EFFECTOR_ID': nc, 'FRAME': t, 'distance': relative_distance[t],
                #                    'velocity': dddt[t], 't0_lysis': t0, 'angle_tc-eff': relative_angle1[t],
                #                    'angle-eff-tc': relative_angle2[t],'probability':0,'angular_velocity': angular_velocity[t]})
        pts = pd.DataFrame(pts)
    # #probs = probabilities(pts)

        df_rel = pd.DataFrame(df_rel)
        for index,row in pts.iterrows():
            df_rel.loc[(df_rel['REFERENCE_ID'] == row['rc']) & (df_rel['NEIGHBOR_ID'] == row['nc']), 'distance_mean'] = row[
                'drel']
            df_rel.loc[(df_rel['REFERENCE_ID'] == row['rc']) & (df_rel['NEIGHBOR_ID'] == row['nc']), 'velocity_mean'] = row[
                'vrel']
            df_rel.loc[(df_rel['REFERENCE_ID'] == row['rc']) & (df_rel['NEIGHBOR_ID'] == row['nc']), 't_residence_rel'] = row[
                't_residence_rel']
        return df_rel
    # #for prob in probs:
    #     #for index,row in prob.iterrows():
    #         #df_rel.loc[(df_rel['TARGET_ID'] == row['tc']) & (df_rel['EFFECTOR_ID'] == row['nk']),'probability']=row['total_prob']
    #
    #
    # return df_rel
    except KeyError:
        print(f"Neighborhood {description} not found in data frame. Measurements for this neighborhood will not be calculated")


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
        try:
            # Set group_neighborhood to 0 where TRACK_ID matches effector
            df_effector.loc[df_effector['TRACK_ID'] == effector, 'group_neighborhood'] = 0
        except:
            df_effector.loc[df_effector['ID'] == effector, 'group_neighborhood'] = 0
    return df_effector

def check_tables(pos):
    tab_tc = pos + os.sep.join(['output', 'tables', 'trajectories_targets.csv'])
    if not os.path.exists(tab_tc):
        return None
    df_targets = pd.read_csv(tab_tc)
    df_effectors = pd.read_csv(tab_tc.replace('targets', 'effectors'))
    neighborhood_columns=[]
    for column in df_targets.columns:
        if column.startswith('inclusive_count_neighborhood'):
            if 'self' in column:
                if 'circle' in column:
                    distance=column.split('circle_')[1]
                    description=column.split('inclusive_count_')[1]
                    neigh={'reference':'targets','neighbor':'targets','type':'circle','distance':distance,'description':description}
                    check=column.split('circle_')[1]
                    print(check)
                else:
                    distance=column.split('contact_')[1]
                    description=column.split('inclusive_count_')[1]
                    neigh={'reference':'targets','neighbor':'targets','type':'contact','distance':distance,'description':description}
                    check=column.split('contact_')[1]
                    print(check)
            else:
                if 'circle' in column:
                    distance=column.split('circle_')[1]
                    description=column.split('inclusive_count_')[1]
                    neigh={'reference':'targets','neighbor':'effectors','type':'circle','distance':distance,'description':description}
                    check=column.split('circle_')[1]
                    print(check)
                else:
                    distance=column.split('contact_')[1]
                    description=column.split('inclusive_count_')[1]
                    neigh={'reference':'targets','neighbor':'effectors','type':'contact','distance':distance,'description':description}
                    check=column.split('contact_')[1]
                    print(check)
            neighborhood_columns.append(neigh)
    for column in df_effectors.columns:
        if column.startswith('inclusive_count_neighborhood'):
            if 'self' in column:
                if 'circle' in column:
                    distance=column.split('circle_')[1]
                    description=column.split('inclusive_count_')[1]
                    neigh={'reference':'effectors','neighbor':'effectors','type':'circle','distance':distance,'description':description}
                    check=column.split('circle_')[1]
                    print(check)
                else:
                    distance=column.split('contact_')[1]
                    description=column.split('inclusive_count_')[1]
                    neigh={'reference':'effectors','neighbor':'effectors','type':'contact','distance':distance,'description':description}
                    check=column.split('contact_')[1]
                    print(check)
            else:
                if 'circle' in column:
                    distance=column.split('circle_')[1]
                    description=column.split('inclusive_count_')[1]
                    neigh={'reference':'effectors','neighbor':'targets','type':'circle','distance':distance,'description':description}
                    check=column.split('circle_')[1]
                    print(check)
                else:
                    distance=column.split('contact_')[1]
                    description=column.split('inclusive_count_')[1]
                    neigh={'reference':'effectors','neighbor':'targets','type':'contact','distance':distance,'description':description}
                    check=column.split('contact_')[1]
                    print(check)
            neighborhood_columns.append(neigh)
    return neighborhood_columns
