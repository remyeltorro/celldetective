import pandas as pd
import numpy as np
from celldetective.utils import derivative
import os
import subprocess
from math import ceil
abs_path = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], 'celldetective'])
import random
from tqdm import tqdm

def measure_pair_signals_at_position(pos, neighborhood_protocol, velocity_kwargs={'window': 3, 'mode': 'bi'}):
	"""
	pos: position to process
	target_classes [list]: target classes to keep
	neigh_dist: neighborhood cut distance
	theta_dist: distance to edge threshold
	velocity_kwargs: params for derivative of relative position
	neighborhood_kwargs: params for neigh
	"""

	reference_population = neighborhood_protocol['reference']
	neighbor_population = neighborhood_protocol['neighbor']
	neighborhood_type = neighborhood_protocol['type']
	neighborhood_distance = neighborhood_protocol['distance']
	neighborhood_description = neighborhood_protocol['description']

	relative_measurements = []

	tab_ref = pos + os.sep.join(['output', 'tables', f'trajectories_{reference_population}.pkl'])
	if os.path.exists(tab_ref):
		df_reference = np.load(tab_ref, allow_pickle=True)
	else:
		df_reference = None

	if os.path.exists(tab_ref.replace(reference_population, neighbor_population)):
		df_neighbor = np.load(tab_ref.replace(reference_population, neighbor_population), allow_pickle=True)
	else:
		if os.path.exists(tab_ref.replace(reference_population, neighbor_population).replace('.pkl','.csv')):
			df_neighbor = pd.read_csv(tab_ref.replace(reference_population, neighbor_population).replace('.pkl','.csv'))
		else:
			df_neighbor = None

	if df_reference is None:
		return None

	assert str(neighborhood_description) in list(df_reference.columns)
	neighborhood = df_reference.loc[:,f'{neighborhood_description}'].to_numpy()

	if 'TRACK_ID' in df_reference:
		df_reference = df_reference.sort_values(by=['TRACK_ID','FRAME'])
	else:
		print('TRACK_ID not found in the reference population... Abort...')
		return None

	if 'TRACK_ID' in list(df_neighbor.columns):
		neigh_id_col = 'TRACK_ID'
		compute_velocity = True
	elif 'ID' in list(df_neighbor.columns):
		neigh_id_col = 'ID'
		compute_velocity = False
	else:
		print('ID or TRACK ID column could not be found in neighbor table. Abort.')
		return None

	try:
		for tid, group in df_reference.groupby('TRACK_ID'):

			neighbor_dicts = group.loc[: , f'{neighborhood_description}'].values
			timeline_reference = group['FRAME'].to_numpy()
			coords_reference = group[['POSITION_X', 'POSITION_Y']].to_numpy()

			neighbor_ids = []
			neighbor_ids_per_t = []

			time_of_first_entrance_in_neighborhood = {}
			t_departure={}

			for t in range(len(timeline_reference)):

				neighbors_at_t = neighbor_dicts[t]
				neighs_t = []
				if isinstance(neighbors_at_t, float) or neighbors_at_t!=neighbors_at_t:
					pass
				else:
					for neigh in neighbors_at_t:
						if neigh['id'] not in neighbor_ids:
							time_of_first_entrance_in_neighborhood[neigh['id']]=t
						neighbor_ids.append(neigh['id'])
						neighs_t.append(neigh['id'])
				neighbor_ids_per_t.append(neighs_t)

			#print(neighbor_ids_per_t)
			unique_neigh = list(np.unique(neighbor_ids))
			#print(f'Reference cell {tid}: found {len(unique_neigh)} neighbour cells: {unique_neigh}...')

			neighbor_properties = df_neighbor.loc[df_neighbor[neigh_id_col].isin(unique_neigh)]

			for nc, group_neigh in neighbor_properties.groupby(neigh_id_col):
				
				coords_neighbor = group_neigh[['POSITION_X', 'POSITION_Y']].to_numpy()
				timeline_neighbor = group_neigh['FRAME'].to_numpy()

				# # Perform timeline matching to have same start-end points and no gaps
				full_timeline, _, _ = timeline_matching(timeline_reference, timeline_neighbor)

				neighbor_vector = np.zeros((len(full_timeline), 2))
				neighbor_vector[:,:] = np.nan

				# relative_distance_xy1 = np.zeros((len(full_timeline), 2))
				# relative_distance_xy1[:, :] = np.nan

				# relative_angle1 = np.zeros(len(full_timeline))
				# relative_angle1[:] = np.nan

				# relative_distance_xy2 = np.zeros((len(full_timeline), 2))
				# relative_distance_xy2[:, :] = np.nan

				# relative_angle2 = np.zeros(len(full_timeline))
				# relative_angle2[:] = np.nan

				# Relative distance
				for t in range(len(full_timeline)):

					if t in timeline_reference and t in timeline_neighbor: # meaning position exists on both sides

						idx_reference = list(timeline_reference).index(t) #index_reference[list(full_timeline).index(t)]
						idx_neighbor = list(timeline_neighbor).index(t) #index_neighbor[list(full_timeline).index(t)]

						neighbor_vector[t, 0] = coords_neighbor[idx_neighbor, 0] - coords_reference[idx_reference, 0]
						neighbor_vector[t, 1] = coords_neighbor[idx_neighbor, 1] - coords_reference[idx_reference, 1]

						# relative_distance_xy1[t, 0] = coords_reference[idx_reference, 0] - coords_neighbor[idx_neighbor, 0]
						# relative_distance_xy1[t, 1] = coords_reference[idx_reference, 1] - coords_neighbor[idx_neighbor, 1]
						# relative_distance[t] = np.sqrt((relative_distance_xy1[t, 0])** 2 + (relative_distance_xy1[t, 1])** 2)

						# # TO CHECK CAREFULLY
						# angle1 = np.arctan2(relative_distance_xy1[t, 1], relative_distance_xy1[t, 0]) * 180 / np.pi
						# if angle1 < 0:
						# 	angle1 += 360
						# relative_angle1[t] = angle1

						# relative_distance_xy2[t, 0] = coords_neighbor[idx_neighbor, 0] - coords_reference[idx_reference, 0]
						# relative_distance_xy2[t, 1] = coords_neighbor[idx_neighbor, 1] - coords_reference[idx_reference, 1]
						# angle2 = np.arctan2(relative_distance_xy2[t, 1], relative_distance_xy2[t, 0]) * 180 / np.pi
						# if angle2 < 0:
						# 	angle2 += 360
						# relative_angle2[t] = angle2

				angle = np.arctan2(neighbor_vector[:, 1], neighbor_vector[:, 0])
				#print(f'Angle before unwrap: {angle}')
				angle = np.unwrap(angle)
				#print(f'Angle after unwrap: {angle}')
				relative_distance = np.sqrt(neighbor_vector[:,0]**2 + neighbor_vector[:, 1]**2)
				#print(f'Timeline: {full_timeline}; Distance: {relative_distance}')

				if compute_velocity:
					rel_velocity = derivative(relative_distance, full_timeline, **velocity_kwargs)
					#rel_velocity = np.insert(rel_velocity, 0, np.nan)[:-1]

					angular_velocity = derivative(angle, full_timeline, **velocity_kwargs)


				# 	angular_velocity = np.zeros(len(full_timeline))
				# 	angular_velocity[:] = np.nan

				# 	for t in range(1, len(relative_angle1)):
				# 		if not np.isnan(relative_angle1[t]) and not np.isnan(relative_angle1[t - 1]):
				# 			delta_angle = relative_angle1[t] - relative_angle1[t - 1]
				# 			delta_time = full_timeline[t] - full_timeline[t - 1]
				# 			if delta_time != 0:
				# 				angular_velocity[t] = delta_angle / delta_time

				duration_in_neigh = list(neighbor_ids).count(nc)
				#print(nc, duration_in_neigh, ' frames')

				cum_sum = 0
				for t in range(len(full_timeline)):

					if t in timeline_reference: # meaning position exists on both sides

						idx_reference = list(timeline_reference).index(t) 

						if nc in neighbor_ids_per_t[idx_reference]:

							cum_sum+=1
							relative_measurements.append(
									{'REFERENCE_ID': tid, 'NEIGHBOR_ID': nc,
									'reference_population': reference_population,
									'neighbor_population': neighbor_population,
									'FRAME': t, 'distance': relative_distance[t],
									'velocity': rel_velocity[t], 
									'angle': angle[t] * 180 / np.pi,
									#'angle-neigh-ref': angle[t] * 180 / np.pi, 
									'angular_velocity': angular_velocity[t],
									f'status_{neighborhood_description}': 1,
									f'residence_time_in_{neighborhood_description}': cum_sum,
									f'class_{neighborhood_description}': 0,
									f't0_{neighborhood_description}': time_of_first_entrance_in_neighborhood[nc],
									 })
						else:
							relative_measurements.append(
									{'REFERENCE_ID': tid, 'NEIGHBOR_ID': nc,
									'reference_population': reference_population,
									'neighbor_population': neighbor_population,
									'FRAME': t, 'distance': relative_distance[t],
									'velocity': rel_velocity[t], 
									'angle': angle[t] * 180 / np.pi,
									#'angle-neigh-ref': angle[t] * 180 / np.pi, 
									'angular_velocity': angular_velocity[t],
									f'status_{neighborhood_description}': 0,
									f'residence_time_in_{neighborhood_description}': cum_sum,
									f'class_{neighborhood_description}': 0,
									f't0_{neighborhood_description}': time_of_first_entrance_in_neighborhood[nc],
									 })

		df_pairs = pd.DataFrame(relative_measurements)

		return df_pairs

	except KeyError:
		print(f"Neighborhood {description} not found in data frame. Measurements for this neighborhood will not be calculated")


	# try:
	# 	for tid, group in df_reference.groupby('TRACK_ID'):
	# 		# loop over targets in lysis class of interest
	# 		# t0 = ceil(group[target_lysis_time].to_numpy()[0])
	# 		# if t0<=0:
	# 		#     t0 = 5
	# 		t0=0
	# 		# print(type(group))
	# 		# print(group)
	# 		neighbours = group.loc[group['FRAME'] >=t0 , f'{description}'].values  # all neighbours

	# 		#timeline_til_lysis = group.loc[group['FRAME'] <= t0, 'FRAME'].to_numpy()
	# 		timeline = group['FRAME'].to_numpy()

	# #
	# #     pi = group['dead_nuclei_channel_mean'].to_numpy()
	# 		coords = group[['POSITION_X', 'POSITION_Y']].to_numpy()
	# #     target_class = group[target_lysis_class].values[0]
	# #
	# #     # all NK neighbours until target death
	# 		neigh_ids = []
	# 		t0_arrival={}
	# 		t_departure={}
	# 		for t in range(len(timeline)):
	# 			n = neighbours[t]
	# 			all_ids_at_t=[]
	# 			if isinstance(n, float):
	# 					pass
	# 			else:
	# 				for nn in n:
	# 					if nn['id'] not in neigh_ids:
	# 						t0_arrival[nn['id']]=t
	# 					neigh_ids.append(nn['id'])
	# 					all_ids_at_t.append(nn['id'])
	# 				# for id in neigh_ids:
	# 				#     if id not in all_ids_at_t:
	# 				#         if id not in t_departure.keys():
	# 				#             t_departure[id]=t

	# 			#print(neigh_ids)
	# 			#for n in neighbours:
	# 			# if isinstance(n, float):
	# 			#     pass
	# 			#     else:
	# 			#         for i in range(0,len(n)):
	# 			#             print(n[i]['id'])
	# 			#             neigh_ids.append(n[i]['id'])
	# #     for t in range(len(timeline_til_lysis)):
	# #         n = neighbours[t]
	# #         if not n:
	# #             pass
	# #         if isinstance(n, float):
	# #             pass
	# #         elif t > t0:
	# #             continue
	# #         else:
	# #             for nn in n:
	# #                 nk_ids.append(nn['id'])
	# #
	# 		unique_neigh = list(np.unique(neigh_ids))
	# 		print(f'reference cell {tid} : found {len(unique_neigh)} neighbour cells: {unique_neigh}...')
	# 		try:
	# 			cells_neighs = df_neighbor.query(f"TRACK_ID.isin({unique_neigh})")  # locate the NKs of interest in NK table
	# 		except:
	# 			cells_neighs = df_neighbor.query(f"ID.isin({unique_neigh})")
	# 		if 'TRACK_ID' in cells_neighs.columns:
	# 			id_type='TRACK_ID'
	# 		else:
	# 			id_type='ID'

	# 		for nc, group_nc in cells_neighs.groupby(id_type):
	# 			coords_nc = group_nc[['POSITION_X', 'POSITION_Y']].to_numpy()
	# 			# lamp = group_nk['fluo_channel_1_mean'].to_numpy()
	# 			timeline_nc = group_nc['FRAME'].to_numpy()
	# 			#
	# 			# # Perform timeline matching to have same start-end points and no gaps
	# 			full_timeline, index_tc, index_nk = timeline_matching(timeline, timeline_nc)
	# 			relative_distance = np.zeros(len(full_timeline))
	# 			relative_distance[:] = np.nan
	# 			relative_distance_xy1 = np.zeros((len(full_timeline), 2))
	# 			relative_distance_xy1[:, :] = np.nan
	# 			relative_angle1 = np.zeros(len(full_timeline))
	# 			relative_angle1[:] = np.nan
	# 			relative_distance_xy2 = np.zeros((len(full_timeline), 2))
	# 			relative_distance_xy2[:, :] = np.nan
	# 			relative_angle2 = np.zeros(len(full_timeline))
	# 			relative_angle2[:] = np.nan
	# #         # Relative distance
	# 			for t in range(len(relative_distance)):

	# 				if t in timeline and t in timeline_nc:
	# 					idx1 = np.where(timeline == t)[0][0]
	# 					idx2 = np.where(timeline_nc == t)[0][0]
	# 					relative_distance[t] = np.sqrt(
	# 						(coords[idx1, 0] - coords_nc[idx2, 0]) ** 2 + (coords[idx1, 1] - coords_nc[idx2, 1]) ** 2)

	# 					relative_distance_xy1[t, 0] = coords[idx1, 0] - coords_nc[idx2, 0]
	# 					relative_distance_xy1[t, 1] = coords[idx1, 1] - coords_nc[idx2, 1]
	# 					angle1 = np.arctan2(relative_distance_xy1[t, 1], relative_distance_xy1[t, 0]) * 180 / np.pi
	# 					if angle1 < 0:
	# 						angle1 += 360
	# 					relative_angle1[t] = angle1
	# 					relative_distance_xy2[t, 0] = coords_nc[idx2, 0] - coords[idx1, 0]
	# 					relative_distance_xy2[t, 1] = coords_nc[idx2, 1] - coords[idx1, 1]
	# 					angle2 = np.arctan2(relative_distance_xy2[t, 1], relative_distance_xy2[t, 0]) * 180 / np.pi
	# 					if angle2 < 0:
	# 						angle2 += 360
	# 					relative_angle2[t] = angle2
	# 			dddt = derivative(relative_distance, full_timeline, **velocity_kwargs)
	# 			dddt = np.insert(dddt, 0, np.nan)[:-1]
	# 			angular_velocity = np.zeros(len(full_timeline))
	# 			angular_velocity[:] = np.nan

	# 			for t in range(1, len(relative_angle1)):
	# 				if not np.isnan(relative_angle1[t]) and not np.isnan(relative_angle1[t - 1]):
	# 					delta_angle = relative_angle1[t] - relative_angle1[t - 1]
	# 					delta_time = full_timeline[t] - full_timeline[t - 1]
	# 					if delta_time != 0:
	# 						angular_velocity[t] = delta_angle / delta_time
	# #         nk_synapse = group_nk.loc[group_nk['FRAME'] <= ceil(t0), 'live_status'].to_numpy()
	# #         if len(nk_synapse) > 0:
	# #             nk_synapse = int(np.any(nk_synapse.astype(bool)))
	# #         else:
	# #             nk_synapse = 0
	# #
	# 			neighb_dist=float(neigh_dist)
	# 			rel_dist_til_lysis = relative_distance[:ceil(t0) + 1]
	# 			duration_in_neigh = len(rel_dist_til_lysis[rel_dist_til_lysis <= neighb_dist]) / (ceil(t0) + 1)
	# #         if target_classes[0] != 1.:
	# #
	# 			t_low = max(ceil(t0) - pre_lysis_time_window, 0)
	# 			t_high = ceil(t0) + 1
	# #
	# 			rel_dist_crop = relative_distance[t_low:t_high]
	# 			rel_v_crop = dddt[t_low:t_high]
	# #
	# #             t_high_lamp = min(ceil(t0) + 1 + pre_lysis_time_window, df_targets['FRAME'].max())
	# #             nk_lamp_crop = lamp[t_low:t_high_lamp]

	# 			if len(rel_dist_crop[rel_dist_crop == rel_dist_crop]) > 0:
	# 				pre_lysis_d_rel = np.nanmean(rel_dist_crop)
	# 			else:
	# 				pre_lysis_d_rel = np.nanmean(relative_distance[:])
	# 			if len(rel_v_crop[rel_v_crop == rel_v_crop]) > 0:
	# 				pre_lysis_v_rel = np.nanmean(rel_v_crop)
	# 			else:
	# 				pre_lysis_v_rel = np.nanmean(dddt[:])
	# #
	# #             if len(nk_lamp_crop[nk_lamp_crop == nk_lamp_crop]) > 0:
	# #                 nk_lamp = np.nanmean(nk_lamp_crop)
	# #             else:
	# #                 nk_lamp=np.nanmean(lamp[:])
	# #
	# #             syn_class = nk_synapse
	# #
	# 		# else:
	# 		#     pre_lysis_d_rel = np.nanmean(relative_distance[:])
	# 		#     pre_lysis_v_rel = np.nanmean(dddt[:])
	# 		#     #syn_class = np.amax(nk_synapse[:])
	# 		#     #nk_lamp = np.nanmean(lamp[:])
	# 			pts.append({'rc': tid, 'lysis_time': t0, 'nc': nc, 'drel': pre_lysis_d_rel, 'vrel': pre_lysis_v_rel,
	# 					 'relxy': relative_distance_xy1,
	# 					't_residence_rel': duration_in_neigh})
	# 	# pts.append({'rc': tid, 'lysis_time': t0, 'nc': nc, 'drel': pre_lysis_d_rel, 'vrel': pre_lysis_v_rel,
	# 	#             'syn_class': syn_class, 'lamp1': nk_lamp, 'relxy': relative_distance_xy1,
	# 	#             't_residence_rel': duration_in_neigh})
	# 			for t in range(len(relative_distance)):
	# 				if t >= t0_arrival[nc]:
	# 					df_rel.append(
	# 							{'REFERENCE_ID': tid, 'NEIGHBOR_ID': nc, 'FRAME': t, 'distance': relative_distance[t],
	# 							 'velocity': dddt[t], f't0_{description}': t0_arrival[nc],
	# 							 'angle_tc-eff': relative_angle1[t],
	# 							 'angle-eff-tc': relative_angle2[t], 'angular_velocity': angular_velocity[t],
	# 							 f'status_{description}': 1,f'class_{description}': 0})
	# 				else:
	# 					df_rel.append(
	# 							{'REFERENCE_ID': tid, 'NEIGHBOR_ID': nc, 'FRAME': t, 'distance': relative_distance[t],
	# 							 'velocity': dddt[t], f't0_{description}': t0_arrival[nc],
	# 							 'angle_tc-eff': relative_angle1[t],
	# 							 'angle-eff-tc': relative_angle2[t], 'angular_velocity': angular_velocity[t],
	# 							 f'status_{description}': 0,f'class_{description}': 0})

	# 				# if nc in t_departure:
	# 				#     if t_departure[nc] > t >= t0_arrival[nc]:
	# 				#         df_rel.append(
	# 				#             {'REFERENCE_ID': tid, 'NEIGHBOR_ID': nc, 'FRAME': t, 'distance': relative_distance[t],
	# 				#              'velocity': dddt[t], f't0_{description}': t0_arrival[nc],
	# 				#              f't1_{description}': t_departure[nc], 'angle_tc-eff': relative_angle1[t],
	# 				#              'angle-eff-tc': relative_angle2[t], 'angular_velocity': angular_velocity[t],
	# 				#              f'status_{description}': 1})
	# 				#     else:
	# 				#
	# 				#         df_rel.append(
	# 				#             {'REFERENCE_ID': tid, 'NEIGHBOR_ID': nc, 'FRAME': t, 'distance': relative_distance[t],
	# 				#              'velocity': dddt[t], f't0_{description}': t0_arrival[nc],
	# 				#              f't1_{description}': t_departure[nc], 'angle_tc-eff': relative_angle1[t],
	# 				#              'angle-eff-tc': relative_angle2[t], 'angular_velocity': angular_velocity[t],
	# 				#              f'status_{description}': 0})
	# 				# else:
	# 				#     if t >= t0_arrival[nc]:
	# 				#
	# 				#         df_rel.append(
	# 				#             {'REFERENCE_ID': tid, 'NEIGHBOR_ID': nc, 'FRAME': t, 'distance': relative_distance[t],
	# 				#              'velocity': dddt[t], f't0_{description}': t0_arrival[nc],
	# 				#              f't1_{description}': -1, 'angle_tc-eff': relative_angle1[t],
	# 				#              'angle-eff-tc': relative_angle2[t], 'angular_velocity': angular_velocity[t],
	# 				#              f'status_{description}': 1})
	# 				#     else:
	# 				#         df_rel.append(
	# 				#             {'REFERENCE_ID': tid, 'NEIGHBOR_ID': nc, 'FRAME': t, 'distance': relative_distance[t],
	# 				#              'velocity': dddt[t], f't0_{description}': t0_arrival[nc],
	# 				#              f't1_{description}': -1, 'angle_tc-eff': relative_angle1[t],
	# 				#              'angle-eff-tc': relative_angle2[t], 'angular_velocity': angular_velocity[t],
	# 				#              f'status_{description}': 0})

	# 			# for t in range(len(relative_distance)):
	# 			#     df_rel.append({'TARGET_ID': tid, 'EFFECTOR_ID': nc, 'FRAME': t, 'distance': relative_distance[t],
	# 			#                    'velocity': dddt[t], 't0_lysis': t0, 'angle_tc-eff': relative_angle1[t],
	# 			#                    'angle-eff-tc': relative_angle2[t],'probability':0,'angular_velocity': angular_velocity[t]})
	# 	pts = pd.DataFrame(pts)
	# # #probs = probabilities(pts)

	# 	df_rel = pd.DataFrame(df_rel)
	# 	# for index,row in pts.iterrows():
	# 	#     df_rel.loc[(df_rel['REFERENCE_ID'] == row['rc']) & (df_rel['NEIGHBOR_ID'] == row['nc']), 'distance_mean'] = row[
	# 	#         'drel']
	# 	#     df_rel.loc[(df_rel['REFERENCE_ID'] == row['rc']) & (df_rel['NEIGHBOR_ID'] == row['nc']), 'velocity_mean'] = row[
	# 	#         'vrel']
	# 	#     df_rel.loc[(df_rel['REFERENCE_ID'] == row['rc']) & (df_rel['NEIGHBOR_ID'] == row['nc']), 't_residence_rel'] = row[
	# 	#         't_residence_rel']
	# # #for prob in probs:
	# #     #for index,row in prob.iterrows():
	# #         #df_rel.loc[(df_rel['TARGET_ID'] == row['tc']) & (df_rel['EFFECTOR_ID'] == row['nk']),'probability']=row['total_prob']
	# #
	# #
	# 	return df_rel
	# except KeyError:
	# 	print(f"Neighborhood {description} not found in data frame. Measurements for this neighborhood will not be calculated")


def timeline_matching(timeline1, timeline2):

	"""
	Match two timelines and create a unified timeline with corresponding indices.

	Parameters
	----------
	timeline1 : array-like
		The first timeline to be matched.
	timeline2 : array-like
		The second timeline to be matched.

	Returns
	-------
	tuple
		A tuple containing:
		- full_timeline : numpy.ndarray
			The unified timeline spanning from the minimum to the maximum time point in the input timelines.
		- index1 : list of int
			The indices of `timeline1` in the `full_timeline`.
		- index2 : list of int
			The indices of `timeline2` in the `full_timeline`.

	Examples
	--------
	>>> timeline1 = [1, 2, 5, 6]
	>>> timeline2 = [2, 3, 4, 6]
	>>> full_timeline, index1, index2 = timeline_matching(timeline1, timeline2)
	>>> print(full_timeline)
	[1 2 3 4 5 6]
	>>> print(index1)
	[0, 1, 4, 5]
	>>> print(index2)
	[1, 2, 3, 5]

	Notes
	-----
	- The function combines the two timelines and generates a continuous range from the minimum to the maximum time point.
	- It then finds the indices of the original timelines in this unified timeline.
	- The function assumes that the input timelines consist of integer values.
	"""

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

def extract_neighborhoods_from_pickles(pos):

	"""
	Extract neighborhood protocols from pickle files located at a given position.

	Parameters
	----------
	pos : str
		The base directory path where the pickle files are located.

	Returns
	-------
	list of dict
		A list of dictionaries, each containing a neighborhood protocol. Each dictionary has the keys:
		- 'reference' : str
			The reference population ('targets' or 'effectors').
		- 'neighbor' : str
			The neighbor population.
		- 'type' : str
			The type of neighborhood ('circle' or 'contact').
		- 'distance' : float
			The distance parameter for the neighborhood.
		- 'description' : str
			The original neighborhood string.

	Notes
	-----
	- The function checks for the existence of pickle files containing target and effector trajectory data.
	- If the files exist, it loads the data and extracts columns that start with 'neighborhood'.
	- The neighborhood settings are extracted using the `extract_neighborhood_settings` function.
	- The function assumes the presence of subdirectories 'output/tables' under the provided `pos`.

	Examples
	--------
	>>> protocols = extract_neighborhoods_from_pickles('/path/to/data')
	>>> for protocol in protocols:
	>>>     print(protocol)
	{'reference': 'targets', 'neighbor': 'targets', 'type': 'contact', 'distance': 5.0, 'description': 'neighborhood_self_contact_5_px'}
	"""

	tab_tc = pos + os.sep.join(['output', 'tables', 'trajectories_targets.pkl'])
	if os.path.exists(tab_tc):
		df_targets = np.load(tab_tc, allow_pickle=True)
	else:
		df_targets = None
	if os.path.exists(tab_tc.replace('targets','effectors')):
		df_effectors = np.load(tab_tc.replace('targets','effectors'), allow_pickle=True)
	else:
		df_effectors = None

	neighborhood_protocols=[]

	if df_targets is not None:
		for column in list(df_targets.columns):
			if column.startswith('neighborhood'):
				neigh_protocol = extract_neighborhood_settings(column, population='targets')
				neighborhood_protocols.append(neigh_protocol)

	if df_effectors is not None:
		for column in list(df_effectors.columns):
			if column.startswith('neighborhood'):
				neigh_protocol = extract_neighborhood_settings(column, population='effectors')
				neighborhood_protocols.append(neigh_protocol)

	return neighborhood_protocols

def extract_neighborhood_settings(neigh_string, population='targets'):
	
	"""
	Extract neighborhood settings from a given string.

	Parameters
	----------
	neigh_string : str
		The string describing the neighborhood settings. Must start with 'neighborhood'.
	population : str, optional
		The population type ('targets' by default). Can be either 'targets' or 'effectors'.

	Returns
	-------
	dict
		A dictionary containing the neighborhood protocol with keys:
		- 'reference' : str
			The reference population.
		- 'neighbor' : str
			The neighbor population.
		- 'type' : str
			The type of neighborhood ('circle' or 'contact').
		- 'distance' : float
			The distance parameter for the neighborhood.
		- 'description' : str
			The original neighborhood string.

	Raises
	------
	AssertionError
		If the `neigh_string` does not start with 'neighborhood'.

	Notes
	-----
	- The function determines the neighbor population based on the given population.
	- The neighborhood type and distance are extracted from the `neigh_string`.
	- The description field in the returned dictionary contains the original neighborhood string.

	Examples
	--------
	>>> extract_neighborhood_settings('neighborhood_self_contact_5_px', 'targets')
	{'reference': 'targets', 'neighbor': 'targets', 'type': 'contact', 'distance': 5.0, 'description': 'neighborhood_self_contact_5_px'}
	"""

	assert neigh_string.startswith('neighborhood')
	if population=='targets':
		neighbor_population = 'effectors'
	elif population=='effectors':
		neighbor_population = 'targets'

	if 'self' in neigh_string:
		
		if 'circle' in neigh_string:
			
			distance = float(neigh_string.split('circle_')[1].replace('_px',''))
			neigh_protocol = {'reference': population,'neighbor': population,'type':'circle','distance':distance,'description': neigh_string}
		elif 'contact' in neigh_string:
			distance=float(neigh_string.split('contact_')[1].replace('_px',''))
			neigh_protocol = {'reference': population,'neighbor': population,'type':'contact','distance':distance,'description': neigh_string}
	else:

		if 'circle' in neigh_string:

			distance=float(neigh_string.split('circle_')[1].replace('_px',''))
			neigh_protocol = {'reference': population,'neighbor': neighbor_population,'type':'circle','distance':distance,'description': neigh_string}
		elif 'contact' in neigh_string:
			
			distance=float(neigh_string.split('contact_')[1].replace('_px',''))
			neigh_protocol = {'reference': population,'neighbor': neighbor_population,'type':'contact','distance':distance,'description': neigh_string}

	return neigh_protocol



