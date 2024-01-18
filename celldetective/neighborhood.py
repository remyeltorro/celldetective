import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import regionprops_table
from functools import reduce
from mahotas.features import haralick
from scipy.ndimage import zoom
import os
import subprocess
from celldetective.utils import rename_intensity_column, create_patch_mask, remove_redundant_features
from celldetective.io import get_position_table
from scipy.spatial.distance import cdist
import re

abs_path = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], 'celldetective'])


def set_live_status(setA,setB,status, not_status_option):
	"""

	Match neighbors in set A and B within a circle of radius d. 

	Parameters
	----------
	setA,setB : pandas DataFrame
		Trajectory or position sets A and B.
	status : list or None
		status columns for the cells to keep in set A and B. 0 is remove, 1 is keep. 

	"""

	if status is None:
		setA.loc[:,'live_status'] = 1
		setB.loc[:,'live_status'] = 1
		status = ['live_status', 'live_status']
	elif isinstance(status,list):
		assert len(status)==2,'Please provide only two columns to classify cells as alive or dead.'
		if status[0] is None:
			setA.loc[:,'live_status'] = 1
			status[0] = 'live_status'
		elif status[0] is not None and isinstance(not_status_option,list):
			setA.loc[setA[status[0]]==2,status[0]] = 1 #already happened events become event
			if not_status_option[0]:
				setA.loc[:,'not_'+status[0]] = [not a if a==0 or a==1 else np.nan for a in setA.loc[:,status[0]].values]
				status[0] = 'not_'+status[0]
		if status[1] is None:
			setB.loc[:,'live_status'] = 1
			status[1] = 'live_status'
		elif status[1] is not None and isinstance(not_status_option,list):
			setB.loc[setB[status[1]]==2,status[1]] = 1 #already happened events become event
			if not_status_option[1]:
				setB.loc[:,'not_'+status[1]] = [not a if a==0 or a==1 else np.nan for a in setB.loc[:,status[1]].values]
				status[1] = 'not_'+status[1]

		assert status[0] in list(setA.columns)
		assert status[1] in list(setB.columns)
	
	setA = setA.reset_index(drop=True)
	setB = setB.reset_index(drop=True)	

	return setA, setB, status

def compute_attention_weight(dist_matrix, cut_distance, opposite_cell_status, opposite_cell_ids, axis=1, include_dead_weight=True):
	
	weights = np.empty(dist_matrix.shape[axis])
	closest_opposite = np.empty(dist_matrix.shape[axis])

	for i in range(dist_matrix.shape[axis]):
		if axis==1:
			row = dist_matrix[:,i]
		elif axis==0:
			row = dist_matrix[i,:]
		row[row==0.] = 1.0E06
		nbr_opposite = len(row[row<=cut_distance])
		
		if not include_dead_weight:
			stat = opposite_cell_status[np.where(row<=cut_distance)[0]]
			nbr_opposite = len(stat[stat==1])
			index_subpop = np.argmin(row[opposite_cell_status==1])
			closest_opposite[i] = opposite_cell_ids[opposite_cell_status==1][index_subpop]
		else:
			closest_opposite[i] = opposite_cell_ids[np.argmin(row)]
		
		if nbr_opposite>0:
			weight = 1./float(nbr_opposite)
			weights[i] = weight

	return weights, closest_opposite

def distance_cut_neighborhood(setA, setB, distance, mode='two-pop', status=None, not_status_option=None, compute_cum_sum=True, 
							  attention_weight=True, symmetrize=True, include_dead_weight=True,
							  column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}):
	
	"""

	Match neighbors in set A and B within a circle of radius d. 

	Parameters
	----------
	setA,setB : pandas DataFrame
		Trajectory or position sets A and B.
	distance : float
		Cut-distance in pixels to match neighboring pairs.
	mode: str
		neighboring mode, between 'two-pop' (e.g. target-effector) and 'self' (target-target or effector-effector).
	status: None or status 
		name to look for cells to ignore (because they are dead). By default all cells are kept.
	compute_cum_sum: bool,
		compute cumulated time of presence of neighbours (only if trajectories available for both sets)
	attention_weight: bool,
		compute the attention weight (how much a cell of set B is shared across cells of set A)
	symmetrize: bool,
		write in set B the neighborhood of set A
	include_dead_weight: bool
		do not count dead cells when establishing attention weight
	"""

	# Check live_status option
	setA, setB, status = set_live_status(setA, setB, status, not_status_option)

	# Check distance option 
	if not isinstance(distance, list):
		distance = [distance]
	
	for d in distance:
		# loop over each provided distance
		
		if mode=='two-pop':
			neigh_col = f'neighborhood_2_circle_{d}_px'
		elif mode=='self':
			neigh_col = f'neighborhood_self_circle_{d}_px'
			
		cl = []
		for s in [setA,setB]:

			# Check whether data can be tracked
			temp_column_labels = column_labels.copy()

			if not 'TRACK_ID' in list(s.columns):
				temp_column_labels.update({'track': 'ID'})
				compute_cum_sum = False # if no tracking data then cum_sum is not relevant
			cl.append(temp_column_labels)

			# Remove nan tracks (cells that do not belong to a track)
			s[neigh_col] = np.nan
			s[neigh_col] = s[neigh_col].astype(object)
			s.dropna(subset=[cl[-1]['track']],inplace=True)

		# Loop over each available timestep
		timeline = np.unique(np.concatenate([setA[cl[0]['time']].to_numpy(), setB[cl[1]['time']].to_numpy()])).astype(int)
		for t in tqdm(timeline):

			index_A = list(setA.loc[setA[cl[0]['time']]==t].index)
			coordinates_A = setA.loc[setA[cl[0]['time']]==t,[cl[0]['x'], cl[0]['y']]].to_numpy()            
			ids_A = setA.loc[setA[cl[0]['time']]==t,cl[0]['track']].to_numpy()
			status_A = setA.loc[setA[cl[0]['time']]==t,status[0]].to_numpy()

			index_B = list(setB.loc[setB[cl[1]['time']]==t].index)            
			coordinates_B = setB.loc[setB[cl[1]['time']]==t,[cl[1]['x'], cl[1]['y']]].to_numpy()
			ids_B = setB.loc[setB[cl[1]['time']]==t,cl[1]['track']].to_numpy()
			status_B = setB.loc[setB[cl[1]['time']]==t,status[1]].to_numpy()

			if len(ids_A) > 0 and len(ids_B) > 0:
				
				# compute distance matrix
				dist_map = cdist(coordinates_A, coordinates_B, metric="euclidean")
				
				if attention_weight:
					weights, closest_A = compute_attention_weight(dist_map, d, status_A, ids_A, axis=1, include_dead_weight=include_dead_weight)
				
				# Target centric
				for k in range(dist_map.shape[0]):
					
					col = dist_map[k,:]
					col[col==0.] = 1.0E06

					neighs_B = np.array([ids_B[i] for i in np.where((col<=d))[0]])
					status_neigh_B = np.array([status_B[i] for i in np.where((col<=d))[0]])
					print(status_neigh_B)
					dist_B = [round(col[i],2) for i in np.where((col<=d))[0]]
					if len(dist_B)>0:
						closest_B_cell = neighs_B[np.argmin(dist_B)]    
					
					if symmetrize and attention_weight:
						n_neighs = float(len(neighs_B))
						if not include_dead_weight:
							n_neighs_alive = len(np.where(status_neigh_B==1)[0])
							neigh_count = n_neighs_alive
						else:
							neigh_count = n_neighs
						if neigh_count>0:
							weight_A = 1./neigh_count
						else:
							weight_A = np.nan

						if not include_dead_weight and status_A[k]==0:
							weight_A = 0
					
					neighs = []
					setA.at[index_A[k], neigh_col] = []
					for n in range(len(neighs_B)):
						
						# index in setB
						n_index = np.where(ids_B==neighs_B[n])[0][0]
						# Assess if neigh B is closest to A
						if attention_weight:
							if closest_A[n_index]==ids_A[k]:
								closest = True
							else:
								closest = False
						
						if symmetrize:
							# Load neighborhood previous data
							sym_neigh = setB.loc[index_B[n_index], neigh_col]
							if neighs_B[n]==closest_B_cell:
								closest_b=True
							else:
								closest_b=False
							if isinstance(sym_neigh, list):
								sym_neigh.append({'id': ids_A[k], 'distance': dist_B[n], 'status': status_A[k]})
							else:
								sym_neigh = [{'id': ids_A[k], 'distance': dist_B[n],'status': status_A[k]}]
							if attention_weight:
								sym_neigh[-1].update({'weight': weight_A, 'closest': closest_b})
						
						# Write the minimum info about neighborhing cell B
						neigh_dico = {'id': neighs_B[n], 'distance': dist_B[n], 'status': status_neigh_B[n]}
						if attention_weight:
							neigh_dico.update({'weight': weights[n_index], 'closest': closest})

						if compute_cum_sum:
							# Compute the integrated presence of the neighboring cell B
							assert cl[1]['track'] == 'TRACK_ID','The set B does not seem to contain tracked data. The cumulative time will be meaningless.'
							past_neighs = [[ll['id'] for ll in l] if len(l)>0 else [None] for l in setA.loc[(setA[cl[0]['track']]==ids_A[k])&(setA[cl[0]['time']]<=t), neigh_col].to_numpy()]
							past_neighs = [item for sublist in past_neighs for item in sublist]
							
							if attention_weight:
								past_weights = [[ll['weight'] for ll in l] if len(l)>0 else [None] for l in setA.loc[(setA[cl[0]['track']]==ids_A[k])&(setA[cl[0]['time']]<=t), neigh_col].to_numpy()]
								past_weights = [item for sublist in past_weights for item in sublist]

							cum_sum = len(np.where(past_neighs==neighs_B[n])[0])
							neigh_dico.update({'cumulated_presence': cum_sum+1})
							
							if attention_weight:
								cum_sum_weighted = np.sum([w if l==neighs_B[n] else 0 for l,w in zip(past_neighs, past_weights)])
								neigh_dico.update({'cumulated_presence_weighted': cum_sum_weighted + weights[n_index]})

						if symmetrize:
							setB.at[index_B[n_index], neigh_col] = sym_neigh
						
						neighs.append(neigh_dico)
											
					setA.at[index_A[k], neigh_col] = neighs
		
	return setA, setB

def compute_neighborhood_at_position(pos, distance, population=['targets','effectors'], theta_dist=None, img_shape=(2048,2048), return_tables=False, clear_neigh=False, event_time_col=None,
	neighborhood_kwargs={'mode': 'two-pop','status': None, 'not_status_option': None,'include_dead_weight': True,"compute_cum_sum": False,"attention_weight": True, 'symmetrize': True}):
	
	pos = pos.replace('\\','/')
	pos = rf"{pos}"
	assert os.path.exists(pos),f'Position {pos} is not a valid path.'

	if isinstance(population, str):
		population = [population, population]

	if not isinstance(distance, list):
		distance = [distance]
	if not theta_dist is None and not isinstance(theta_dist, list):
		theta_dist = [theta_dist]

	if theta_dist is None:
		theta_dist = [0.9*d for d in distance]
	assert len(theta_dist)==len(distance),'Incompatible number of distances and number of edge thresholds.'

	if population[0]==population[1]:
		neighborhood_kwargs.update({'mode': 'self'})
	if population[1]!=population[0]:
		neighborhood_kwargs.update({'mode': 'two-pop'})

	df_A, path_A = get_position_table(pos, population=population[0], return_path=True)
	df_B, path_B = get_position_table(pos, population=population[1], return_path=True)

	if clear_neigh:
		unwanted = df_A.columns[df_A.columns.str.contains('neighborhood')]
		df_A = df_A.drop(columns=unwanted)
		unwanted = df_B.columns[df_B.columns.str.contains('neighborhood')]
		df_B = df_B.drop(columns=unwanted)		

	df_A, df_B = distance_cut_neighborhood(df_A,df_B, distance,**neighborhood_kwargs)

	for td,d in zip(theta_dist, distance):

		if neighborhood_kwargs['mode']=='two-pop':
			neigh_col = f'neighborhood_2_circle_{d}_px'
		elif neighborhood_kwargs['mode']=='self':
			neigh_col = f'neighborhood_self_circle_{d}_px'

		edge_filter_A = (df_A['POSITION_X'] > td)&(df_A['POSITION_Y'] > td)&(df_A['POSITION_Y'] < (img_shape[0] - td))&(df_A['POSITION_X'] < (img_shape[1] - td))
		edge_filter_B = (df_B['POSITION_X'] > td)&(df_B['POSITION_Y'] > td)&(df_B['POSITION_Y'] < (img_shape[0] - td))&(df_B['POSITION_X'] < (img_shape[1] - td))
		df_A.loc[~edge_filter_A, neigh_col] = np.nan
		df_B.loc[~edge_filter_B, neigh_col] = np.nan

		df_A = compute_neighborhood_metrics(df_A, neigh_col, metrics=['inclusive','exclusive','intermediate'], decompose_by_status=True)
		if neighborhood_kwargs['symmetrize']:
			df_B = compute_neighborhood_metrics(df_B, neigh_col, metrics=['inclusive','exclusive','intermediate'], decompose_by_status=True)
		
		df_A = mean_neighborhood_before_event(df_A, neigh_col, event_time_col)

	df_A.to_pickle(path_A.replace('.csv','.pkl'))
	if not population[0]==population[1]:
		df_B.to_pickle(path_B.replace('.csv','.pkl'))

	unwanted = df_A.columns[df_A.columns.str.startswith('neighborhood_')]
	df_A2 = df_A.drop(columns=unwanted)
	df_A2.to_csv(path_A, index=False)

	if not population[0]==population[1]:
		unwanted = df_B.columns[df_B.columns.str.startswith('neighborhood_')]
		df_B_csv = df_B.drop(unwanted, axis=1, inplace=False)
		df_B_csv.to_csv(path_B,index=False)

	if return_tables:
		return df_A, df_B

def compute_neighborhood_metrics(neigh_table, neigh_col, metrics=['inclusive','exclusive','intermediate'], decompose_by_status=False):

	neigh_table = neigh_table.reset_index(drop=True)
	if 'position' in list(neigh_table.columns):
		groupbycols = ['position','TRACK_ID']
	else:
		groupbycols = ['TRACK_ID']
	neigh_table.sort_values(by=groupbycols+['FRAME'],inplace=True)

	for tid,group in neigh_table.groupby(groupbycols):
		group = group.dropna(subset=neigh_col)
		indices = list(group.index)
		neighbors = group[neigh_col].to_numpy()

		if 'inclusive' in metrics:
			n_inclusive = [len(n) for n in neighbors]

		if 'intermediate' in metrics:
			n_intermediate = np.zeros(len(neighbors))
			n_intermediate[:] = np.nan

		if 'exclusive' in metrics:
			n_exclusive = np.zeros(len(neighbors))
			n_exclusive[:] = np.nan

		if decompose_by_status:

			if 'inclusive' in metrics:
				n_inclusive_status_0 = np.zeros(len(neighbors))
				n_inclusive_status_0[:] = np.nan
				n_inclusive_status_1 = np.zeros(len(neighbors))
				n_inclusive_status_1[:] = np.nan

			if 'intermediate' in metrics:
				n_intermediate_status_0 = np.zeros(len(neighbors))
				n_intermediate_status_0[:] = np.nan
				n_intermediate_status_1 = np.zeros(len(neighbors))
				n_intermediate_status_1[:] = np.nan

			if 'exclusive' in metrics:
				n_exclusive_status_0 = np.zeros(len(neighbors))
				n_exclusive_status_0[:] = np.nan
				n_exclusive_status_1 = np.zeros(len(neighbors))
				n_exclusive_status_1[:] = np.nan

		for t in range(len(neighbors)):

			neighs_at_t = neighbors[t]
			weights_at_t = [n['weight'] for n in neighs_at_t]
			status_at_t = [n['status'] for n in neighs_at_t]
			closest_at_t = [n['closest'] for n in neighs_at_t]

			if 'intermediate' in metrics:
				n_intermediate[t] = np.sum(weights_at_t)
			if 'exclusive' in metrics:
				n_exclusive[t] = sum([c==1.0 for c in closest_at_t])

			if decompose_by_status:

				if 'inclusive' in metrics:
					n_inclusive_status_0[t] = sum([s==0.0 for s in status_at_t])
					n_inclusive_status_1[t] = sum([s==1.0 for s in status_at_t])

				if 'intermediate' in metrics:
					weights_at_t = np.array(weights_at_t)
					
					# intermediate
					weights_status_1 = weights_at_t[np.array([s==1.0 for s in status_at_t],dtype=bool)]
					weights_status_0 = weights_at_t[np.array([s==0.0 for s in status_at_t],dtype=bool)]
					n_intermediate_status_1[t] = np.sum(weights_status_1)
					n_intermediate_status_0[t] = np.sum(weights_status_0)

				if 'exclusive' in metrics:
					n_exclusive_status_0[t] = sum([c==1.0 if s==0.0 else False for c,s in zip(closest_at_t,status_at_t)])
					n_exclusive_status_1[t] = sum([c==1.0 if s==1.0 else False for c,s in zip(closest_at_t,status_at_t)])

		if 'inclusive' in metrics:
			neigh_table.loc[indices, 'inclusive_count_'+neigh_col] = n_inclusive
		if 'intermediate' in metrics:
			neigh_table.loc[indices, 'intermediate_count_'+neigh_col] = n_intermediate
		if 'exclusive' in metrics:
			neigh_table.loc[indices, 'exclusive_count_'+neigh_col] = n_exclusive

		if decompose_by_status:
			if 'inclusive' in metrics:
				neigh_table.loc[indices, 'inclusive_count_s0_'+neigh_col] = n_inclusive_status_0
				neigh_table.loc[indices, 'inclusive_count_s1_'+neigh_col] = n_inclusive_status_1
			if 'intermediate' in metrics:
				neigh_table.loc[indices, 'intermediate_count_s0_'+neigh_col] = n_intermediate_status_0
				neigh_table.loc[indices, 'intermediate_count_s1_'+neigh_col] = n_intermediate_status_1
			if 'exclusive' in metrics:	
				neigh_table.loc[indices, 'exclusive_count_s0_'+neigh_col] = n_exclusive_status_0
				neigh_table.loc[indices, 'exclusive_count_s1_'+neigh_col] = n_exclusive_status_1

	return neigh_table

def mean_neighborhood_before_event(neigh_table, neigh_col, event_time_col):
	
	if 'position' in list(neigh_table.columns):
		groupbycols = ['position','TRACK_ID']
	else:
		groupbycols = ['TRACK_ID']
	neigh_table.sort_values(by=groupbycols+['FRAME'],inplace=True)
	
	if event_time_col is None:
		neigh_table.loc[:,'event_time_temp'] = neigh_table['FRAME'].max()
		event_time_col = 'event_time_temp'
	
	for tid,group in neigh_table.groupby(groupbycols):

		group = group.dropna(subset=neigh_col)
		indices = list(group.index)

		event_time_values = group[event_time_col].to_numpy()
		if len(event_time_values)>0:
			event_time = event_time_values[0]
		else:
			continue

		if event_time<0.:
			event_time = group['FRAME'].max()

		valid_counts_intermediate = group.loc[group['FRAME']<=event_time,'intermediate_count_s1_'+neigh_col].to_numpy()
		valid_counts_inclusive = group.loc[group['FRAME']<=event_time,'inclusive_count_s1_'+neigh_col].to_numpy()
		valid_counts_exclusive = group.loc[group['FRAME']<=event_time,'exclusive_count_s1_'+neigh_col].to_numpy()

		if len(valid_counts_intermediate[valid_counts_intermediate==valid_counts_intermediate])>0:
			neigh_table.loc[indices, f'mean_count_intermediate_{neigh_col}_before_event'] = np.nanmean(valid_counts_intermediate)
		if len(valid_counts_inclusive[valid_counts_inclusive==valid_counts_inclusive])>0:
			neigh_table.loc[indices, f'mean_count_inclusive_{neigh_col}_before_event'] = np.nanmean(valid_counts_inclusive)
		if len(valid_counts_exclusive[valid_counts_exclusive==valid_counts_exclusive])>0:
			neigh_table.loc[indices, f'mean_count_exclusive_{neigh_col}_before_event'] = np.nanmean(valid_counts_exclusive)

	if event_time_col=='event_time_temp':
		neigh_table = neigh_table.drop(columns='event_time_temp')
	return neigh_table

# def mask_intersection_neighborhood(setA, labelsA, setB, labelsB, threshold_iou=0.5, viewpoint='B'):
# 	# do whatever to match objects in A and B
# 	return setA, setB

if __name__ == "__main__":

	print('None')
	pos = "/home/torro/Documents/Experiments/NKratio_Exp/W5/500"
	
	test,_ = compute_neighborhood_at_position(pos, [62], population=['targets','effectors'], theta_dist=None, img_shape=(2048,2048), return_tables=True, clear_neigh=True,
	neighborhood_kwargs={'mode': 'two-pop','status': ['class', None],'not_status_option': [True, False],'include_dead_weight': True,"compute_cum_sum": False,"attention_weight": True, 'symmetrize': False})
	
	#test = compute_neighborhood_metrics(test, 'neighborhood_self_circle_150_px', metrics=['inclusive','exclusive','intermediate'], decompose_by_status=True)
	print(test.columns)
	#print(segment(None,'test'))
