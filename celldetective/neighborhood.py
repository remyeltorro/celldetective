import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import regionprops_table
from scipy.ndimage.morphology import distance_transform_edt
from functools import reduce
from mahotas.features import haralick
from scipy.ndimage import zoom
import os
import subprocess
from celldetective.utils import rename_intensity_column, create_patch_mask, remove_redundant_features

abs_path = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], 'celldetective'])


def set_live_status(setA,setB,status):
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
		if status[1] is None:
			setB.loc[:,'live_status'] = 1
			status[1] = 'live_status'
			
		assert status[0] in list(setA.columns)
		assert status[1] in list(setB.columns)
		
	setA = setA.reset_index(drop=True)
	setB = setB.reset_index(drop=True)	

	return setA, setB

def compute_attention_weight(dist_matrix, opposite_cell_status, opposite_cell_ids, axis=1, include_dead_weight=True):
	
	weights = np.empty(dist_matrix.shape[axis])
	closest_opposite = np.empty(dist_matrix.shape[axis])

	for i in range(dist_matrix.shape[axis]):
		if axis==1:
			row = dist_matrix[:,i]
		elif axis==0:
			row = dist_matrix[i,:]
		nbr_opposite = len(row[row<=d])
		
		if not include_dead_weight:
			stat = opposite_cell_status[np.where(row<=d)[0]]
			nbr_opposite = len(stat[stat==1])
			index_subpop = np.argmin(row[opposite_cell_status==1])
			closest_opposite[i] = opposite_cell_ids[opposite_cell_status==1][index_subpop]
		else:
			closest_opposite[i] = opposite_cell_ids[np.argmin(row)]
		
		if nbr_opposite>0:
			weight = 1./float(nbr_opposite)
			weights[i] = weight

	return weights, closest

def distance_cut_neighborhood(setA, setB, distance, mode='two-pop', status=None, compute_cum_sum=True, 
							  attention_weight=True, symmetrize=True, include_dead_weight=False,
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
	setA, setB = set_live_status(setA, setB, status)
	
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
			if not 'TRACK_ID' in s.columns:
				column_labels.update({'track': 'ID'})
				compute_cum_sum = False # if no tracking data then cum_sum is not relevant
			cl.append(column_labels)

			# Remove nan tracks (cells that do not belong to a track)
			s[neigh_col] = np.nan
			s[neigh_col] = s[neigh_col].astype(object)
			s.dropna(subset=[cl[-1]['track']],inplace=True)


		# Loop over each available timestep
		timeline = np.unique(np.concatenate([setA[cl[0]['time']].to_numpy(), setB[cl[1]['time']].to_numpy()])).astype(int)
		for t in timeline:

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
					weights, closest_A = compute_attention_weight(dist_map, status_A, ids_A, axis=1, include_dead_weight=include_dead_weight)
				
				# Target centric
				for k in range(dist_map.shape[0]):
					
					col = dist_map[k,:]
					neighs_B = np.array([ids_B[i] for i in np.where(col<=d)[0]])
					status_neigh_B = np.array([status_B[i] for i in np.where(col<=d)[0]])
					dist_B = [round(col[i],2) for i in np.where(col<=d)[0]]
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
						neigh_dico = {'id': neighs_B[n], 'distance': dist_B[n], 'status': status_B[n]}
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

def compute_neighborhood_at_position(pos, distance, population=['targets','effectors'], theta_dist=None, img_shape=(2048,2048), return_tables=False,
	neighborhood_kwargs={'mode': 'two-pop','status': None,'include_dead_weight': True,"compute_cum_sum": False,"attention_weight": True, 'symmetrize': True}):
	
	pos = pos.replace('\\','/')
	pos = rf"{pos}"
	assert os.path.exists(pos),f'Position {pos} is not a valid path.'

	if isinstance(population, str):
		population = [population, population]

	if not isinstance(distance, list):
		distance = [distance]
	if theta_dist is None:
		theta_dist = [0.9*d for d in distance]
	assert len(theta_dist)==len(distance),'Incompatible number of distances and number of edge thresholds.'

	if population[0]==population[1]:
		neighborhood_kwargs.update({'mode': 'self'})

	df_A, path_A = get_position_table(pos, population=population[0], return_path=True)
	df_B, path_B = get_position_table(pos, population=population[1], return_path=True)

	df_A, df_B = distance_cut_neighborhood(df_A,df_B,neigh_dist,**neighborhood_kwargs)

	for td,d in zip(theta_dist, distance):

		if neighborhood_kwargs['mode']=='two-pop':
			neigh_col = f'neighborhood_2_circle_{d}_px'
		elif neighborhood_kwargs['mode']=='self':
			neigh_col = f'neighborhood_self_circle_{d}_px'

		edge_filter_A = (df_A['POSITION_X'] > td)&(df_A['POSITION_Y'] > td)&(df_A['POSITION_Y'] < (img_shape[0] - td))&(df_A['POSITION_X'] < (img_shape[1] - td))
		edge_filter_B = (df_B['POSITION_X'] > td)&(df_B['POSITION_Y'] > td)&(df_B['POSITION_Y'] < (img_shape[0] - td))&(df_B['POSITION_X'] < (img_shape[1] - td))
		df_A.loc[edge_filter_A, neigh_col] = np.nan
		df_B.loc[edge_filter_B, neigh_col] = np.nan

	df_A.to_pickle(path_A.replace('.csv','.pkl'))
	df_B.to_pickle(path_B.replace('.csv','.pkl'))

	if return_tables:
		return df_A, df_B


# def mask_intersection_neighborhood(setA, labelsA, setB, labelsB, threshold_iou=0.5, viewpoint='B'):
# 	# do whatever to match objects in A and B
# 	return setA, setB

if __name__ == "__main__":
	print('None')
	#print(segment(None,'test'))
