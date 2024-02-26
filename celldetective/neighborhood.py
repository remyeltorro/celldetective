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
	Updates the live status for cells in two datasets based on specified status columns and options.

	This function assigns a live status to cells in two datasets (setA and setB) based on the provided
	status columns and options. If no status column is provided, all cells are marked as live. Otherwise,
	the function updates the datasets based on the status criteria, potentially inverting the status
	based on the `not_status_option`.

	Parameters
	----------
	setA : pandas.DataFrame
		The first dataset containing trajectory or position information for cells.
	setB : pandas.DataFrame
		The second dataset containing trajectory or position information for cells.
	status : list or None
		A list containing the names of the columns in setA and setB that classify cells as alive (1) or dead (0).
		If None, all cells are considered alive. The list should contain exactly two elements.
	not_status_option : list
		A list containing boolean values indicating whether to invert the status for setA and setB, respectively.
		True means the status should be inverted; False means it should not.

	Returns
	-------
	tuple
		A tuple containing the updated setA and setB DataFrames, along with the final status column names
		used to classify cells in each set.

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
	
	"""
	Computes the attention weight for each cell based on its proximity to cells of an opposite type within a specified distance.

	This function calculates the attention weight for cells by considering the distance to the cells of an opposite type
	within a given cutoff distance. It optionally considers only the 'live' opposite cells based on their status. The function 
	returns two arrays: one containing the attention weights and another containing the IDs of the closest opposite cells.

	Parameters
	----------
	dist_matrix : ndarray
		A 2D array representing the distance matrix between cells of two types.
	cut_distance : float
		The cutoff distance within which opposite cells will influence the attention weight.
	opposite_cell_status : ndarray
		An array indicating the status (e.g., live or dead) of each opposite cell. Only used when `include_dead_weight` is False.
	opposite_cell_ids : ndarray
		An array containing the IDs of the opposite cells.
	axis : int, optional
		The axis along which to compute the weights (default is 1). Axis 0 corresponds to rows, and axis 1 corresponds to columns.
	include_dead_weight : bool, optional
		If True, includes all opposite cells within the cutoff distance in the weight calculation, regardless of their status.
		If False, only considers opposite cells that are 'live' (default is True).

	Returns
	-------
	tuple of ndarrays
		A tuple containing two arrays: `weights` and `closest_opposite`. `weights` is an array of attention weights for each cell,
		and `closest_opposite` is an array of the IDs of the closest opposite cells within the cutoff distance.

	"""

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
	
	"""
	Computes neighborhood metrics for specified cell populations within a given position, based on distance criteria and additional parameters.

	This function assesses the neighborhood interactions between two specified cell populations (or within a single population) at a given position.
	It computes various neighborhood metrics based on specified distances, considering the entire image or excluding edge regions.
	The results are optionally cleared of previous neighborhood calculations and can be returned as updated tables.

	Parameters
	----------
	pos : str
		The path to the position directory where the analysis is to be performed.
	distance : float or list of float
		The distance(s) in pixels to define neighborhoods.
	population : list of str, optional
		Names of the cell populations to analyze. If a single population is provided, it is used for both populations in the analysis (default is ['targets', 'effectors']).
	theta_dist : float or list of float, optional
		Edge threshold(s) in pixels to exclude cells close to the image boundaries from the analysis. If not provided, defaults to 90% of each specified distance.
	img_shape : tuple of int, optional
		The dimensions (height, width) of the images in pixels (default is (2048, 2048)).
	return_tables : bool, optional
		If True, returns the updated data tables for both populations (default is False).
	clear_neigh : bool, optional
		If True, clears existing neighborhood columns from the data tables before computing new metrics (default is False).
	event_time_col : str, optional
		The column name indicating the event time for each cell, required if mean neighborhood metrics are to be computed before events.
	neighborhood_kwargs : dict, optional
		Additional keyword arguments for neighborhood computation, including mode, status options, and metrics (default includes mode 'two-pop', and symmetrization).

	Returns
	-------
	pandas.DataFrame or (pandas.DataFrame, pandas.DataFrame)
		If `return_tables` is True, returns the updated data tables for the specified populations. If only one population is analyzed, both returned data frames will be identical.

	Raises
	------
	AssertionError
		If the specified position path does not exist or if the number of distances and edge thresholds do not match.
	
	"""

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

	"""
	Computes and appends neighborhood metrics to a dataframe based on specified neighborhood characteristics.

	This function iterates through a dataframe grouped by either 'TRACK_ID' or ['position', 'TRACK_ID'] (if 'position' column exists)
	and computes various neighborhood metrics (inclusive, exclusive, intermediate counts) for each cell. It can also decompose these
	metrics by cell status (e.g., live or dead) if specified.

	Parameters
	----------
	neigh_table : pandas.DataFrame
		A dataframe containing neighborhood information for each cell, including position, track ID, frame, and a specified neighborhood column.
	neigh_col : str
		The column name in `neigh_table` that contains neighborhood information (e.g., a list of neighbors with their attributes).
	metrics : list of str, optional
		The metrics to be computed from the neighborhood information. Possible values include 'inclusive', 'exclusive', and 'intermediate'.
		Default is ['inclusive', 'exclusive', 'intermediate'].
	decompose_by_status : bool, optional
		If True, the metrics are computed separately for different statuses (e.g., live or dead) of the neighboring cells. Default is False.

	Returns
	-------
	pandas.DataFrame
		The input dataframe with additional columns for each of the specified metrics, and, if `decompose_by_status` is True, separate
		metrics for each status.

	Notes
	-----
	- 'inclusive' count refers to the total number of neighbors.
	- 'exclusive' count refers to the number of neighbors that are closest.
	- 'intermediate' count refers to the sum of weights attributed to neighbors, representing a weighted count.
	- If `decompose_by_status` is True, metrics are appended with '_s0' or '_s1' to indicate the status they correspond to.

	Examples
	--------
	>>> neigh_table = pd.DataFrame({
	...     'TRACK_ID': [1, 1, 2, 2],
	...     'FRAME': [1, 2, 1, 2],
	...     'neighborhood_info': [{'weight': 1, 'status': 1, 'closest': 1}, ...]  # example neighborhood info
	... })
	>>> neigh_col = 'neighborhood_info'
	>>> updated_neigh_table = compute_neighborhood_metrics(neigh_table, neigh_col, metrics=['inclusive'], decompose_by_status=True)
	# Computes the inclusive count of neighbors for each cell, decomposed by cell status.
	
	"""

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
	
	"""
	Computes the mean neighborhood metrics for each cell track before a specified event time.

	This function calculates the mean values of specified neighborhood metrics (inclusive, exclusive, intermediate)
	for each cell track up to and including the frame of an event. The function requires the neighborhood metrics to
	have been previously computed and appended to the input dataframe. It operates on grouped data based on position
	and track ID, handling cases with or without position information.

	Parameters
	----------
	neigh_table : pandas.DataFrame
		A dataframe containing cell track data with precomputed neighborhood metrics and event time information.
	neigh_col : str
		The base name of the neighborhood metric columns in `neigh_table`.
	event_time_col : str or None
		The column name indicating the event time for each cell track. If None, the maximum frame number in the
		dataframe is used as the event time for all tracks.

	Returns
	-------
	pandas.DataFrame
		The input dataframe with added columns for the mean neighborhood metrics before the event for each cell track.
		The new columns are named as 'mean_count_{metric}_{neigh_col}_before_event', where {metric} is one of
		'inclusive', 'exclusive', 'intermediate'.

	"""


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
