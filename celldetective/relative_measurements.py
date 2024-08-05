import pandas as pd
import numpy as np
from celldetective.utils import derivative, extract_identity_col
import os
import subprocess
from math import ceil
abs_path = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], 'celldetective'])
import random
from tqdm import tqdm

def measure_pairs(pos, neighborhood_protocol):
	
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

	ref_id_col = extract_identity_col(df_reference)
	ref_tracked = False
	if ref_id_col is None:
		return None
	elif ref_id_col=='TRACK_ID':
		ref_tracked = True
	neigh_id_col = extract_identity_col(df_neighbor)
	neigh_tracked = False
	if neigh_id_col is None:
		return None
	elif neigh_id_col=='TRACK_ID':
		neigh_tracked = True
	
	centre_of_mass_columns = [(c,c.replace('POSITION_X','POSITION_Y')) for c in list(df_neighbor.columns) if c.endswith('centre_of_mass_POSITION_X')]
	centre_of_mass_labels = [c.replace('_centre_of_mass_POSITION_X','') for c in list(df_neighbor.columns) if c.endswith('centre_of_mass_POSITION_X')]

	for t in np.unique(list(df_reference['FRAME'].unique())+list(df_neighbor['FRAME'])):
		
		group_reference = df_reference.loc[df_reference['FRAME']==t,:]
		group_neighbors = df_neighbor.loc[df_neighbor['FRAME']==t, :]
		
		for tid, group in group_reference.groupby(ref_id_col):

			neighborhood = group.loc[: , f'{neighborhood_description}'].to_numpy()[0]
			coords_reference = group[['POSITION_X', 'POSITION_Y']].to_numpy()[0]

			neighbors = []
			if isinstance(neighborhood, float) or neighborhood!=neighborhood:
				pass
			else:
				for neigh in neighborhood:
					neighbors.append(neigh['id'])

			unique_neigh = list(np.unique(neighbors))
			print(f'{unique_neigh=}')

			neighbor_properties = group_neighbors.loc[group_neighbors[neigh_id_col].isin(unique_neigh)]
			
			for nc, group_neigh in neighbor_properties.groupby(neigh_id_col):
				
				neighbor_vector = np.zeros((2))
				neighbor_vector[:] = np.nan
				mass_displacement_vector = np.zeros((len(centre_of_mass_columns), 2))

				coords_centre_of_mass = []
				for col in centre_of_mass_columns:
					coords_centre_of_mass.append(group_neigh[[col[0],col[1]]].to_numpy()[0])

				dot_product_vector = np.zeros((len(centre_of_mass_columns)))
				dot_product_vector[:] = np.nan

				cosine_dot_vector = np.zeros((len(centre_of_mass_columns)))
				cosine_dot_vector[:] = np.nan

				coords_neighbor = group_neigh[['POSITION_X', 'POSITION_Y']].to_numpy()[0]
				neighbor_vector[0] = coords_neighbor[0] - coords_reference[0]
				neighbor_vector[1] = coords_neighbor[1] - coords_reference[1]

				if neighbor_vector[0]==neighbor_vector[0] and neighbor_vector[1]==neighbor_vector[1]:
					angle = np.arctan2(neighbor_vector[1], neighbor_vector[0])
					relative_distance = np.sqrt(neighbor_vector[0]**2 + neighbor_vector[1]**2)
					
					for z,cols in enumerate(centre_of_mass_columns):

							mass_displacement_vector[z,0] = coords_centre_of_mass[z][0] - coords_neighbor[0]
							mass_displacement_vector[z,1] = coords_centre_of_mass[z][1] - coords_neighbor[1]

							dot_product_vector[z] = np.dot(mass_displacement_vector[z], -neighbor_vector)
							cosine_dot_vector[z] = np.dot(mass_displacement_vector[z], -neighbor_vector) / (np.linalg.norm(mass_displacement_vector[z])*np.linalg.norm(-neighbor_vector))

					relative_measurements.append(
							{'REFERENCE_ID': tid, 'NEIGHBOR_ID': nc,
							'reference_population': reference_population,
							'neighbor_population': neighbor_population,
							'FRAME': t, 'distance': relative_distance,
							'angle': angle * 180 / np.pi,
							f'status_{neighborhood_description}': 1,
							f'class_{neighborhood_description}': 0,
							'reference_tracked': ref_tracked, 'neighbors_tracked': neigh_tracked,
							 })
					for z,lbl in enumerate(centre_of_mass_labels):
						relative_measurements[-1].update({lbl+'_centre_of_mass_dot_product': dot_product_vector[z], lbl+'_centre_of_mass_dot_cosine': cosine_dot_vector[z]})
	
	df_pairs = pd.DataFrame(relative_measurements)
	
	return df_pairs







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

	ref_id_col = extract_identity_col(df_reference)
	if ref_id_col is not None:
		df_reference = df_reference.sort_values(by=[ref_id_col, 'FRAME'])

	ref_tracked = False
	if ref_id_col=='TRACK_ID':
		compute_velocity = True
		ref_tracked = True
	elif ref_id_col=='ID':
		df_pairs = measure_pairs(pos, neighborhood_protocol)
		return df_pairs
	else:
		print('ID or TRACK ID column could not be found in neighbor table. Abort.')
		return None

	print(f'Measuring pair signals...')

	neigh_id_col = extract_identity_col(df_neighbor)
	neigh_tracked = False
	if neigh_id_col=='TRACK_ID':
		compute_velocity = True
		neigh_tracked = True
	elif neigh_id_col=='ID':
		df_pairs = measure_pairs(pos, neighborhood_protocol)
		return df_pairs
	else:
		print('ID or TRACK ID column could not be found in neighbor table. Abort.')
		return None

	try:
		for tid, group in df_reference.groupby(ref_id_col):

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
			print(f'Reference cell {tid}: found {len(unique_neigh)} neighbour cells: {unique_neigh}...')

			neighbor_properties = df_neighbor.loc[df_neighbor[neigh_id_col].isin(unique_neigh)]

			for nc, group_neigh in neighbor_properties.groupby(neigh_id_col):
				
				coords_neighbor = group_neigh[['POSITION_X', 'POSITION_Y']].to_numpy()
				timeline_neighbor = group_neigh['FRAME'].to_numpy()

				# # Perform timeline matching to have same start-end points and no gaps
				full_timeline, _, _ = timeline_matching(timeline_reference, timeline_neighbor)

				neighbor_vector = np.zeros((len(full_timeline), 2))
				neighbor_vector[:,:] = np.nan

				centre_of_mass_columns = [(c,c.replace('POSITION_X','POSITION_Y')) for c in list(neighbor_properties.columns) if c.endswith('centre_of_mass_POSITION_X')]
				centre_of_mass_labels = [c.replace('_centre_of_mass_POSITION_X','') for c in list(neighbor_properties.columns) if c.endswith('centre_of_mass_POSITION_X')]

				mass_displacement_vector = np.zeros((len(centre_of_mass_columns), len(full_timeline), 2))
				mass_displacement_vector[:,:,:] = np.nan

				dot_product_vector = np.zeros((len(centre_of_mass_columns), len(full_timeline)))
				dot_product_vector[:,:] = np.nan

				cosine_dot_vector = np.zeros((len(centre_of_mass_columns), len(full_timeline)))
				cosine_dot_vector[:,:] = np.nan

				coords_centre_of_mass = []
				for col in centre_of_mass_columns:
					coords_centre_of_mass.append(group_neigh[[col[0],col[1]]].to_numpy())

				# Relative distance
				for t in range(len(full_timeline)):

					if t in timeline_reference and t in timeline_neighbor: # meaning position exists on both sides

						idx_reference = list(timeline_reference).index(t) #index_reference[list(full_timeline).index(t)]
						idx_neighbor = list(timeline_neighbor).index(t) #index_neighbor[list(full_timeline).index(t)]

						neighbor_vector[t, 0] = coords_neighbor[idx_neighbor, 0] - coords_reference[idx_reference, 0]
						neighbor_vector[t, 1] = coords_neighbor[idx_neighbor, 1] - coords_reference[idx_reference, 1]

						for z,cols in enumerate(centre_of_mass_columns):

							mass_displacement_vector[z,t,0] = coords_centre_of_mass[z][idx_neighbor, 0] - coords_neighbor[idx_neighbor, 0]
							mass_displacement_vector[z,t,1] = coords_centre_of_mass[z][idx_neighbor, 1] - coords_neighbor[idx_neighbor, 1]

							dot_product_vector[z,t] = np.dot(mass_displacement_vector[z,t], -neighbor_vector[t])
							cosine_dot_vector[z,t] = np.dot(mass_displacement_vector[z,t], -neighbor_vector[t]) / (np.linalg.norm(mass_displacement_vector[z,t])*np.linalg.norm(-neighbor_vector[t]))
							if tid==44.0 and nc==173.0:
								print(f'{centre_of_mass_columns[z]=} {mass_displacement_vector[z,t]=} {-neighbor_vector[t]=} {dot_product_vector[z,t]=} {cosine_dot_vector[z,t]=}')
							

				angle = np.zeros(len(full_timeline))
				angle[:] = np.nan

				exclude = neighbor_vector[:,1]!=neighbor_vector[:,1]
				angle[~exclude] = np.arctan2(neighbor_vector[:, 1][~exclude], neighbor_vector[:, 0][~exclude])
				#print(f'Angle before unwrap: {angle}')
				angle[~exclude] = np.unwrap(angle[~exclude])
				#print(f'Angle after unwrap: {angle}')
				relative_distance = np.sqrt(neighbor_vector[:,0]**2 + neighbor_vector[:, 1]**2)
				#print(f'Timeline: {full_timeline}; Distance: {relative_distance}')

				if compute_velocity:
					rel_velocity = derivative(relative_distance, full_timeline, **velocity_kwargs)
					rel_velocity_long_timescale = derivative(relative_distance, full_timeline, window = 7, mode='bi')
					#rel_velocity = np.insert(rel_velocity, 0, np.nan)[:-1]
					
					angular_velocity = np.zeros(len(full_timeline))
					angular_velocity[:] = np.nan
					angular_velocity_long_timescale = np.zeros(len(full_timeline))
					angular_velocity_long_timescale[:] = np.nan

					angular_velocity[~exclude] = derivative(angle[~exclude], full_timeline[~exclude], **velocity_kwargs)
					angular_velocity_long_timescale[~exclude] = derivative(angle[~exclude], full_timeline[~exclude], window = 7, mode='bi')

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
									'velocity_smooth': rel_velocity_long_timescale[t], 
									'angle': angle[t] * 180 / np.pi,
									#'angle-neigh-ref': angle[t] * 180 / np.pi, 
									'angular_velocity': angular_velocity[t],
									'angular_velocity_smooth': angular_velocity_long_timescale[t],
									f'status_{neighborhood_description}': 1,
									f'residence_time_in_{neighborhood_description}': cum_sum,
									f'class_{neighborhood_description}': 0,
									f't0_{neighborhood_description}': time_of_first_entrance_in_neighborhood[nc],
									'reference_tracked': ref_tracked, 'neighbors_tracked': neigh_tracked,
									 })
							for z,lbl in enumerate(centre_of_mass_labels):
								relative_measurements[-1].update({lbl+'_centre_of_mass_dot_product': dot_product_vector[z,t], lbl+'_centre_of_mass_dot_cosine': cosine_dot_vector[z,t]})

						else:
							relative_measurements.append(
									{'REFERENCE_ID': tid, 'NEIGHBOR_ID': nc,
									'reference_population': reference_population,
									'neighbor_population': neighbor_population,
									'FRAME': t, 'distance': relative_distance[t],
									'velocity': rel_velocity[t], 
									'velocity_smooth': rel_velocity_long_timescale[t],
									'angle': angle[t] * 180 / np.pi,
									#'angle-neigh-ref': angle[t] * 180 / np.pi, 
									'angular_velocity': angular_velocity[t],
									'angular_velocity_smooth': angular_velocity_long_timescale[t],
									f'status_{neighborhood_description}': 0,
									f'residence_time_in_{neighborhood_description}': cum_sum,
									f'class_{neighborhood_description}': 0,
									f't0_{neighborhood_description}': time_of_first_entrance_in_neighborhood[nc],
									'reference_tracked': ref_tracked, 'neighbors_tracked': neigh_tracked,
									 })
							for z,lbl in enumerate(centre_of_mass_labels):
								relative_measurements[-1].update({lbl+'_centre_of_mass_dot_product': dot_product_vector[z,t], lbl+'_centre_of_mass_dot_cosine': cosine_dot_vector[z,t]})

		df_pairs = pd.DataFrame(relative_measurements)

		return df_pairs

	except KeyError:
		print(f"Neighborhood {description} not found in data frame. Measurements for this neighborhood will not be calculated")


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


# def mcf7_size_model(x,x0,x2):
# 	return np.piecewise(x, [x<= x0, (x > x0)*(x<=x2), x > x2], [lambda x: 1, lambda x: -1/(x2-x0)*x + (1+x0/(x2-x0)), 0])
# def sigmoid(x,x0,k):
# 	return 1/(1 + np.exp(-(x-x0)/k))
# def velocity_law(x):
# 	return np.piecewise(x, [x<=-10, x > -10],[lambda x: 0., lambda x: (1*x+10)*(1-sigmoid(x, 1,1))/10])


# def probabilities(pairs,radius_critical=80,radius_max=150):
# 	scores = []
# 	pair_dico=[]
# 	print(f'Found {len(pairs)} TC-NK pairs...')
# 	if len(pairs) > 0:
# 		unique_tcs = np.unique(pairs['tc'].to_numpy())
# 		unique_nks = np.unique(pairs['nk'].to_numpy())
# 		matrix = np.zeros((len(unique_tcs), len(unique_nks)))
# 		for index, row in pairs.iterrows():

# 			i = np.where(unique_tcs == row['tc'])[0]
# 			j = np.where(unique_nks == row['nk'])[0]

# 			d_prob = mcf7_size_model(row['drel'], radius_critical, radius_max)
# 			lamp_prob = sigmoid(row['lamp1'], 1.05, 0.01)
# 			synapse_prob = row['syn_class']
# 			velocity_prob = velocity_law(row['vrel'])  # 1-sigmoid(row['vrel'], 1,1)
# 			time_prob = row['t_residence_rel']

# 			hypotheses = [d_prob, velocity_prob, lamp_prob, synapse_prob,
# 						  time_prob]  # lamp_prob d_prob, synapse_prob, velocity_prob, lamp_prob
# 			s = np.sum(hypotheses) / len(hypotheses)

# 			matrix[i, j] = s  # synapse_prob': synapse_prob,
# 			pair_dico.append(
# 				{ 'tc': row['tc'], 'nk': row['nk'], 'synapse_prob': synapse_prob,
# 				 'd_prob': d_prob, 'lamp_prob': lamp_prob, 'velocity_prob': velocity_prob, 'time_prob': time_prob})
# 		pair_dico = pd.DataFrame(pair_dico)

# 		hypotheses = ['velocity_prob', 'd_prob', 'time_prob', 'lamp_prob', 'synapse_prob']

# 		for i in tqdm(range(2000)):
# 			sample = np.array(random.choices(np.linspace(0, 1, 100), k=len(hypotheses)))
# 			weights = sample / np.sum(sample)

# 			score_i = {}
# 			for k, hyp in enumerate(hypotheses):
# 				score_i.update({'w_' + hyp: weights[k]})
# 			probs=[]
# 			for cells, group in pair_dico.groupby(['tc']):

# 				group['total_prob'] = 0
# 				for hyp in hypotheses:
# 					group['total_prob'] += group[hyp] * score_i['w_' + hyp]
# 					probs.append(group)
# 	return probs

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



