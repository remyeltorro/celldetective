from multiprocessing import Process
import time
import datetime
import os
import json
from pathlib import Path, PurePath

from celldetective.io import auto_load_number_of_frames, load_frames
from celldetective.utils import extract_experiment_channels, ConfigSectionMap, _get_img_num_per_channel
from celldetective.utils import remove_trajectory_measurements
from celldetective.measure import drop_tonal_features, measure_features, measure_isotropic_intensity

from glob import glob
from tqdm import tqdm
import numpy as np
import concurrent.futures
import pandas as pd
from natsort import natsorted
from tifffile import imread
from art import tprint

class MeasurementProcess(Process):

	def __init__(self, queue=None, process_args=None):
		
		super().__init__()
		
		self.queue = queue

		if process_args is not None:
			for key, value in process_args.items():
				setattr(self, key, value)

		self.column_labels = {'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}

		tprint("Measure")
		self.timestep_dataframes = []

		# Experiment		
		self.prepare_folders()

		self.locate_experiment_config()
		self.extract_experiment_parameters()
		self.read_measurement_instructions()
		self.detect_movie_and_labels()
		self.detect_tracks()
		self.detect_channels()

		self.check_possible_measurements()

		self.write_log()

		self.sum_done = 0
		self.t0 = time.time()

	def check_possible_measurements(self):

		if (self.file is None) or (self.intensity_measurement_radii is None):
			self.do_iso_intensities = False
			print('Either no image, no positions or no radii were provided... Isotropic intensities will not be computed...')
		else:
			self.do_iso_intensities = True

		if self.label_path is None:
			self.do_features = False
			print('No labels were provided... Features will not be computed...')
		else:
			self.do_features = True
		
		if self.trajectories is None:
			print('Use features as a substitute for the trajectory table.')
			if 'label' not in self.features:
				self.features.append('label')


	def read_measurement_instructions(self):
		
		instr_path = PurePath(self.expfolder,Path(f"{self.instruction_file}"))
		if os.path.exists(instr_path):
			print(f"Tracking instructions for the {self.mode} population has been successfully located.")
			with open(instr_path, 'r') as f:
				self.instructions = json.load(f)
				print("Reading the following instructions: ", self.instructions)
			if 'background_correction' in self.instructions:
				self.background_correction = self.instructions['background_correction']
			else:
				self.background_correction = None

			if 'features' in self.instructions:
				self.features = self.instructions['features']
			else:
				self.features = None

			if 'border_distances' in self.instructions:
				self.border_distances = self.instructions['border_distances']
			else:
				self.border_distances = None

			if 'spot_detection' in self.instructions:
				self.spot_detection = self.instructions['spot_detection']
			else:
				self.spot_detection = None

			if 'haralick_options' in self.instructions:
				self.haralick_options = self.instructions['haralick_options']
			else:
				self.haralick_options = None

			if 'intensity_measurement_radii' in self.instructions:
				self.intensity_measurement_radii = self.instructions['intensity_measurement_radii']
			else:
				self.intensity_measurement_radii = None

			if 'isotropic_operations' in self.instructions:
				self.isotropic_operations = self.instructions['isotropic_operations']
			else:
				self.isotropic_operations = None

			if 'clear_previous' in self.instructions:
				self.clear_previous = self.instructions['clear_previous']
			else:
				self.clear_previous = True

		else:
			print('No measurement instructions found. Use default measurements.')
			self.features = ['area', 'intensity_mean']
			self.border_distances = None
			self.haralick_options = None
			self.clear_previous = False
			self.background_correction = None
			self.spot_detection = None
			self.intensity_measurement_radii = 10
			self.isotropic_operations = ['mean']

		if self.features is None:
			self.features = []


	def detect_channels(self):
		self.img_num_channels = _get_img_num_per_channel(self.channel_indices, self.len_movie, self.nbr_channels)

	def write_log(self):

		features_log=f'features: {self.features}'
		border_distances_log=f'border_distances: {self.border_distances}'
		haralick_options_log=f'haralick_options: {self.haralick_options}'
		background_correction_log=f'background_correction: {self.background_correction}'
		spot_detection_log=f'spot_detection: {self.spot_detection}'
		intensity_measurement_radii_log=f'intensity_measurement_radii: {self.intensity_measurement_radii}'
		isotropic_options_log=f'isotropic_operations: {self.isotropic_operations} \n'
		log='\n'.join([features_log,border_distances_log,haralick_options_log,background_correction_log,spot_detection_log,intensity_measurement_radii_log,isotropic_options_log])
		with open(self.pos + f'log_{self.mode}.json', 'a') as f:
			f.write(f'{datetime.datetime.now()} MEASURE \n')
			f.write(log+'\n')

	def prepare_folders(self):

		if self.mode.lower()=="target" or self.mode.lower()=="targets":
			self.label_folder = "labels_targets"
			self.table_name = "trajectories_targets.csv"
			self.instruction_file = os.sep.join(["configs","measurement_instructions_targets.json"])

		elif self.mode.lower()=="effector" or self.mode.lower()=="effectors":
			self.label_folder = "labels_effectors"
			self.table_name = "trajectories_effectors.csv"
			self.instruction_file = os.sep.join(["configs","measurement_instructions_effectors.json"])

	def extract_experiment_parameters(self):

		self.movie_prefix = ConfigSectionMap(self.config,"MovieSettings")["movie_prefix"]
		self.spatial_calibration = float(ConfigSectionMap(self.config,"MovieSettings")["pxtoum"])
		self.time_calibration = float(ConfigSectionMap(self.config,"MovieSettings")["frametomin"])
		self.len_movie = float(ConfigSectionMap(self.config,"MovieSettings")["len_movie"])
		self.shape_x = int(ConfigSectionMap(self.config,"MovieSettings")["shape_x"])
		self.shape_y = int(ConfigSectionMap(self.config,"MovieSettings")["shape_y"])

		self.channel_names, self.channel_indices = extract_experiment_channels(self.config)
		self.nbr_channels = len(self.channel_names)

	def locate_experiment_config(self):

		parent1 = Path(self.pos).parent
		self.expfolder = parent1.parent
		self.config = PurePath(self.expfolder,Path("config.ini"))

		if not os.path.exists(self.config):
			print('The configuration file for the experiment was not found...')
			self.abort_process()

	def detect_tracks(self):

		# Load trajectories, add centroid if not in trajectory
		self.trajectories = self.pos+os.sep.join(['output','tables', self.table_name])
		if os.path.exists(self.trajectories):
			print('trajectory exists...')
			self.trajectories = pd.read_csv(self.trajectories)
			if 'TRACK_ID' not in list(self.trajectories.columns):
				self.do_iso_intensities = False
				self.intensity_measurement_radii = None
				if self.clear_previous:
					print('No TRACK_ID... Clear previous measurements...')
					self.trajectories = None #remove_trajectory_measurements(trajectories, column_labels)
					self.do_features = True
					self.features += ['centroid']
			else:
				if self.clear_previous:
					print('TRACK_ID found... Clear previous measurements...')
					self.trajectories = remove_trajectory_measurements(self.trajectories, self.column_labels)
		else:
			self.trajectories = None
			self.do_features = True
			self.features += ['centroid']
			self.do_iso_intensities = False

	def detect_movie_and_labels(self):

		self.label_path = natsorted(glob(self.pos+f"{self.label_folder}"+os.sep+"*.tif"))
		if len(self.label_path)>0:
			print(f"Found {len(self.label_path)} segmented frames...")
		else:
			self.features = None
			self.haralick_options = None
			self.border_distances = None
			self.label_path = None

		try:
			self.file = glob(self.pos+os.sep.join(["movie", f"{self.movie_prefix}*.tif"]))[0]
		except IndexError:
			self.file = None
			self.haralick_option = None
			self.features = drop_tonal_features(self.features)

		len_movie_auto = auto_load_number_of_frames(self.file)
		if len_movie_auto is not None:
			self.len_movie = len_movie_auto

	def parallel_job(self, indices):

		try:

			for t in tqdm(indices,desc="frame"):

				if self.file is not None:
					img = load_frames(self.img_num_channels[:,t], self.file, scale=None, normalize_input=False)

				if self.label_path is not None:
					lbl = imread(self.label_path[t])

				if self.trajectories is not None:

					positions_at_t = self.trajectories.loc[self.trajectories[self.column_labels['time']]==t].copy()

				if self.do_features:
					feature_table = measure_features(img, lbl, features=self.features, border_dist=self.border_distances,
													 channels=self.channel_names, haralick_options=self.haralick_options, verbose=False,
													 normalisation_list=self.background_correction, spot_detection=self.spot_detection)
					if self.trajectories is None:
						positions_at_t = feature_table[['centroid-1', 'centroid-0', 'class_id']].copy()
						positions_at_t['ID'] = np.arange(len(positions_at_t))  # temporary ID for the cells, that will be reset at the end since they are not tracked
						positions_at_t.rename(columns={'centroid-1': 'POSITION_X', 'centroid-0': 'POSITION_Y'}, inplace=True)
						positions_at_t['FRAME'] = int(t)
						column_labels = {'track': "ID", 'time': self.column_labels['time'], 'x': self.column_labels['x'],
										 'y': self.column_labels['y']}
					feature_table.rename(columns={'centroid-1': 'POSITION_X', 'centroid-0': 'POSITION_Y'}, inplace=True)
				
				if self.do_iso_intensities:
					iso_table = measure_isotropic_intensity(positions_at_t, img, channels=self.channel_names, intensity_measurement_radii=self.intensity_measurement_radii, column_labels=self.column_labels, operations=self.isotropic_operations, verbose=False)

				if self.do_iso_intensities and self.do_features:
					measurements_at_t = iso_table.merge(feature_table, how='outer', on='class_id',suffixes=('', '_delme'))
					measurements_at_t = measurements_at_t[[c for c in measurements_at_t.columns if not c.endswith('_delme')]]
				elif self.do_iso_intensities * (not self.do_features):
					measurements_at_t = iso_table
				elif self.do_features:
					measurements_at_t = positions_at_t.merge(feature_table, how='outer', on='class_id',suffixes=('', '_delme'))
					measurements_at_t = measurements_at_t[[c for c in measurements_at_t.columns if not c.endswith('_delme')]]
			
				center_of_mass_x_cols = [c for c in list(measurements_at_t.columns) if c.endswith('centre_of_mass_x')]
				center_of_mass_y_cols = [c for c in list(measurements_at_t.columns) if c.endswith('centre_of_mass_y')]
				for c in center_of_mass_x_cols:
					measurements_at_t.loc[:,c.replace('_x','_POSITION_X')] = measurements_at_t[c] + measurements_at_t['POSITION_X']
				for c in center_of_mass_y_cols:
					measurements_at_t.loc[:,c.replace('_y','_POSITION_Y')] = measurements_at_t[c] + measurements_at_t['POSITION_Y']
				measurements_at_t = measurements_at_t.drop(columns = center_of_mass_x_cols+center_of_mass_y_cols)
				
				if measurements_at_t is not None:
					measurements_at_t[self.column_labels['time']] = t
					self.timestep_dataframes.append(measurements_at_t)
				
				self.sum_done+=1/self.len_movie*100
				mean_exec_per_step = (time.time() - self.t0) / (self.sum_done*self.len_movie / 100 + 1)
				pred_time = (self.len_movie - (self.sum_done*self.len_movie / 100 + 1)) * mean_exec_per_step
				self.queue.put([self.sum_done, pred_time])
		
		except Exception as e:
			print(e)

	def run(self):

		self.indices = list(range(self.img_num_channels.shape[1]))
		chunks = np.array_split(self.indices, self.n_threads)

		with concurrent.futures.ThreadPoolExecutor() as executor:
			executor.map(self.parallel_job, chunks)

		if len(self.timestep_dataframes)>0:

			df = pd.concat(self.timestep_dataframes)	
			df.reset_index(inplace=True, drop=True)

			if self.trajectories is None:
				df['ID'] = np.arange(len(df))

			if self.column_labels['track'] in df.columns:
				df = df.sort_values(by=[self.column_labels['track'], self.column_labels['time']])
			else:
				df = df.sort_values(by=self.column_labels['time'])

			df.to_csv(self.pos+os.sep.join(["output", "tables", self.table_name]), index=False)
			print(f'Measurements successfully written in table {self.pos+os.sep.join(["output", "tables", self.table_name])}')
			print('Done.')
		else:
			print('No measurement could be performed. Check your inputs.')
			print('Done.')	
		
		# Send end signal
		self.queue.put("finished")
		self.queue.close()

	def end_process(self):

		self.terminate()
		self.queue.put("finished")

	def abort_process(self):
		
		self.terminate()
		self.queue.put("error")