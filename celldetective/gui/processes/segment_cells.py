from multiprocessing import Process
import time
import datetime
import os
import json
from stardist.models import StarDist2D
from cellpose.models import CellposeModel
from celldetective.io import locate_segmentation_model, auto_load_number_of_frames, load_frames
from celldetective.utils import extract_experiment_channels, interpolate_nan, _estimate_scale_factor, _extract_channel_indices_from_config, ConfigSectionMap, _extract_nbr_channels_from_config, _get_img_num_per_channel
from pathlib import Path, PurePath
from glob import glob
from shutil import rmtree
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
from csbdeep.io import save_tiff_imagej_compatible
from scipy.ndimage import zoom
from celldetective.segmentation import segment_frame_from_thresholds
import gc
from art import tprint

import concurrent.futures

class BaseSegmentProcess(Process):

	def __init__(self, queue=None, process_args=None, *args, **kwargs):
		
		super().__init__(*args, **kwargs)
		
		self.queue = queue

		if process_args is not None:
			for key, value in process_args.items():
				setattr(self, key, value)

		tprint("Segment")

		# Experiment
		self.locate_experiment_config()
		self.extract_experiment_parameters()
		self.detect_movie_length()
		self.write_folders()

	def write_folders(self):

		if self.mode.lower()=="target" or self.mode.lower()=="targets":
			self.label_folder = "labels_targets"
		elif self.mode.lower()=="effector" or self.mode.lower()=="effectors":
			self.label_folder = "labels_effectors"

		print('Erasing previous segmentation folder.')
		if os.path.exists(self.pos+self.label_folder):
			rmtree(self.pos+self.label_folder)
		os.mkdir(self.pos+self.label_folder)
		print(f'Folder {self.pos+self.label_folder} successfully generated.')


	def extract_experiment_parameters(self):

		self.spatial_calibration = float(ConfigSectionMap(self.config,"MovieSettings")["pxtoum"])
		self.len_movie = float(ConfigSectionMap(self.config,"MovieSettings")["len_movie"])
		self.movie_prefix = ConfigSectionMap(self.config,"MovieSettings")["movie_prefix"]
		self.nbr_channels = _extract_nbr_channels_from_config(self.config)
		self.channel_names, self.channel_indices = extract_experiment_channels(self.config)

	def locate_experiment_config(self):

		parent1 = Path(self.pos).parent
		expfolder = parent1.parent
		self.config = PurePath(expfolder,Path("config.ini"))

		if not os.path.exists(self.config):
			print('The configuration file for the experiment was not found...')
			self.abort_process()

	def detect_movie_length(self):

		try:
			self.file = glob(self.pos+f"movie/{self.movie_prefix}*.tif")[0]
		except IndexError:
			print('Movie could not be found. Check the prefix.')
			self.abort_process()

		len_movie_auto = auto_load_number_of_frames(self.file)
		if len_movie_auto is not None:
			self.len_movie = len_movie_auto

	def end_process(self):

		self.terminate()
		self.queue.put("finished")

	def abort_process(self):
		
		self.terminate()
		self.queue.put("error")


class SegmentCellDLProcess(BaseSegmentProcess):
	
	def __init__(self, *args, **kwargs):
		
		super().__init__(*args, **kwargs)

		self.check_gpu()

		# Model
		self.locate_model_path()
		self.extract_model_input_parameters()
		self.detect_channels()
		self.detect_rescaling()

		self.write_log()

		self.sum_done = 0
		self.t0 = time.time()

	def extract_model_input_parameters(self):

		self.required_channels = self.input_config["channels"]
		self.normalization_percentile = self.input_config['normalization_percentile']
		self.normalization_clip = self.input_config['normalization_clip']
		self.normalization_values = self.input_config['normalization_values']
		self.model_type = self.input_config['model_type']
		self.required_spatial_calibration = self.input_config['spatial_calibration']
		
		if self.model_type=='cellpose':
			self.diameter = self.input_config['diameter']
			self.cellprob_threshold = self.input_config['cellprob_threshold']
			self.flow_threshold = self.input_config['flow_threshold']

	def write_log(self):

		log=f'segmentation model: {self.model_name}\n'
		with open(self.pos+f'log_{self.mode}.txt', 'a') as f:
			f.write(f'{datetime.datetime.now()} SEGMENT \n')
			f.write(log)

	def detect_channels(self):
		
		self.channel_indices = _extract_channel_indices_from_config(self.config, self.required_channels)
		print(f'Required channels: {self.required_channels} located at channel indices {self.channel_indices}...')
		self.img_num_channels = _get_img_num_per_channel(self.channel_indices, int(self.len_movie), self.nbr_channels)

	def detect_rescaling(self):

		self.scale = _estimate_scale_factor(self.spatial_calibration, self.required_spatial_calibration)
		print(f"Scale = {self.scale}...")

	def locate_model_path(self):

		self.model_complete_path = locate_segmentation_model(self.model_name)
		if self.model_complete_path is None:
			print('Model could not be found. Abort.')
			self.abort_process()
		else:
			print(f'Model successfully located in {self.model_complete_path}')
		
		if not os.path.exists(self.model_complete_path+"config_input.json"):
			print('The configuration for the inputs to the model could not be located. Abort.')
			self.abort_process()

		with open(self.model_complete_path+"config_input.json") as config_file:
			self.input_config = json.load(config_file)
					
	def check_gpu(self):

		if not self.use_gpu:
			os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

	def run(self):

		try:

			if self.model_type=='stardist':
				model = StarDist2D(None, name=self.model_name, basedir=Path(self.model_complete_path).parent)
				model.config.use_gpu = self.use_gpu
				model.use_gpu = self.use_gpu
				print(f"StarDist model {self.model_name} successfully loaded.")
				scale_model = self.scale

			elif self.model_type=='cellpose':


				import torch
				if not self.use_gpu:
					device = torch.device("cpu")
				else:
					device = torch.device("cuda")

				model = CellposeModel(gpu=self.use_gpu, device=device, pretrained_model=self.model_complete_path+self.model_name, model_type=None, nchan=len(self.required_channels)) #diam_mean=30.0,
				if self.scale is None:
					scale_model = model.diam_mean / model.diam_labels
				else:
					scale_model = self.scale * model.diam_mean / model.diam_labels
				print(f"Diam mean: {model.diam_mean}; Diam labels: {model.diam_labels}; Final rescaling: {scale_model}...")
				print(f'Cellpose model {self.model_name} successfully loaded.')

			for t in tqdm(range(self.len_movie),desc="frame"):
				
				# Load channels at time t
				values = []
				percentiles = []
				for k in range(len(self.normalization_percentile)):
					if self.normalization_percentile[k]:
						percentiles.append(self.normalization_values[k])
						values.append(None)
					else:
						percentiles.append(None)
						values.append(self.normalization_values[k])

				f = load_frames(self.img_num_channels[:,t], self.file, scale=scale_model, normalize_input=True, normalize_kwargs={"percentiles": percentiles, 'values': values, 'clip': self.normalization_clip})
				f = np.moveaxis([interpolate_nan(f[:,:,c].copy()) for c in range(f.shape[-1])],0,-1)

				if np.any(self.img_num_channels[:,t]==-1):
					f[:,:,np.where(self.img_num_channels[:,t]==-1)[0]] = 0.
				

				if self.model_type=="stardist":
					Y_pred, details = model.predict_instances(f, n_tiles=model._guess_n_tiles(f), show_tile_progress=False, verbose=False)
					Y_pred = Y_pred.astype(np.uint16)

				elif self.model_type=="cellpose":

					img = np.moveaxis(f, -1, 0)
					Y_pred, _, _ = model.eval(img, diameter = self.diameter, cellprob_threshold=self.cellprob_threshold, flow_threshold=self.flow_threshold, channels=None, normalize=False)
					Y_pred = Y_pred.astype(np.uint16)

				if self.scale is not None:
					Y_pred = zoom(Y_pred, [1./scale_model,1./scale_model],order=0)

				template = load_frames(0,self.file,scale=1,normalize_input=False)
				if Y_pred.shape != template.shape[:2]:
					Y_pred = resize(Y_pred, template.shape[:2], order=0)

				save_tiff_imagej_compatible(self.pos+os.sep.join([self.label_folder,f"{str(t).zfill(4)}.tif"]), Y_pred, axes='YX')
				
				# Send signal for progress bar
				self.sum_done+=1/self.len_movie*100
				mean_exec_per_step = (time.time() - self.t0) / (t+1)
				pred_time = (self.len_movie - (t+1)) * mean_exec_per_step
				self.queue.put([self.sum_done, pred_time])

		except Exception as e:
			print(e)

		# Send end signal
		self.queue.put("finished")
		self.queue.close()


class SegmentCellThresholdProcess(BaseSegmentProcess):
	
	def __init__(self, *args, **kwargs):
		
		super().__init__(*args, **kwargs)

		self.equalize = False

		# Model
		self.load_threshold_config()
		self.extract_threshold_parameters()
		self.detect_channels()
		self.prepare_equalize()

		self.write_log()

		self.sum_done = 0
		self.t0 = time.time()

	def prepare_equalize(self):

		if self.equalize:
			f_reference = load_frames(self.img_num_channels[:,self.equalize_time], self.file, scale=None, normalize_input=False)
			f_reference = f_reference[:,:,self.threshold_instructions['target_channel']]
		else:
			f_reference = None

		self.threshold_instructions.update({'equalize_reference': f_reference})

	def load_threshold_config(self):
		
		if os.path.exists(self.threshold_instructions):
			with open(self.threshold_instructions, 'r') as f:
				self.threshold_instructions = json.load(f)
		else:
			print('The configuration path is not valid. Abort.')
			self.abort_process()

	def extract_threshold_parameters(self):
		
		self.required_channels = [self.threshold_instructions['target_channel']]
		if 'equalize_reference' in self.threshold_instructions:
			self.equalize, self.equalize_time = self.threshold_instructions['equalize_reference']
			

	def write_log(self):

		log=f'Threshold segmentation: {self.threshold_instructions}\n'
		with open(self.pos+f'log_{self.mode}.txt', 'a') as f:
			f.write(f'{datetime.datetime.now()} SEGMENT \n')
			f.write(log)

	def detect_channels(self):

		self.channel_indices = _extract_channel_indices_from_config(self.config, self.required_channels)
		print(f'Required channels: {self.required_channels} located at channel indices {self.channel_indices}.')
		
		self.img_num_channels = _get_img_num_per_channel(np.arange(self.nbr_channels), self.len_movie, self.nbr_channels)
		self.threshold_instructions.update({'target_channel': self.channel_indices[0]})
		self.threshold_instructions.update({'channel_names': self.channel_names})

	def parallel_job(self, indices):

		try:

			for t in tqdm(indices,desc="frame"): #for t in tqdm(range(self.len_movie),desc="frame"):
				
				# Load channels at time t
				f = load_frames(self.img_num_channels[:,t], self.file, scale=None, normalize_input=False)
				mask = segment_frame_from_thresholds(f, **self.threshold_instructions)
				save_tiff_imagej_compatible(os.sep.join([self.pos, self.label_folder, f"{str(t).zfill(4)}.tif"]), mask.astype(np.uint16), axes='YX')

				del f;
				del mask;
				gc.collect()

				# Send signal for progress bar
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
		
		# Send end signal
		self.queue.put("finished")
		self.queue.close()