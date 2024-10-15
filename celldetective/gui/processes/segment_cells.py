from multiprocessing import Process
import time
import datetime
import os
import json
from stardist.models import StarDist2D
from cellpose.models import CellposeModel
from celldetective.io import locate_segmentation_model, auto_load_number_of_frames, load_frames
from celldetective.utils import interpolate_nan, _estimate_scale_factor, _extract_channel_indices_from_config, ConfigSectionMap, _extract_nbr_channels_from_config, _get_img_num_per_channel
from pathlib import Path, PurePath
from glob import glob
from shutil import rmtree
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
from csbdeep.io import save_tiff_imagej_compatible
from scipy.ndimage import zoom

class SegmentCellProcess(Process):
	
	def __init__(self, queue=None, parent_window=None):
		Process.__init__(self)
		self.__queue = queue
		self.parent_window = parent_window

		self.pos = self.parent_window.pos
		self.mode = self.parent_window.mode
		self.model_name = self.parent_window.model_name
		
		if self.mode.lower()=="target" or self.mode.lower()=="targets":
			self.label_folder = "labels_targets"
		elif self.mode.lower()=="effector" or self.mode.lower()=="effectors":
			self.label_folder = "labels_effectors"

		self.check_gpu()
		self.locate_experiment_config()
		self.locate_model_path()

		# Parse target channels
		self.required_channels = self.input_config["channels"]

		channel_indices = _extract_channel_indices_from_config(self.config, self.required_channels)
		print(f'Required channels: {self.required_channels} located at channel indices {channel_indices}.')
		required_spatial_calibration = self.input_config['spatial_calibration']
		print(f'Expected spatial calibration is {required_spatial_calibration}.')

		self.normalization_percentile = self.input_config['normalization_percentile']
		self.normalization_clip = self.input_config['normalization_clip']
		self.normalization_values = self.input_config['normalization_values']

		self.model_type = self.input_config['model_type']

		movie_prefix = ConfigSectionMap(self.config,"MovieSettings")["movie_prefix"]
		spatial_calibration = float(ConfigSectionMap(self.config,"MovieSettings")["pxtoum"])
		self.len_movie = float(ConfigSectionMap(self.config,"MovieSettings")["len_movie"])

		# Try to find the file
		try:
			self.file = glob(self.pos+f"movie/{movie_prefix}*.tif")[0]
		except IndexError:
			print('Movie could not be found. Check the prefix.')
			os.abort()

		len_movie_auto = auto_load_number_of_frames(self.file)
		if len_movie_auto is not None:
			self.len_movie = len_movie_auto

		if self.model_type=='cellpose':
			self.diameter = self.input_config['diameter']
			self.cellprob_threshold = self.input_config['cellprob_threshold']
			self.flow_threshold = self.input_config['flow_threshold']

		self.scale = _estimate_scale_factor(spatial_calibration, required_spatial_calibration)
		print(f"Scale = {self.scale}...")
		
		nbr_channels = _extract_nbr_channels_from_config(self.config)
		print(f'Number of channels in the input movie: {nbr_channels}')
		self.img_num_channels = _get_img_num_per_channel(channel_indices, int(self.len_movie), nbr_channels)

		# If everything OK, prepare output, load models
		print('Erasing previous segmentation folder.')
		if os.path.exists(self.pos+self.label_folder):
			rmtree(self.pos+self.label_folder)
		os.mkdir(self.pos+self.label_folder)
		print(f'Folder {self.pos+self.label_folder} successfully generated.')
		log=f'segmentation model: {self.model_name}\n'
		with open(self.pos+f'log_{self.mode}.json', 'a') as f:
			f.write(f'{datetime.datetime.now()} SEGMENT \n')
			f.write(log)

		self.sum_done = 0
		self.t0 = time.time()

	def locate_experiment_config(self):

		parent1 = Path(self.pos).parent
		expfolder = parent1.parent
		self.config = PurePath(expfolder,Path("config.ini"))

		assert os.path.exists(self.config),'The configuration file for the experiment could not be located. Abort.'
		print("Configuration file: ",self.config)

	def locate_model_path(self):

		self.model_complete_path = locate_segmentation_model(self.model_name)
		if self.model_complete_path is None:
			print('Model could not be found. Abort.')
			os.abort()
		else:
			print(f'Model successfully located in {self.model_complete_path}')
		assert os.path.exists(self.model_complete_path+"config_input.json"),'The configuration for the inputs to the model could not be located. Abort.'
		with open(self.model_complete_path+"config_input.json") as config_file:
			self.input_config = json.load(config_file)
					
	def check_gpu(self):

		self.use_gpu = self.parent_window.use_gpu
		if not self.use_gpu:
			os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

	def run(self):

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
			self.__queue.put([self.sum_done, pred_time])

		# Send end signal
		self.__queue.put("finished")
		self.__queue.close()

	def end_process(self):
		
		self.terminate()
		self.__queue.put("finished")
