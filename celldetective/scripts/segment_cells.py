"""
Copright © 2022 Laboratoire Adhesion et Inflammation, Authored by Remy Torro.
"""

import argparse
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
import gc
from art import tprint
from scipy.ndimage import zoom


tprint("Segment")

parser = argparse.ArgumentParser(description="Segment a movie in position with the selected model",
								formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p',"--position", required=True, help="Path to the position")
parser.add_argument('-m',"--model", required=True,help="Model name")
parser.add_argument("--mode", default="target", choices=["target","effector","targets","effectors"],help="Cell population of interest")
parser.add_argument("--use_gpu", default="True", choices=["True","False"],help="use GPU")
parser.add_argument("--threads", default="1",help="Number of parallel threads")

args = parser.parse_args()
process_arguments = vars(args)
pos = str(process_arguments['position'])
mode = str(process_arguments['mode'])
use_gpu = process_arguments['use_gpu']
n_threads = int(process_arguments['threads'])

if use_gpu=='True' or use_gpu=='true' or use_gpu=='1':
	use_gpu = True
	n_threads = 1  # avoid misbehavior on GPU with multithreading
else:
	use_gpu = False

if not use_gpu:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
n_threads = int(process_arguments['threads'])

modelname = str(process_arguments['model'])

if mode.lower()=="target" or mode.lower()=="targets":
	label_folder = "labels_targets"
elif mode.lower()=="effector" or mode.lower()=="effectors":
	label_folder = "labels_effectors"

# Locate experiment config
parent1 = Path(pos).parent
expfolder = parent1.parent
config = PurePath(expfolder,Path("config.ini"))
assert os.path.exists(config),'The configuration file for the experiment could not be located. Abort.'
print("Configuration file: ",config)

####################################
# Check model requirements #########
####################################

modelpath = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],"models"])
print(modelpath)
model_complete_path = locate_segmentation_model(modelname)
if model_complete_path is None:
	print('Model could not be found. Abort.')
	os.abort()
else:
	print(f'Model successfully located in {model_complete_path}')

# load config
assert os.path.exists(model_complete_path+"config_input.json"),'The configuration for the inputs to the model could not be located. Abort.'
with open(model_complete_path+"config_input.json") as config_file:
	input_config = json.load(config_file)

# Parse target channels
required_channels = input_config["channels"]

channel_indices = _extract_channel_indices_from_config(config, required_channels)
print(f'Required channels: {required_channels} located at channel indices {channel_indices}.')
required_spatial_calibration = input_config['spatial_calibration']
print(f'Expected spatial calibration is {required_spatial_calibration}.')

normalization_percentile = input_config['normalization_percentile']
normalization_clip = input_config['normalization_clip']
normalization_values = input_config['normalization_values']

model_type = input_config['model_type']

movie_prefix = ConfigSectionMap(config,"MovieSettings")["movie_prefix"]
spatial_calibration = float(ConfigSectionMap(config,"MovieSettings")["pxtoum"])
len_movie = float(ConfigSectionMap(config,"MovieSettings")["len_movie"])

# Try to find the file
try:
	file = glob(pos+f"movie/{movie_prefix}*.tif")[0]
except IndexError:
	print('Movie could not be found. Check the prefix.')
	os.abort()

len_movie_auto = auto_load_number_of_frames(file)
if len_movie_auto is not None:
	len_movie = len_movie_auto

if model_type=='cellpose':
	diameter = input_config['diameter']
	# if diameter!=30:
	# 	required_spatial_calibration = None 	# ignore spatial calibration and use diameter
	cellprob_threshold = input_config['cellprob_threshold']
	flow_threshold = input_config['flow_threshold']

scale = _estimate_scale_factor(spatial_calibration, required_spatial_calibration)
print(f"Scale = {scale}...")

nbr_channels = _extract_nbr_channels_from_config(config)
print(f'Number of channels in the input movie: {nbr_channels}')
img_num_channels = _get_img_num_per_channel(channel_indices, int(len_movie), nbr_channels)

# If everything OK, prepare output, load models
print('Erasing previous segmentation folder.')
if os.path.exists(pos+label_folder):
	rmtree(pos+label_folder)
os.mkdir(pos+label_folder)
print(f'Folder {pos+label_folder} successfully generated.')
log=f'segmentation model: {modelname}\n'
with open(pos+f'log_{mode}.json', 'a') as f:
	f.write(f'{datetime.datetime.now()} SEGMENT \n')
	f.write(log)


# Loop over all frames and segment
def segment_index(indices):
	global scale

	if model_type=='stardist':
		model = StarDist2D(None, name=modelname, basedir=Path(model_complete_path).parent)
		model.config.use_gpu = use_gpu
		model.use_gpu = use_gpu
		print(f"StarDist model {modelname} successfully loaded.")
		scale_model = scale

	elif model_type=='cellpose':

		import torch
		if not use_gpu:
			device = torch.device("cpu")
		else:
			device = torch.device("cuda")

		model = CellposeModel(gpu=use_gpu, device=device, pretrained_model=model_complete_path+modelname, model_type=None, nchan=len(required_channels)) #diam_mean=30.0,
		if scale is None:
			scale_model = model.diam_mean / model.diam_labels
		else:
			scale_model = scale * model.diam_mean / model.diam_labels
		print(f"Diam mean: {model.diam_mean}; Diam labels: {model.diam_labels}; Final rescaling: {scale_model}...")
		print(f'Cellpose model {modelname} successfully loaded.')

	for t in tqdm(indices,desc="frame"):
		
		# Load channels at time t
		values = []
		percentiles = []
		for k in range(len(normalization_percentile)):
			if normalization_percentile[k]:
				percentiles.append(normalization_values[k])
				values.append(None)
			else:
				percentiles.append(None)
				values.append(normalization_values[k])

		f = load_frames(img_num_channels[:,t], file, scale=scale_model, normalize_input=True, normalize_kwargs={"percentiles": percentiles, 'values': values, 'clip': normalization_clip})
		f = np.moveaxis([interpolate_nan(f[:,:,c].copy()) for c in range(f.shape[-1])],0,-1)

		if np.any(img_num_channels[:,t]==-1):
			f[:,:,np.where(img_num_channels[:,t]==-1)[0]] = 0.
		

		if model_type=="stardist":
			Y_pred, details = model.predict_instances(f, n_tiles=model._guess_n_tiles(f), show_tile_progress=False, verbose=False)
			Y_pred = Y_pred.astype(np.uint16)

		elif model_type=="cellpose":

			img = np.moveaxis(f, -1, 0)
			Y_pred, _, _ = model.eval(img, diameter = diameter, cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold, channels=None, normalize=False)
			Y_pred = Y_pred.astype(np.uint16)

		if scale is not None:
			Y_pred = zoom(Y_pred, [1./scale_model,1./scale_model],order=0)

		template = load_frames(0,file,scale=1,normalize_input=False)
		if Y_pred.shape != template.shape[:2]:
			Y_pred = resize(Y_pred, template.shape[:2], order=0)

		save_tiff_imagej_compatible(pos+os.sep.join([label_folder,f"{str(t).zfill(4)}.tif"]), Y_pred, axes='YX')

		del f;
		del template;
		del Y_pred;
		gc.collect()


import concurrent.futures

# Multithreading
indices = list(range(img_num_channels.shape[1]))
chunks = np.array_split(indices, n_threads)

with concurrent.futures.ThreadPoolExecutor() as executor:
	executor.map(segment_index, chunks)

# threads = []
# for i in range(n_threads):
# 	thread_i = threading.Thread(target=segment_index, args=[chunks[i]])
# 	threads.append(thread_i)
# for th in threads:
# 	th.start()
# for th in threads:
# 	th.join()

print('Done.')

try:
	del model
except:
	pass

gc.collect()





