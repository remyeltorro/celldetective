"""
Copright © 2022 Laboratoire Adhesion et Inflammation, Authored by Remy Torro.
"""

import argparse
import os
import json
from stardist.models import StarDist2D
from cellpose.models import CellposeModel
from celldetective.io import locate_segmentation_model, auto_load_number_of_frames, load_frames
from celldetective.utils import _estimate_scale_factor, _extract_channel_indices_from_config, _extract_channel_indices, ConfigSectionMap, _extract_nbr_channels_from_config, _get_img_num_per_channel, normalize_per_channel
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
import threading

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
if use_gpu=='True' or use_gpu=='true' or use_gpu=='1':
	use_gpu = True
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

modelpath = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]+"/models"
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

nbr_channels = _extract_nbr_channels_from_config(config)
print(f'Number of channels in the input movie: {nbr_channels}')
img_num_channels = _get_img_num_per_channel(channel_indices, int(len_movie), nbr_channels)

# If everything OK, prepare output, load models
print('Erasing previous segmentation folder.')
if os.path.exists(os.sep.join([pos,label_folder])):
	rmtree(os.sep.join([pos,label_folder]))
os.mkdir(os.sep.join([pos,label_folder]))
print(f'Folder {os.sep.join([pos,label_folder])} successfully generated.')

if model_type=='stardist':
	model = StarDist2D(None, name=modelname, basedir=Path(model_complete_path).parent)
	model.config.use_gpu = use_gpu
	model.use_gpu = use_gpu
	#model._axes_tile_overlap("YXC")
	#fov = np.array(model._axes_tile_overlap('YX'))
	#print('Tile overlap: ', model._tile_overlap)
	print(f"StarDist model {modelname} successfully loaded.")

elif model_type=='cellpose':
	print(model_complete_path+modelname, 'nchan',len(required_channels), required_channels)
	model = CellposeModel(gpu=use_gpu, pretrained_model=model_complete_path+modelname, model_type=None, nchan=len(required_channels)) #diam_mean=30.0, 
	print(f'Cellpose model {modelname} successfully loaded.')

# Loop over all frames and segment
def segment_index(indices):

	for t in tqdm(indices,desc="frame"):
		
		# Load channels at time t
		f = load_frames(img_num_channels[:,t], file, scale=scale, normalize_input=False)
		f = normalize_per_channel([f], normalization_percentile_mode=normalization_percentile, normalization_values=normalization_values,
									normalization_clipping=normalization_clip)
		f = f[0]
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
			Y_pred = zoom(Y_pred, [1./scale,1./scale],order=0)

		template = load_frames(0,file,scale=1,normalize_input=False)
		if Y_pred.shape != template.shape[:2]:
			Y_pred = resize(Y_pred, template.shape[:2], order=0)

		save_tiff_imagej_compatible(os.sep.join([pos,label_folder,f"{str(t).zfill(4)}.tif"]), Y_pred, axes='YX')

		del f;
		del template;
		del Y_pred;
		gc.collect()

# Multithreading
indices = list(range(img_num_channels.shape[1]))
chunks = np.array_split(indices, n_threads)
threads = []
for i in range(n_threads):
	thread_i = threading.Thread(target=segment_index, args=[chunks[i]])
	threads.append(thread_i)
for th in threads:
	th.start()
for th in threads:
	th.join()

print('Done.')
del model
gc.collect()





