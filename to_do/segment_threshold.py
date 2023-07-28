#!/usr/bin/python

import sys
import os
import sys
import gc

import time
import shutil

import numpy as np
import pandas as pd

from glob import glob
from natsort import natsorted
from tqdm import tqdm
from tifffile import imread, TiffFile
from pathlib import Path, PurePath
from adccfactory.core.parse import ConfigSectionMap

from adccfactory.core.utils import nbr_channels_from_config
from adccfactory.core.parse import load_frame_i
from skimage.filters import difference_of_gaussians
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter, gaussian_laplace, laplace, minimum_filter, maximum_filter, percentile_filter
from scipy.ndimage import generate_binary_structure, white_tophat

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops_table
from csbdeep.io import save_tiff_imagej_compatible

from scipy.ndimage import zoom
import json
from skimage import io
import matplotlib.pyplot as plt
import argparse
from skimage.exposure import match_histograms

def minmax(img,amplification=1000.):

	img = np.copy(img).astype(float)
	min_value = float(np.amin(img[img!=0.]))
	max_value = float(np.amax(img[img!=0.]))
	img = amplification * (img-min_value) / (max_value - min_value)
	return img


parser = argparse.ArgumentParser(description="Segment a movie from a threshold configuration file",
								formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p',"--position", required=True, help="Location of the position")
parser.add_argument('-c',"--config", required=True,help="Path to the configuration")

args = parser.parse_args()
config_subprocess = vars(args)

pos = config_subprocess["position"] #str(sys.argv[1])
print(f"Position folder: {pos}")

parent1 = Path(pos).parent
expfolder = parent1.parent
config = PurePath(expfolder,Path("config.ini"))
print("Configuration file: ",config)
label_folder = "labels/"

####################################
# Check model requirements #########
####################################

threshold_instructions = config_subprocess["config"] #str(sys.argv[2])
# load config
with open(threshold_instructions) as config_file:
	config_threshold = json.load(config_file)

print(config_threshold)

req_channels = [config_threshold["target_channel"]]
print(f"Required channels: {req_channels}")

#####################################
## Check that data meets requirements
#####################################

channels = []
for c in req_channels:
	try:
		c1 = int(ConfigSectionMap(config,"MovieSettings")[c])
		channels.append(c1)
	except Exception as e:
		print(f"Error {e}. The channel required by the model is not available in your data... Check the configuration file.")
		os.abort()

print(f"Channels: {channels} successfully meet the requirements {req_channels}")

movie_prefix = ConfigSectionMap(config,"MovieSettings")["movie_prefix"]

# Try to find the file
try:
	file = glob(pos+f"movie/{movie_prefix}*.tif")[0]
except IndexError:
	os.abort()

# Detect what must be loaded of the file
# need to estimate nbr_channels

# Try to estimate automatically # frames
with TiffFile(file) as tif:
	try:
		tif_tags = {}
		for tag in tif.pages[0].tags.values():
			name, value = tag.name, tag.value
			tif_tags[name] = value
		img_desc = tif_tags["ImageDescription"]
		attr = img_desc.split("\n")
	except:
		pass
	try:
		# Try nslices
		nslices = int(attr[np.argmax([s.startswith("slices") for s in attr])].split("=")[-1])
		len_movie = nslices
		print(f"Auto-detected movie length movie: {len_movie}")
	except:
		try:
			# try nframes
			frames = int(attr[np.argmax([s.startswith("frames") for s in attr])].split("=")[-1])
			len_movie = frames
			print(f"Auto-detected movie length movie: {len_movie}")
		except:
			pass
try:
	del tif;
	del tif_tags;
	del img_desc;
except:
	pass
gc.collect()

nbr_channels = nbr_channels_from_config(config)
indices_all_channels = []
for c in channels:
	indices = np.arange(len_movie*nbr_channels)[c::nbr_channels]
	indices_all_channels.append(indices)
indices_all_channels = np.array(indices_all_channels, dtype=int)
nbr_frame_per_channel = np.array([len(i) for i in indices_all_channels])
assert np.all(nbr_frame_per_channel == nbr_frame_per_channel[0]),"There is a different number of frames per channel.. Please check your image."

if os.path.exists(pos+label_folder):
	shutil.rmtree(pos+label_folder)
os.mkdir(pos+label_folder)

def threshold_segmentation(img, min_threshold, max_threshold, target_channel=0, filters=[], gaussian_blur_sigma=1.6,
	median_filter_size=4, maximum_filter_size=4, minimum_filter_size=4, percentile_filter_size=4, percentile_filter_percentile=5,
	variance_filter_size=4, std_filter_size=4, tophat_filter_connectivity=4, tophat_filter_size=100 ,dog_sigma_low=1, dog_sigma_high=1.6, log_sigma=2, marker_footprint=20,
	marker_min_distance=20, delete_pass_1=None, delete_pass_2=None, filter1=None, filter2=None,
	cell_properties=None, equalize_histogram=False, reference_frame=None):

	img_target = img[:,:,target_channel].astype(float)

	#img_target[img_target0==0] = np.median(img_target)
	# plt.imshow(img_target)
	# plt.show()

	# cell_properties_options = list(np.copy(cell_properties))
	# cell_properties_options.remove("centroid")
	# for k in range(nbr_channels):
	# 	cell_properties_options.append(f'intensity_mean-{k}')
	# cell_properties_options.remove('label')
	# cell_properties_options.remove('intensity_mean')

	#img_target_pre = np.copy(img_target)

	if equalize_histogram*(reference_frame is not None):
		print('equalizing!')
		img_target = match_histograms(img_target, reference_frame, channel_axis=-1)
		#img_target[img_target_pre==0] = 0.0
	

	# fig,ax = plt.subplots(1,2,figsize=(15,5),sharex=True,sharey=True)
	# ax[0].imshow(reference_frame)
	# ax[1].imshow(img_target)
	# plt.show()


	# plt.imshow(img_target)
	# plt.show()

	# Preprocessing

	if len(filters)==0:
		pass 
	else:
		for filt in filters:

			if filt=="gaussian blur":
				img_target = gaussian_filter(img_target, gaussian_blur_sigma)

			if filt=="DoG filter":
				img_target = difference_of_gaussians(img_target, dog_sigma_low, high_sigma=dog_sigma_high)

			if filt=="LoG filter":
				img_target = gaussian_laplace(img_target, log_sigma)

			if filt=="median filter":
				img_target = median_filter(img_target, size=median_filter_size)

			if filt=="maximum filter":
				img_target = maximum_filter(img_target, size=maximum_filter_size)

			if filt=="minimum filter":
				img_target = minimum_filter(img_target, size=minimum_filter_size)

			if filt=="percentile filter":
				img_target = percentile_filter(img_target, percentile_filter_percentile,size=percentile_filter_size)

			if filt=="variance filter":
				win_mean = uniform_filter(img_target, (variance_filter_size, variance_filter_size))
				win_sqr_mean = uniform_filter(img_target**2, (variance_filter_size, variance_filter_size))
				img_target = win_sqr_mean - win_mean**2

			if filt=="standard deviation filter":
				win_mean = uniform_filter(img_target, (std_filter_size, std_filter_size))
				win_sqr_mean = uniform_filter(img_target**2, (std_filter_size, std_filter_size))
				img_target = win_sqr_mean - win_mean**2
				img_target = np.sqrt(img_target)

			if filt=="laplace filter":
				img_target = laplace(img_target)

			if filt=="tophat filter":

				struct = generate_binary_structure(rank=2,connectivity=tophat_filter_connectivity)
				img_target = white_tophat(img_target, structure=struct, size=tophat_filter_size)


	binary = (img_target>=min_threshold)*(img_target<=max_threshold)*255.
	# plt.imshow(binary)
	# plt.show()

	# Marker identification
	binary = binary_fill_holes(binary)
	distance = ndi.distance_transform_edt(binary.astype(float))
	coords = peak_local_max(distance, footprint=np.ones((marker_footprint, marker_footprint)),
		labels=binary.astype(int), min_distance=marker_min_distance)

	# Watershed
	mask = np.zeros(distance.shape, dtype=bool)
	mask[tuple(coords.T)] = True
	markers, _ = ndi.label(mask)
	labels = watershed(-distance, markers, mask=binary)

	if cell_properties is not None:

		# Delete pass
		props = pd.DataFrame(regionprops_table(labels, intensity_image=img, properties=cell_properties))
		props["class"] = 1

		if (delete_pass_1 is not None)*(delete_pass_2 is not None):
			col_x = delete_pass_1[0]
			col_y = delete_pass_2[0]
			min_x = delete_pass_1[1]; max_x = delete_pass_1[2];
			min_y = delete_pass_2[1]; max_y = delete_pass_2[2];
			props.loc[(props[col_x]>min_x)
							&(props[col_x]<max_x)
							&(props[col_y]>min_y)
							&(props[col_y]<max_y),"class"] = 0	
			props = props.loc[props["class"]==1]

		if (filter1 is not None)*(filter2 is not None):
			col_x = filter1[0]
			col_y = filter2[0]
			min_x = filter1[1]; max_x = filter1[2];
			min_y = filter2[1]; max_y = filter2[2];
			props['class'] = 0
			props.loc[(props[col_x]>min_x)
					&(props[col_x]<max_x)
					&(props[col_y]>min_y)
					&(props[col_y]<max_y),"class"] = 1

		to_keep = props.loc[props["class"]==1,"label"].to_numpy()
		
		# All cells
		ids = np.unique(labels)
		for c in ids:
			if c not in to_keep:
				labels[np.where(labels==c)] = 0.

	return labels

if config_threshold["equalize_histogram"]:
	reference_frame = config_threshold["reference_frame"]
	index = nbr_channels*reference_frame + channels[0]
	print(f"Reference frame index: {index}")
	histogram_reference = load_frame_i(index,file,normalize_input=False)
else:
	histogram_reference = None

# Loop over all frames and segment
for t in tqdm(range(len_movie),desc="frame"):
	
	# sys.stderr.write(f"Frame: {int(float(t+1)/float(indices_all_channels.shape[1])*100)}")
	# sys.stderr.flush()
	
	# Run regionprops to have properties for filtering
	indices = [nbr_channels*t]
	for i in range(nbr_channels-1):
		indices += [indices[-1]+1]

	# Load channels at time t
	multichannel = []
	for i in indices:
		f = load_frame_i(i, file,normalize_input=False)
		multichannel.append(f)

	multichannel = np.array(multichannel)
	if (len(multichannel.shape)==3)*(multichannel.shape[0]<=12):
		multichannel = np.moveaxis(multichannel,0,-1)
	print(multichannel.shape)	
	
	Y_pred = threshold_segmentation(multichannel, config_threshold["min_threshold"], config_threshold["max_threshold"], target_channel=channels[0], filters=config_threshold["filters"], gaussian_blur_sigma=config_threshold["gaussian_blur_sigma"],
	median_filter_size=config_threshold["median_filter_size"], maximum_filter_size=config_threshold["maximum_filter_size"], minimum_filter_size=config_threshold["minimum_filter_size"], percentile_filter_size=config_threshold["percentile_filter_size"],
	percentile_filter_percentile=config_threshold["percentile_filter_percentile"], variance_filter_size=config_threshold["variance_filter_size"], std_filter_size=config_threshold["std_filter_size"],
	dog_sigma_low=config_threshold["dog_sigma_low"], dog_sigma_high=config_threshold["dog_sigma_high"], tophat_filter_connectivity = config_threshold["tophat_filter_connectivity"], tophat_filter_size = config_threshold["tophat_filter_size"],
	log_sigma = config_threshold["log_sigma"], marker_footprint=config_threshold["marker_footprint"], marker_min_distance=config_threshold["marker_min_distance"], delete_pass_1=config_threshold["delete_pass_1"], delete_pass_2=config_threshold["delete_pass_2"], filter1=config_threshold["filter1"], filter2=config_threshold["filter2"],
	cell_properties=config_threshold["cell_properties"], equalize_histogram=config_threshold["equalize_histogram"], reference_frame=histogram_reference)

	Y_pred = Y_pred.astype(np.uint16)

	save_tiff_imagej_compatible(pos+label_folder+f"{str(t).zfill(4)}.tif", Y_pred, axes='YX')

	del f;
	del multichannel;
	del Y_pred;
	gc.collect()

#sys.stderr.flush()