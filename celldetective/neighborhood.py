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

def distance_cut_neighborhood(setA, setB, distance):
	# compute cdist matrix
	# run matrix along both axes and write #
	return setA, setB

def mask_intersection_neighborhood(setA, labelsA, setB, labelsB, threshold_iou=0.5, viewpoint='B'):
	# do whatever to match objects in A and B
	return setA, setB