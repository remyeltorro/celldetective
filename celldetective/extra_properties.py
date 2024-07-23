"""
Add extra properties to directly to regionprops
Functions must take regionmask as first argument and optionally intensity_image as second argument
If intensity is in function name, it will be replaced by the name of the channel. These measurements are applied automatically to all channels

"""
import warnings

import numpy as np
from scipy.ndimage import distance_transform_edt, center_of_mass
from scipy.spatial.distance import euclidean
from celldetective.utils import interpolate_nan

# Percentiles

def intensity_percentile_ninety_nine(regionmask, intensity_image):
	return np.nanpercentile(intensity_image[regionmask],99)

def intensity_percentile_ninety_five(regionmask, intensity_image):
	return np.nanpercentile(intensity_image[regionmask],95)

def intensity_percentile_ninety(regionmask, intensity_image):
	return np.nanpercentile(intensity_image[regionmask],90)

def intensity_percentile_seventy_five(regionmask, intensity_image):
	return np.nanpercentile(intensity_image[regionmask],75)

def intensity_percentile_fifty(regionmask, intensity_image):
	return np.nanpercentile(intensity_image[regionmask],50)

def intensity_percentile_twenty_five(regionmask, intensity_image):
	return np.nanpercentile(intensity_image[regionmask],25)

# STD

def intensity_std(regionmask, intensity_image):
	return np.nanstd(intensity_image[regionmask])


def intensity_median(regionmask, intensity_image):
	return np.nanmedian(intensity_image[regionmask])

def intensity_nanmean(regionmask, intensity_image):
	return np.nanmean(intensity_image[regionmask])

def intensity_centre_of_mass_displacement(regionmask, intensity_image):

	intensity_image = interpolate_nan(intensity_image.copy())

	y, x = np.mgrid[:regionmask.shape[0], :regionmask.shape[1]]
	xtemp = x.copy()
	ytemp = y.copy()
	intensity_weighted_center = center_of_mass(intensity_image, regionmask)
	centroid_x = intensity_weighted_center[1]
	centroid_y = intensity_weighted_center[0]

	#centroid_x = np.sum(xtemp * intensity_image) / np.sum(intensity_image)
	geometric_centroid_x = np.sum(xtemp * regionmask) / np.sum(regionmask)
	geometric_centroid_y = np.sum(ytemp * regionmask) / np.sum(regionmask)
	try:
		distance = euclidean(np.array((geometric_centroid_y, geometric_centroid_x)), np.array((centroid_y, centroid_x)))
	except:
		distance = np.nan

	delta_x = geometric_centroid_x - centroid_x
	delta_y = geometric_centroid_y - centroid_y
	direction_arctan = np.arctan2(delta_y, delta_x) * 180 / np.pi
	if direction_arctan < 0:
		direction_arctan += 360

	return distance, direction_arctan, centroid_x - geometric_centroid_x, centroid_y - geometric_centroid_y

def intensity_radial_gradient(regionmask, intensity_image):

	try:
		warnings.filterwarnings('ignore', message="Polyfit may be poorly conditioned")
		cell_mask = regionmask.copy()
		intensity = intensity_image.copy()
		y = intensity[cell_mask].flatten()
		x = distance_transform_edt(cell_mask)
		x = x[cell_mask].flatten()
		params = np.polyfit(x, y, 1)
		line = np.poly1d(params)

		return line.coefficients[0], line.coefficients[1]
	except Exception as e:
		print(e)
		return np.nan, np.nan


def intensity_centre_of_mass_displacement_edge(regionmask, intensity_image):
	
	intensity_image = interpolate_nan(intensity_image.copy())

	edt = distance_transform_edt(regionmask)
	min_distance = 0
	max_distance = np.amax([3,0.1*edt.max()]) # minimum 3 px edge
	thresholded = (edt <= max_distance) * (edt > min_distance)
	edge_mask = np.copy(regionmask)
	edge_mask[np.where(thresholded == 0)] = 0

	if np.sum(edge_mask)>0:
		
		y, x = np.mgrid[:edge_mask.shape[0], :edge_mask.shape[1]]
		xtemp = x.copy()
		ytemp = y.copy()
		intensity_weighted_center = center_of_mass(intensity_image, edge_mask)
		centroid_x = intensity_weighted_center[1]
		centroid_y = intensity_weighted_center[0]

		#centroid_x = np.sum(xtemp * intensity_image) / np.sum(intensity_image)
		geometric_centroid_x = np.sum(xtemp * regionmask) / np.sum(regionmask)
		geometric_centroid_y = np.sum(ytemp * regionmask) / np.sum(regionmask)
		
		try:
			distance = euclidean(np.array((geometric_centroid_y, geometric_centroid_x)), np.array((centroid_y, centroid_x)))
		except:
			distance = np.nan

		delta_x = geometric_centroid_x - centroid_x
		delta_y = geometric_centroid_y - centroid_y
		direction_arctan = np.arctan2(delta_y, delta_x) * 180 / np.pi
		if direction_arctan < 0:
			direction_arctan += 360

		return distance, direction_arctan, centroid_x - geometric_centroid_x, centroid_y - geometric_centroid_y
	else:
		return np.nan, np.nan, np.nan, np.nan