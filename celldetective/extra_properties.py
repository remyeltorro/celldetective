"""
Add extra properties to directly to regionprops
Functions must take regionmask as first argument and optionally intensity_image as second argument
If intensity is in function name, it will be replaced by the name of the channel. These measurements are applied automatically to all channels

"""

import numpy as np


# Percentiles

def intensity_percentile_99(regionmask, intensity_image):
	return np.nanpercentile(intensity_image[regionmask],99)

def intensity_percentile_95(regionmask, intensity_image):
	return np.nanpercentile(intensity_image[regionmask],95)

def intensity_percentile_90(regionmask, intensity_image):
	return np.nanpercentile(intensity_image[regionmask],90)

def intensity_percentile_75(regionmask, intensity_image):
	return np.nanpercentile(intensity_image[regionmask],75)

def intensity_percentile_50(regionmask, intensity_image):
	return np.nanpercentile(intensity_image[regionmask],50)

def intensity_percentile_25(regionmask, intensity_image):
	return np.nanpercentile(intensity_image[regionmask],25)

# STD

def intensity_std(regionmask, intensity_image):
	return np.nanstd(intensity_image[regionmask])