import numpy as np

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