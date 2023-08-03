from skimage.filters import difference_of_gaussians
import scipy.ndimage as snd
import numpy as np

def gauss_filter(img, sigma, *kwargs):
	return snd.gaussian_filter(img.astype(float), sigma, *kwargs)

def median_filter(img, size, *kwargs):
	size = int(size)
	return snd.median_filter(img, size, *kwargs)

def maximum_filter(img, size, *kwargs):
	return snd.maximum_filter(img.astype(float), size, *kwargs)

def minimum_filter(img, size, *kwargs):
	return snd.minimum_filter(img.astype(float), size, *kwargs)

def percentile_filter(img, percentile, size, *kwargs):
	return snd.percentile_filter(img.astype(float), percentile, size, *kwargs)

def variance_filter(img, size):

	size = int(size)
	img = img.astype(float)
	win_mean = snd.uniform_filter(img, (size,size))
	win_sqr_mean = snd.uniform_filter(img**2, (size,size))
	img = win_sqr_mean - win_mean**2

	return img

def std_filter(img, size):

	size = int(size)
	img = img.astype(float)
	win_mean = snd.uniform_filter(img, (size,size))
	win_sqr_mean = snd.uniform_filter(img**2, (size, size))
	win_sqr_mean[win_sqr_mean<=0.] = 0. # add this to prevent sqrt from breaking
	img = np.sqrt(win_sqr_mean - win_mean**2)

	return img

def laplace_filter(img, output=float, *kwargs):
	return snd.laplace(img.astype(float), *kwargs)

def dog_filter(img, sigma_low, sigma_high, *kwargs):
	return difference_of_gaussians(img.astype(float), sigma_low, sigma_high, *kwargs)

def log_filter(img, sigma, *kwargs):
	return snd.gaussian_laplace(img.astype(float), sigma, *kwargs)

def tophat_filter(img, size, connectivity=4, *kwargs):
	structure = snd.generate_binary_structure(rank=2, connectivity=connectivity)
	img = snd.white_tophat(img.astype(float), structure=structure, size=size, *kwargs)
	return img

