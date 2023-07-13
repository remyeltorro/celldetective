from skimage.filters import difference_of_gaussians
import scipy.ndimage as snd

def gauss_filter(img, sigma, *kwargs):
	return snd.gaussian_filter(img, sigma, *kwargs)

def median_filter(img, size, *kwargs):
	return snd.median_filter(img, size, *kwargs)

def maximum_filter(img, size, *kwargs):
	return snd.maximum_filter(img, size, *kwargs)

def minimum_filter(img, size, *kwargs):
	return snd.minimum_filter(img, size, *kwargs)

def percentile_filter(img, percentile, size, *kwargs):
	return snd.percentile_filter(img, percentile, size, *kwargs)

def variance_filter(img, size):
	win_mean = snd.uniform_filter(img, (size,size))
	win_sqr_mean = snd.uniform_filter(img**2, (size,size))
	img = win_sqr_mean - win_mean**2
	return img

def std_filter(img, size):
	win_mean = snd.uniform_filter(img, (size,size))
	win_sqr_mean = snd.uniform_filter(img**2, (size, size))
	img = np.sqrt(win_sqr_mean - win_mean**2)
	return img

def laplace_filter(img, *kwargs):
	return snd.laplace(img, *kwargs)

def dog_filter(img, sigma_low, sigma_high, *kwargs):
	return difference_of_gaussians(img, sigma_low, sigma_high, *kwargs)

def log_filter(img, sigma, *kwargs):
	return snd.gaussian_laplace(img, sigma, *kwargs)

def tophat_filter(img, size, connectivity=4, *kwargs):
	structure = snd.generate_binary_structure(rank=2, connectivity=connectivity)
	img = snd.white_tophat(img, structure=structure, size=size, *kwargs)
	return img

