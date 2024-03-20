"""
Add extra properties to directly to regionprops
Functions must take regionmask as first argument and optionally intensity_image as second argument
If intensity is in function name, it will be replaced by the name of the channel. These measurements are applied automatically to all channels

"""
import warnings

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import euclidean


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


def intensity_median(regionmask, intensity_image):
    return np.nanmedian(intensity_image[regionmask])


def intensity_centroid_distance(regionmask, intensity_image):
    y, x = np.mgrid[:regionmask.shape[0], :regionmask.shape[1]]
    xtemp = x.copy()
    ytemp = y.copy()

    centroid_x = np.sum(xtemp * intensity_image) / np.sum(intensity_image)
    centroid_y = np.sum(ytemp * intensity_image) / np.sum(intensity_image)
    geometric_centroid_x = np.sum(xtemp * regionmask) / np.sum(regionmask)
    geometric_centroid_y = np.sum(ytemp * regionmask) / np.sum(regionmask)
    distance = euclidean(np.array((geometric_centroid_y, geometric_centroid_x)), np.array((centroid_y, centroid_x)))
    delta_x = geometric_centroid_x - centroid_x
    delta_y = geometric_centroid_y - centroid_y
    direction_arctan = np.arctan2(delta_x, delta_y) * 180 / np.pi
    return distance, direction_arctan


def intensity_peripheral(regionmask, intensity_image):
    warnings.filterwarnings('ignore', message="Polyfit may be poorly conditioned")
    cell_mask = regionmask.copy()
    intensity = intensity_image.copy()
    y = intensity[cell_mask].flatten()
    x = distance_transform_edt(cell_mask)
    x = x[cell_mask].flatten()
    params = np.polyfit(x, y, 1)
    line = np.poly1d(params)

    return line.coefficients[0], line.coefficients[1]


def intensity_centroid_distance_edge(regionmask, intensity_image):
    edt = distance_transform_edt(regionmask)
    min_distance = 0
    max_distance = 0.1*edt.max()
    thresholded = (edt <= max_distance) * (edt > min_distance)
    edge_mask = np.copy(regionmask)
    edge_mask[np.where(thresholded == 0)] = 0
    y, x = np.mgrid[:edge_mask.shape[0], :edge_mask.shape[1]]
    xtemp = x.copy()
    ytemp = y.copy()
    intensity_edge = intensity_image.copy()
    intensity_edge[np.where(edge_mask == 0)] = 0.
    sum_intensity_edge = np.sum(intensity_edge)
    sum_regionmask = np.sum(regionmask)

    if sum_intensity_edge != 0 and sum_regionmask != 0:
        centroid_x = np.sum(xtemp * intensity_edge) / sum_intensity_edge
        centroid_y = np.sum(ytemp * intensity_edge) / sum_intensity_edge
        geometric_centroid_x = np.sum(xtemp * regionmask) / sum_regionmask
        geometric_centroid_y = np.sum(ytemp * regionmask) / sum_regionmask

        distance = euclidean((geometric_centroid_y, geometric_centroid_x), (centroid_y, centroid_x))
        delta_x = geometric_centroid_x - centroid_x
        delta_y = geometric_centroid_y - centroid_y
        direction_arctan = np.arctan2(delta_y, delta_x) * 180 / np.pi
        return distance, direction_arctan
    else:
        return np.nan, np.nan