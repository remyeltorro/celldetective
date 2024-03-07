"""
Add extra properties to directly to regionprops
Functions must take regionmask as first argument and optionally intensity_image as second argument
If intensity is in function name, it will be replaced by the name of the channel. These measurements are applied automatically to all channels

"""
import json

import matplotlib.pyplot as plt
import numpy as np
from lmfit import models
from scipy.spatial.distance import euclidean
from distributed.protocol import scipy
from scipy.ndimage.morphology import distance_transform_edt


# Percentiles

def intensity_percentile_99(regionmask, intensity_image):
    return np.nanpercentile(intensity_image[regionmask], 99)


def intensity_percentile_95(regionmask, intensity_image):
    return np.nanpercentile(intensity_image[regionmask], 95)


def intensity_percentile_90(regionmask, intensity_image):
    return np.nanpercentile(intensity_image[regionmask], 90)


def intensity_percentile_75(regionmask, intensity_image):
    return np.nanpercentile(intensity_image[regionmask], 75)


def intensity_percentile_50(regionmask, intensity_image):
    return np.nanpercentile(intensity_image[regionmask], 50)


def intensity_percentile_25(regionmask, intensity_image):
    return np.nanpercentile(intensity_image[regionmask], 25)


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
    # print(regionmask)
    # print(np.unique(regionmask))
    # for cell in np.unique(regionmask):
    # 	print(cell)
    # print(regionmask)
    cell_mask = regionmask.copy()
    # print(cell_mask[np.where(cell_mask==True)].shape)
    # print(intensity_image[np.where(cell_mask==True)].shape)
    # print(intensity_image[np.where(cell_mask == True)])
    # print(regionmask.shape)
    # print(cell_mask.shape)
    # print(cell_mask)
    # cell_mask[np.where(cell_mask != cell)] = 0
    intensity = intensity_image.copy()
    y = intensity[np.where(cell_mask == True)]
    # print(len(y))
    #print(type(y))
    x = distance_transform_edt(cell_mask[np.where(cell_mask == True)])
    #print(type(x))
    params = np.polyfit(x, y, 1)
    line = np.poly1d(params)
    plt.scatter(x, y, label='Data', alpha=0.1)
	#
    # plt.plot(x, line(x), color='red', label='Linear Fit')
    # plt.show()
    # print(line.coefficients[0])
    # print(line.coefficients[1])
    return line.coefficients[0], line.coefficients[1]
# model = models.LinearModel()
# result = model.fit(list(y), list(x))
# slope = result.params['slope'].value
# intercept = result.params['intercept'].value

# Scatter plot of distances vs intensity values
#     plt.scatter(x, y, label='Data',alpha=0.1)
#
#
#     plt.plot(x, line(x), color='red', label='Linear Fit')
# print(len(x))

# plt.show()
