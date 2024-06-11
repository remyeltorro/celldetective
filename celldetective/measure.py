import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from lmfit import Parameters, Model
import skimage.exposure
import tifffile
from lmfit import Parameters, Model, models
# import lmfit
from scipy import ndimage
from stardist import fill_label_holes
from tqdm import tqdm
from skimage.measure import regionprops_table
from scipy.ndimage.morphology import distance_transform_edt
from functools import reduce
from mahotas.features import haralick
from scipy.ndimage import zoom, binary_fill_holes
import os
import subprocess

from celldetective.filters import std_filter, gauss_filter
import datetime
from skimage.draw import disk as dsk

from celldetective.filters import std_filter, gauss_filter
from celldetective.utils import rename_intensity_column, create_patch_mask, remove_redundant_features, \
    remove_trajectory_measurements
import celldetective.extra_properties as extra_properties
from celldetective.extra_properties import *
import cv2
from inspect import getmembers, isfunction
from skimage.morphology import disk

abs_path = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], 'celldetective'])

def measure(stack=None, labels=None, trajectories=None, channel_names=None,
            features=None, intensity_measurement_radii=None, isotropic_operations=['mean'], border_distances=None,
            haralick_options=None, column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}, clear_previous=False):

    """

    Perform measurements on a stack of images or labels.

    Parameters
    ----------
    stack : numpy array, optional
        Stack of images with shape (T, Y, X, C), where T is the number of frames, Y and X are the spatial dimensions,
        and C is the number of channels. Default is None.
    labels : numpy array, optional
        Label stack with shape (T, Y, X) representing cell segmentations. Default is None.
    trajectories : pandas DataFrame, optional
        DataFrame of cell trajectories with columns specified in `column_labels`. Default is None.
    channel_names : list, optional
        List of channel names corresponding to the image stack. Default is None.
    features : list, optional
        List of features to measure using the `measure_features` function. Default is None.
    intensity_measurement_radii : int, float, or list, optional
        Radius or list of radii specifying the size of the isotropic measurement area for intensity measurements.
        If a single value is provided, a circular measurement area is used. If a list of values is provided, multiple
        measurements are performed using ring-shaped measurement areas. Default is None.
    isotropic_operations : list, optional
        List of operations to perform on the isotropic intensity values. Default is ['mean'].
    border_distances : int, float, or list, optional
        Distance or list of distances specifying the size of the border region for intensity measurements.
        If a single value is provided, measurements are performed at a fixed distance from the cell borders.
        If a list of values is provided, measurements are performed at multiple border distances. Default is None.
    haralick_options : dict, optional
        Dictionary of options for Haralick feature measurements. Default is None.
    column_labels : dict, optional
        Dictionary containing the column labels for the DataFrame. Default is {'track': "TRACK_ID",
        'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.

    Returns
    -------
    pandas DataFrame
        DataFrame containing the measured features and intensities.

    Notes
    -----
    This function performs measurements on a stack of images or labels. If both `stack` and `labels` are provided,
    measurements are performed on each frame of the stack. The measurements include isotropic intensity values, computed
    using the `measure_isotropic_intensity` function, and additional features, computed using the `measure_features` function.
    The intensity measurements are performed at the positions specified in the `trajectories` DataFrame, using the
    specified `intensity_measurement_radii` and `border_distances`. The resulting measurements are combined into a single
    DataFrame and returned.

    Examples
    --------
    >>> stack = np.random.rand(10, 100, 100, 3)
    >>> labels = np.random.randint(0, 2, (10, 100, 100))
    >>> trajectories = pd.DataFrame({'TRACK_ID': [1, 2, 3], 'FRAME': [1, 1, 1],
    ...                              'POSITION_X': [10, 20, 30], 'POSITION_Y': [15, 25, 35]})
    >>> channel_names = ['channel1', 'channel2', 'channel3']
    >>> features = ['area', 'intensity_mean']
    >>> intensity_measurement_radii = [5, 10]
    >>> border_distances = 2
    >>> measurements = measure(stack=stack, labels=labels, trajectories=trajectories, channel_names=channel_names,
    ...                        features=features, intensity_measurement_radii=intensity_measurement_radii,
    ...                        border_distances=border_distances)
    # Perform measurements on the stack, labels, and trajectories, computing isotropic intensities and additional features.

    """


    do_iso_intensities = True
    do_features = True


    # Check that conditions are satisfied to perform measurements
    assert (labels is not None) or (stack is not None),'Please pass a stack and/or labels... Abort.'
    if (labels is not None)*(stack is not None):
        assert labels.shape==stack.shape[:-1],f"Shape mismatch between the stack of shape {stack.shape} and the segmentation {labels.shape}..."

    # Condition to compute features
    if labels is None:
        do_features = False
        nbr_frames = len(stack)
        print('No labels were provided... Features will not be computed...')
    else:
        nbr_frames = len(labels)

    # Condition to compute isotropic intensities
    if (stack is None) or (trajectories is None) or (intensity_measurement_radii is None):
        do_iso_intensities = False
        print('Either no image, no positions or no radii were provided... Isotropic intensities will not be computed...')

    # Compensate for non provided channel names
    if (stack is not None)*(channel_names is None):
        nbr_channels = stack.shape[-1]
        channel_names = [f'intensity-{k}' for k in range(nbr_channels)]

    if isinstance(intensity_measurement_radii, int) or isinstance(intensity_measurement_radii, float):
        intensity_measurement_radii = [intensity_measurement_radii]

    if isinstance(border_distances, int) or isinstance(border_distances, float):
        border_distances = [border_distances]

    if features is not None:
        features = remove_redundant_features(features, trajectories.columns,
                                            channel_names=channel_names)

    if features is None:
        features = []

    # Prep for the case where no trajectory is provided but still want to measure isotropic intensities...
    if (trajectories is None):
        do_features = True
        features += ['centroid']
    else:
        if clear_previous:
            trajectories = remove_trajectory_measurements(trajectories, column_labels)

    timestep_dataframes = []

    for t in tqdm(range(nbr_frames),desc='frame'):

        if stack is not None:
            img = stack[t]
        else:
            img = None
        if labels is not None:
            lbl = labels[t]
        else:
            lbl = None

        if trajectories is not None:
            positions_at_t = trajectories.loc[trajectories[column_labels['time']]==t].copy()

        if do_features:
            feature_table = measure_features(img, lbl, features = features, border_dist=border_distances,
                                            channels=channel_names, haralick_options=haralick_options, verbose=False)
            if trajectories is None:
                # Use the centroids as estimate for the location of the cells, to be passed to the measure_isotropic_intensity function.
                positions_at_t = feature_table[['centroid-1', 'centroid-0','class_id']].copy()
                positions_at_t['ID'] = np.arange(len(positions_at_t))	# temporary ID for the cells, that will be reset at the end since they are not tracked
                positions_at_t.rename(columns={'centroid-1': 'POSITION_X', 'centroid-0': 'POSITION_Y'},inplace=True)
                positions_at_t['FRAME'] = int(t)
                column_labels = {'track': "ID", 'time': column_labels['time'], 'x': column_labels['x'], 'y': column_labels['y']}

        # Isotropic measurements (circle, ring)
        if do_iso_intensities:
            iso_table = measure_isotropic_intensity(positions_at_t, img, channels=channel_names, intensity_measurement_radii=intensity_measurement_radii,
                                                    column_labels=column_labels, operations=isotropic_operations, verbose=False)

        if do_iso_intensities*do_features:
            measurements_at_t = iso_table.merge(feature_table, how='outer', on='class_id')
        elif do_iso_intensities*(not do_features):
            measurements_at_t = iso_table
        elif do_features*(trajectories is not None):
            measurements_at_t = positions_at_t.merge(feature_table, how='outer', on='class_id')
        elif do_features*(trajectories is None):
            measurements_at_t = positions_at_t

        timestep_dataframes.append(measurements_at_t)

    measurements = pd.concat(timestep_dataframes)
    if trajectories is not None:
        measurements = measurements.sort_values(by=[column_labels['track'],column_labels['time']])
        measurements = measurements.dropna(subset=[column_labels['track']])
    else:
        measurements['ID'] = np.arange(len(df))

    measurements = measurements.reset_index(drop=True)

    return measurements

def write_first_detection_class(tab, column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}):

    tab = tab.sort_values(by=[column_labels['track'],column_labels['time']])
    if 'area' in tab.columns:
        for tid,track_group in tab.groupby(column_labels['track']):
            indices = track_group.index
            area = track_group['area'].values
            timeline = track_group[column_labels['time']].values
            if np.any(area==area):
                t_first = timeline[area==area][0]
                cclass = 1
                if t_first==0:
                    t_first = 0
                    cclass = 2
            else:
                t_first = -1
                cclass = 2

            tab.loc[indices, 'class_firstdetection'] = cclass
            tab.loc[indices, 't_firstdetection'] = t_first
    return tab


def contour_of_instance_segmentation(label, distance):

    """

    Generate an instance mask containing the contour of the segmented objects.

    Parameters
    ----------
    label : ndarray
        The instance segmentation labels.
    distance : int, float, list, or tuple
        The distance or range of distances from the edge of each instance to include in the contour.
        If a single value is provided, it represents the maximum distance. If a tuple or list is provided,
        it represents the minimum and maximum distances.

    Returns
    -------
    border_label : ndarray
        An instance mask containing the contour of the segmented objects.

    Notes
    -----
    This function generates an instance mask representing the contour of the segmented instances in the label image.
    It use the distance_transform_edt function from the scipy.ndimage module to compute the Euclidean distance transform.
    The contour is defined based on the specified distance(s) from the edge of each instance.
    The resulting mask, `border_label`, contains the contour regions, while the interior regions are set to zero.

    Examples
    --------
    >>> border_label = contour_of_instance_segmentation(label, distance=3)
    # Generate a binary mask containing the contour of the segmented instances with a maximum distance of 3 pixels.

    """
    if isinstance(distance,(list,tuple)) or distance >= 0 :

        edt = distance_transform_edt(label)

        if isinstance(distance, list) or isinstance(distance, tuple):
            min_distance = distance[0]; max_distance = distance[1]

        elif isinstance(distance, (int, float)):
            min_distance = 0
            max_distance = distance

        thresholded = (edt <= max_distance) * (edt > min_distance)
        border_label = np.copy(label)
        border_label[np.where(thresholded == 0)] = 0

    else:
        size = (2*abs(int(distance))+1, 2*abs(int(distance))+1)
        dilated_image = ndimage.grey_dilation(label, footprint=disk(int(abs(distance)))) #size=size,
        border_label=np.copy(dilated_image)
        matching_cells = np.logical_and(dilated_image != 0, label == dilated_image)
        border_label[np.where(matching_cells == True)] = 0
        border_label[label!=0] = 0.

    return border_label

def drop_tonal_features(features):

    """
    Removes features related to intensity from a list of feature names.

    This function iterates over a list of feature names and removes any feature that includes the term 'intensity' in its name.
    The operation is performed in-place, meaning the original list of features is modified directly.

    Parameters
    ----------
    features : list of str
        A list of feature names from which intensity-related features are to be removed.

    Returns
    -------
    list of str
        The modified list of feature names with intensity-related features removed. Note that this operation modifies the
        input list in-place, so the return value is the same list object with some elements removed.

    """

    feat2 = features[:]
    for f in features:
        if 'intensity' in f:
            feat2.remove(f)
    return feat2

def measure_features(img, label, features=['area', 'intensity_mean'], channels=None,
                     border_dist=None, haralick_options=None, verbose=True, normalisation_list=None,
                     radial_intensity=None,
                     radial_channel=None, spot_detection=None):
    """

    Measure features within segmented regions of an image.

    Parameters
    ----------
    img : ndarray
        The input image as a NumPy array.
    label : ndarray
        The segmentation labels corresponding to the image regions.
    features : list, optional
        The list of features to measure within the segmented regions. The default is ['area', 'intensity_mean'].
    channels : list, optional
        The list of channel names in the image. The default is ["brightfield_channel", "dead_nuclei_channel", "live_nuclei_channel"].
    border_dist : int, float, or list, optional
        The distance(s) in pixels from the edge of each segmented region to measure features. The default is None.
    haralick_options : dict, optional
        The options for computing Haralick features. The default is None.

    Returns
    -------
    df_props : DataFrame
        A pandas DataFrame containing the measured features for each segmented region.

    Notes
    -----
    This function measures features within segmented regions of an image.
    It utilizes the regionprops_table function from the skimage.measure module for feature extraction.
    The features to measure can be specified using the 'features' parameter.
    Optional parameters such as 'channels' and 'border_dist' allow for additional measurements.
    If provided, Haralick features can be computed using the 'haralick_options' parameter.
    The results are returned as a pandas DataFrame.

    Examples
    --------
    >>> df_props = measure_features(img, label, features=['area', 'intensity_mean'], channels=["brightfield_channel", "dead_nuclei_channel", "live_nuclei_channel"])
    # Measure area and mean intensity within segmented regions of the image.

    """

    if features is None:
        features = []

    # Add label to have identity of mask
    if 'label' not in features:
        features.append('label')

    if img is None:
        if verbose:
            print('No image was provided... Skip intensity measurements.')
        border_dist = None;
        haralick_options = None;
        features = drop_tonal_features(features)
    if img is not None:
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        if channels is None:
            channels = [f'intensity-{k}' for k in range(img.shape[-1])]
        if (channels is not None) * (img.ndim == 3):
            assert len(channels) == img.shape[
                -1], "Mismatch between the provided channel names and the shape of the image"

        if spot_detection is not None:
            for index, channel in enumerate(channels):
                if channel == spot_detection['channel']:
                    ind = index
                    blobs = blob_detection(img[:, :, ind], label, diameter=spot_detection['diameter'],
                                   threshold=spot_detection['threshold'])
                    df_spots = pd.DataFrame.from_dict(blobs, orient='index',
                                                      columns=['count', 'spot_mean_intensity']).reset_index()
            # Rename columns
            df_spots.columns = ['label', 'spot_count', 'spot_mean_intensity']
        
        if normalisation_list:
            for norm in normalisation_list:
                for index, channel in enumerate(channels):
                    if channel == norm['target_channel']:
                        ind = index
                if norm['correction_type'] == 'local':
                    normalised_image = normalise_by_cell(img[:, :, ind].copy(), label,
                                                         distance=int(norm['distance']), model=norm['model'],
                                                         operation=norm['operation'], clip=norm['clip'])
                    img[:, :, ind] = normalised_image
                else:
                    corrected_image = field_correction(img[:,:,ind].copy(), threshold_on_std=norm['threshold_on_std'], operation=norm['operation'], model=norm['model'], clip=norm['clip'])
                    img[:, :, ind] = corrected_image

    extra_props = getmembers(extra_properties, isfunction)
    extra_props = [extra_props[i][0] for i in range(len(extra_props))]

    extra_props_list = []
    feats = features.copy()
    for f in features:
        if f in extra_props:
            feats.remove(f)
            extra_props_list.append(getattr(extra_properties, f))
    if len(extra_props_list) == 0:
        extra_props_list = None
    else:
        extra_props_list = tuple(extra_props_list)
    props = regionprops_table(label, intensity_image=img, properties=feats, extra_properties=extra_props_list)
    df_props = pd.DataFrame(props)
    if spot_detection is not None:
        df_props = df_props.merge(df_spots, how='outer', on='label')
        df_props['spot_count'] = df_props['spot_count'].replace(np.nan, 0)
        df_props['spot_mean_intensity'] = df_props['spot_mean_intensity'].replace(np.nan, 0)



    # if spot_detection is not None:
    #     for index, channel in enumerate(channels):
    #         if channel == spot_detection['channel']:
    #             ind = index
    #     blobs = blob_detection(img[:, :, ind], label, diameter=spot_detection['diameter'],
    #                            threshold=spot_detection['threshold'])
    #     df_spots = pd.DataFrame.from_dict(blobs, orient='index', columns=['count', 'spot_mean_intensity']).reset_index()
    #     # Rename columns
    #     df_spots.columns = ['label', 'spot_count', 'spot_mean_intensity']
    #     df_props = df_props.merge(df_spots, how='outer', on='label')
    #     df_props['spot_count'] = df_props['spot_count'].replace(np.nan, 0)
    #     df_props['spot_mean_intensity'] = df_props['spot_mean_intensity'].replace(np.nan, 0)


    if border_dist is not None:
        # automatically drop all non intensity features
        intensity_features_test = [('intensity' in s and 'centroid' not in s and 'peripheral' not in s) for s in
                                   features]
        intensity_features = list(np.array(features)[np.array(intensity_features_test)])
        # intensity_extra = [(s in extra_props_list)for s in intensity_features]
        # print(intensity_extra)
        intensity_extra = []
        for s in intensity_features:
            if s in extra_props:
                intensity_extra.append(getattr(extra_properties, s))
                intensity_features.remove(s)
        # print(intensity_features)
        # If no intensity feature was passed still measure mean intensity
        if len(intensity_features) == 0:
            if verbose:
                print('No intensity feature was passed... Adding mean intensity for edge measurement...')
            intensity_features = np.append(intensity_features, 'intensity_mean')
        intensity_features = list(np.append(intensity_features, 'label'))

        # Remove extra intensity properties from border measurements
        new_intensity_features = intensity_features.copy()
        for int_feat in intensity_features:
            if int_feat in extra_props:
                new_intensity_features.remove(int_feat)
        intensity_features = new_intensity_features

        if (isinstance(border_dist, int) or isinstance(border_dist, float)):
            border_label = contour_of_instance_segmentation(label, border_dist)
            props_border = regionprops_table(border_label, intensity_image=img, properties=intensity_features)
            df_props_border = pd.DataFrame(props_border)
            for c in df_props_border.columns:
                if 'intensity' in c:
                    df_props_border = df_props_border.rename({c: c+f'_edge_{border_dist}px'},axis=1)

        if isinstance(border_dist, list):
            df_props_border_list = []
            for d in border_dist:
                border_label = contour_of_instance_segmentation(label, d)
                props_border = regionprops_table(border_label, intensity_image=img, properties=intensity_features)
                df_props_border_d = pd.DataFrame(props_border)
                for c in df_props_border_d.columns:
                    if 'intensity' in c:
                        if '-' in str(d):
                            df_props_border_d = df_props_border_d.rename({c: c + f'_outer_edge_{d}px'}, axis=1)
                        else:
                            df_props_border_d = df_props_border_d.rename({c: c + f'_edge_{d}px'}, axis=1)
                df_props_border_list.append(df_props_border_d)

            df_props_border = reduce(lambda  left,right: pd.merge(left,right,on=['label'],
                                            how='outer'), df_props_border_list)

        df_props = df_props.merge(df_props_border, how='outer', on='label')

    if haralick_options is not None:
        try:
            df_haralick = compute_haralick_features(img, label, channels=channels, **haralick_options)
            df_props = df_props.merge(df_haralick, left_on='label',right_on='cell_id')
            #df_props = df_props.drop(columns=['cell_label'])
        except Exception as e:
            print(e)
            pass

    if channels is not None:
        df_props = rename_intensity_column(df_props, channels)
    df_props.rename(columns={"label": "class_id"},inplace=True)
    df_props['class_id'] = df_props['class_id'].astype(float)

    return df_props

def compute_haralick_features(img, labels, channels=None, target_channel=0, scale_factor=1, percentiles=(0.01,99.99), clip_values=None,
                                n_intensity_bins=256, ignore_zero=True, return_mean=True, return_mean_ptp=False, distance=1, disable_progress_bar=False, return_norm_image_only=False, return_digit_image_only=False):

    """

    Compute Haralick texture features on each segmented region of an image.

    Parameters
    ----------
    img : ndarray
        The input image as a NumPy array.
    labels : ndarray
        The segmentation labels corresponding to the image regions.
    target_channel : int, optional
        The target channel index of the image. The default is 0.
    modality : str, optional
        The modality or channel type of the image. The default is 'brightfield_channel'.
    scale_factor : float, optional
        The scale factor for resampling the image and labels. The default is 1.
    percentiles : tuple of float, optional
        The percentiles to use for image normalization. The default is (0.01, 99.99).
    clip_values : tuple of float, optional
        The minimum and maximum values to clip the image. If None, percentiles are used. The default is None.
    n_intensity_bins : int, optional
        The number of intensity bins for image normalization. The default is 255.
    ignore_zero : bool, optional
        Flag indicating whether to ignore zero values during feature computation. The default is True.
    return_mean : bool, optional
        Flag indicating whether to return the mean value of each Haralick feature. The default is True.
    return_mean_ptp : bool, optional
        Flag indicating whether to return the mean and peak-to-peak values of each Haralick feature. The default is False.
    distance : int, optional
        The distance parameter for Haralick feature computation. The default is 1.

    Returns
    -------
    features : DataFrame
        A pandas DataFrame containing the computed Haralick features for each segmented region.

    Notes
    -----
    This function computes Haralick features on an image within segmented regions.
    It uses the mahotas library for feature extraction and pandas DataFrame for storage.
    The image is rescaled, normalized and digitized based on the specified parameters.
    Haralick features are computed for each segmented region, and the results are returned as a DataFrame.

    Examples
    --------
    >>> features = compute_haralick_features(img, labels, target_channel=0, modality="brightfield_channel")
    # Compute Haralick features on the image within segmented regions.

    """

    assert ((img.ndim==2)|(img.ndim==3)),f'Invalid image shape to compute the Haralick features. Expected YXC, got {img.shape}...'
    assert img.shape[:2]==labels.shape,f'Mismatch between image shape {img.shape} and labels shape {labels.shape}'

    if img.ndim==2:
        img = img[:,:,np.newaxis]
        target_channel = 0
        if isinstance(channels, list):
            modality = channels[0]
        elif isinstance(channels, str):
            modality = channels
        else:
            print('Channel name unrecognized...')
            modality=''
    elif img.ndim==3:
        assert target_channel is not None,"The image is multichannel. Please provide a target channel to compute the Haralick features. Abort."
        modality = channels[target_channel]

    haralick_labels = ["angular_second_moment",
                       "contrast",
                       "correlation",
                       "sum_of_square_variance",
                       "inverse_difference_moment",
                       "sum_average",
                       "sum_variance",
                       "sum_entropy",
                       "entropy",
                       "difference_variance",
                       "difference_entropy",
                       "information_measure_of_correlation_1",
                       "information_measure_of_correlation_2",
                       "maximal_correlation_coefficient"]

    haralick_labels = ['haralick_'+h+"_"+modality for h in haralick_labels]
    if len(img.shape)==3:
        img = img[:,:,target_channel]

    # Rescale image and mask
    img = zoom(img,[scale_factor,scale_factor],order=3).astype(float)
    labels = zoom(labels, [scale_factor,scale_factor],order=0)

    # Normalize image
    if clip_values is None:
        min_value = np.nanpercentile(img[img!=0.].flatten(), percentiles[0])
        max_value = np.nanpercentile(img[img!=0.].flatten(), percentiles[1])
    else:
        min_value = clip_values[0]; max_value = clip_values[1]

    img -= min_value
    img /= (max_value-min_value) / n_intensity_bins
    img[img<=0.] = 0.
    img[img>=n_intensity_bins] = n_intensity_bins

    if return_norm_image_only:
        return img

    hist,bins = np.histogram(img.flatten(),bins=n_intensity_bins)
    centered_bins = [bins[0]] + [bins[i] + (bins[i+1] - bins[i])/2. for i in range(len(bins)-1)]

    digitized = np.digitize(img, bins)
    img_binned = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_binned[i,j] = centered_bins[digitized[i,j] - 1]

    img = img_binned.astype(int)
    if return_digit_image_only:
        return img

    haralick_properties = []

    for cell in tqdm(np.unique(labels)[1:],disable=disable_progress_bar):

        mask = labels==cell
        f = img*mask
        features = haralick(f, ignore_zeros=ignore_zero,return_mean=return_mean,distance=distance)

        dictionary = {'cell_id': cell}
        for k in range(len(features)):
            dictionary.update({haralick_labels[k]: features[k]})
        haralick_properties.append(dictionary)

    assert len(haralick_properties)==(len(np.unique(labels))-1),'Some cells have not been measured...'

    return pd.DataFrame(haralick_properties)


def measure_isotropic_intensity(positions, # Dataframe of cell positions @ t
                                 img,  # multichannel frame (YXC) @ t
                                 channels=None, #channels, need labels to name measurements
                                 intensity_measurement_radii=None, #list of radii, single value is circle, tuple is ring?
                                 operations = ['mean'],
                                 measurement_kernel = None,
                                 pbar=None,
                                 column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'},
                                 verbose=True,
                                ):

    """

    Measure isotropic intensity values around cell positions in an image.

    Parameters
    ----------
    positions : pandas DataFrame
        DataFrame of cell positions at time 't' containing columns specified in `column_labels`.
    img : numpy array
        Multichannel frame (YXC) at time 't' used for intensity measurement.
    channels : list or str, optional
        List of channel names corresponding to the image channels. Default is None.
    intensity_measurement_radii : int, list, or tuple
        Radius or list of radii specifying the size of the isotropic measurement area.
        If a single value is provided, a circular measurement area is used. If a list or tuple of two values
        is provided, a ring-shaped measurement area is used. Default is None.
    operations : list, optional
        List of operations to perform on the intensity values. Default is ['mean'].
    measurement_kernel : numpy array, optional
        Kernel used for intensity measurement. If None, a circular or ring-shaped kernel is generated
        based on the provided `intensity_measurement_radii`. Default is None.
    pbar : tqdm progress bar, optional
        Progress bar for tracking the measurement process. Default is None.
    column_labels : dict, optional
        Dictionary containing the column labels for the DataFrame. Default is {'track': "TRACK_ID",
        'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.
    verbose : bool, optional
        If True, enables verbose output. Default is True.

    Returns
    -------
    pandas DataFrame
        The updated DataFrame `positions` with additional columns representing the measured intensity values.

    Notes
    -----
    This function measures the isotropic intensity values around the cell positions specified in the `positions`
    DataFrame using the provided image `img`. The intensity measurements are performed using circular or ring-shaped
    measurement areas defined by the `intensity_measurement_radii`. The measurements are calculated for each channel
    specified in the `channels` list. The resulting intensity values are stored in additional columns of the `positions`
    DataFrame. The `operations` parameter allows specifying different operations to be performed on the intensity
    values, such as 'mean', 'median', etc. The measurement kernel can be customized by providing the `measurement_kernel`
    parameter. If not provided, the measurement kernel is automatically generated based on the `intensity_measurement_radii`.
    The progress bar `pbar` can be used to track the measurement process. The `column_labels` dictionary is used to
    specify the column labels for the DataFrame.

    Examples
    --------
    >>> positions = pd.DataFrame({'TRACK_ID': [1, 2, 3], 'FRAME': [1, 1, 1],
    ...                           'POSITION_X': [10, 20, 30], 'POSITION_Y': [15, 25, 35]})
    >>> img = np.random.rand(100, 100, 3)
    >>> channels = ['channel1', 'channel2', 'channel3']
    >>> intensity_measurement_radii = 5
    >>> positions = measure_isotropic_intensity(positions, img, channels=channels,
    ...                                         intensity_measurement_radii=intensity_measurement_radii)
    # Measure isotropic intensity values around cell positions in the image.

    """

    epsilon = -10000
    assert ((img.ndim==2)|(img.ndim==3)),f'Invalid image shape to compute the Haralick features. Expected YXC, got {img.shape}...'

    print(img.shape, 'before measurement')

    if img.ndim==2:
        img = img[:,:,np.newaxis]
        if isinstance(channels, str):
            channels = [channels]
        else:
            if verbose:
                print('Channel name unrecognized...')
            channels=['intensity']
    elif img.ndim==3:
        assert channels is not None,"The image is multichannel. Please provide the list of channel names. Abort."

    if isinstance(intensity_measurement_radii, int) or isinstance(intensity_measurement_radii, float):
        intensity_measurement_radii = [intensity_measurement_radii]

    if (measurement_kernel is None)*(intensity_measurement_radii is not None):

        for r in intensity_measurement_radii:

            if isinstance(r,list):
                mask = create_patch_mask(2*max(r)+1,2*max(r)+1,((2*max(r))//2,(2*max(r))//2),radius=r)
            else:
                mask = create_patch_mask(2*r+1,2*r+1,((2*r)//2,(2*r)//2),r)

            pad_value_x = mask.shape[0]//2 + 1
            pad_value_y = mask.shape[1]//2 + 1
            frame_padded = np.pad(img.astype(float), [(pad_value_x,pad_value_x),(pad_value_y,pad_value_y),(0,0)], constant_values=[(epsilon,epsilon),(epsilon,epsilon),(0,0)])

            # Find a way to measure intensity in mask
            for tid,group in positions.groupby(column_labels['track']):

                x = group[column_labels['x']].to_numpy()[0]
                y = group[column_labels['y']].to_numpy()[0]

                xmin = int(x)
                xmax = int(x) + 2*pad_value_y - 1
                ymin = int(y)
                ymax = int(y) + 2*pad_value_x - 1

                assert frame_padded[ymin:ymax,xmin:xmax,0].shape == mask.shape,"Shape mismatch between the measurement kernel and the image..."

                expanded_mask = np.expand_dims(mask, axis=-1)  # shape: (X, Y, 1)
                crop = frame_padded[ymin:ymax,xmin:xmax]

                crop_temp = crop.copy()
                crop_temp[crop_temp==epsilon] = 0.
                projection = np.multiply(crop_temp, expanded_mask)

                projection[crop==epsilon] = epsilon
                projection[expanded_mask[:,:,0]==0.,:] = epsilon

                for op in operations:
                    func = eval('np.'+op)
                    intensity_values = func(projection, axis=(0,1), where=projection>epsilon)
                    for k in range(crop.shape[-1]):
                        if isinstance(r,list):
                            positions.loc[group.index, f'{channels[k]}_ring_{min(r)}_{max(r)}_{op}'] = intensity_values[k]
                        else:
                            positions.loc[group.index, f'{channels[k]}_circle_{r}_{op}'] = intensity_values[k]

    elif (measurement_kernel is not None):
        # do something like this
        mask = measurement_kernel
        pad_value_x = mask.shape[0]//2 + 1
        pad_value_y = mask.shape[1]//2 + 1
        frame_padded = np.pad(img, [(pad_value_x,pad_value_x),(pad_value_y,pad_value_y),(0,0)])

        for tid,group in positions.groupby(column_labels['track']):

            x = group[column_labels['x']].to_numpy()[0]
            y = group[column_labels['y']].to_numpy()[0]

            xmin = int(x)
            xmax = int(x) + 2*pad_value_y - 1
            ymin = int(y)
            ymax = int(y) + 2*pad_value_x - 1

            assert frame_padded[ymin:ymax,xmin:xmax,0].shape == mask.shape,"Shape mismatch between the measurement kernel and the image..."

            expanded_mask = np.expand_dims(mask, axis=-1)  # shape: (X, Y, 1)
            crop = frame_padded[ymin:ymax,xmin:xmax]
            projection = np.multiply(crop, expanded_mask)

            for op in operations:
                func = eval('np.'+op)
                intensity_values = func(projection, axis=(0,1), where=projection==projection)
                for k in range(crop.shape[-1]):
                    positions.loc[group.index, f'{channels[k]}_custom_kernel_{op}'] = intensity_values[k]

    if pbar is not None:
        pbar.update(1)
    positions['class_id'] = positions['class_id'].astype(float)
    return positions

def measure_at_position(pos, mode, return_measurements=False, threads=1):

    """
    Executes a measurement script at a specified position directory, optionally returning the measured data.

    This function calls an external Python script to perform measurements on data
    located in a specified position directory. The measurement mode determines the type of analysis performed by the script.
    The function can either return the path to the resulting measurements table or load and return the measurements as a
    pandas DataFrame.

    Parameters
    ----------
    pos : str
        The path to the position directory where the measurements should be performed. The path should be a valid directory.
    mode : str
        The measurement mode to be used by the script. This determines the type of analysis performed (e.g., 'tracking',
        'feature_extraction').
    return_measurements : bool, optional
        If True, the function loads the resulting measurements from a CSV file into a pandas DataFrame and returns it. If
        False, the function returns None (default is False).

    Returns
    -------
    pandas.DataFrame or None
        If `return_measurements` is True, returns a pandas DataFrame containing the measurements. Otherwise, returns None.

    """

    pos = pos.replace('\\','/')
    pos = rf"{pos}"
    assert os.path.exists(pos),f'Position {pos} is not a valid path.'
    if not pos.endswith('/'):
        pos += '/'
    script_path = os.sep.join([abs_path, 'scripts', 'measure_cells.py'])
    cmd = f'python "{script_path}" --pos "{pos}" --mode "{mode}" --threads "{threads}"'
    subprocess.call(cmd, shell=True)

    table = pos + os.sep.join(["output","tables",f"trajectories_{mode}.csv"])
    if return_measurements:
        df = pd.read_csv(table)
        return df
    else:
        return None


def local_normalisation(image, labels, background_intensity, measurement='intensity_median', operation='subtract', clip=False):
    """
     Perform local normalization on an image based on labels.

     Parameters:
     - image (numpy.ndarray): The input image.
     - labels (numpy.ndarray): An array specifying the labels for different regions in the image.
     - background_intensity (pandas.DataFrame): A DataFrame containing background intensity values
                                                 corresponding to each label.
     - mode (str): The normalization mode ('Mean' or 'Median').
     - operation (str): The operation to perform ('Subtract' or 'Divide').

     Returns:
     - numpy.ndarray: The normalized image.

     This function performs local normalization on an image based on the provided labels. It iterates over
     each unique label, excluding the background label (0), and performs the specified operation with the
     background intensity values corresponding to that label. The background intensity values are obtained
     from the provided background_intensity DataFrame based on the normalization mode.

     If the operation is 'Subtract', the background intensity is subtracted from the image pixel values.
     If the operation is 'Divide', the image pixel values are divided by the background intensity.

     Example:
     >>> image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
     >>> labels = np.array([[0, 1, 1], [2, 2, 3], [3, 3, 0]])
     >>> background_intensity = pd.DataFrame({'intensity_mean': [10, 20, 30]})
     >>> mode = 'Mean'
     >>> operation = 'Subtract'
     >>> result = local_normalisation(image, labels, background_intensity, mode, operation)
     >>> print(result)
     [[-9. -8. -7.]
      [14. 15.  6.]
      [27. 28.  9.]]

     Note:
     - The background intensity DataFrame should have columns named 'intensity_mean' or 'intensity_median'
       based on the mode specified.
     - The background intensity values should be provided in the same order as the labels.
     """
    
    for index, cell in enumerate(np.unique(labels)):
        if cell == 0:
            continue
        if operation == 'subtract':
            image[np.where(labels == cell)] = image[np.where(labels == cell)].astype(float) - \
                                              background_intensity[measurement][index-1].astype(float)
        elif operation == 'divide':
            image[np.where(labels == cell)] = image[np.where(labels == cell)].astype(float) / \
                                              background_intensity[measurement][index-1].astype(float)
    if clip:
        image[image<=0.] = 0.

    return image.astype(float)


def normalise_by_cell(image, labels, distance=5, model='median', operation='subtract', clip=False):
    """
    Normalize an image based on cell regions.

    Parameters:
    - image (numpy.ndarray): The input image.
    - labels (numpy.ndarray): An array specifying the labels for different regions in the image.
    - distance (float): The distance parameter for finding the contour of cell regions.
    - mode (str): The normalization mode ('Mean' or 'Median').
    - operation (str): The operation to perform ('Subtract' or 'Divide').

    Returns:
    - numpy.ndarray: The normalized image.

    This function normalizes an image based on cell regions defined by the provided labels. It calculates
    the border of cell regions using the contour_of_instance_segmentation function with the specified
    distance parameter. Then, it computes the background intensity of each cell region based on the mode
    ('Mean' or 'Median'). Finally, it performs local normalization using the local_normalisation function
    and returns the normalized image.

    Example:
    >>> image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> labels = np.array([[0, 1, 1], [2, 2, 3], [3, 3, 0]])
    >>> distance = 2.0
    >>> mode = 'Mean'
    >>> operation = 'Subtract'
    >>> result = normalise_by_cell(image, labels, distance, mode, operation)
    >>> print(result)
    [[-9. -8. -7.]
     [14. 15.  6.]
     [27. 28.  9.]]

    Note:
    - The contour of cell regions is calculated using the contour_of_instance_segmentation function.
    - The background intensity is computed based on the specified mode ('Mean' or 'Median').
    - The operation determines whether to subtract or divide the background intensity from the image.
    """
    border = contour_of_instance_segmentation(label=labels, distance=distance * (-1))
    if model == 'mean':
        measurement = 'intensity_nanmean'
        extra_props = [getattr(extra_properties, measurement)]
        background_intensity = regionprops_table(intensity_image=image, label_image=border,
                                                 extra_properties=extra_props)
    elif model == 'median':
        measurement = 'intensity_median'
        extra_props = [getattr(extra_properties, measurement)]
        background_intensity = regionprops_table(intensity_image=image, label_image=border,
                                                 extra_properties=extra_props)

    normalised_frame = local_normalisation(image=image.astype(float).copy(),
                                           labels=labels, background_intensity=background_intensity, measurement=measurement,
                                           operation=operation, clip=clip)

    return normalised_frame


def paraboloid(x, y, a, b, c, d, e, g):
    return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + g


def plane(x, y, a, b, c):
    return a * x + b * y + c


def fit_plane(image, cell_masks=None):
    """
    Fit a plane to the given image data.

    Parameters:
    - image (numpy.ndarray): The input image data.
    - cell_masks (numpy.ndarray, optional): An array specifying cell masks. If provided, areas covered by
                                            cell masks will be excluded from the fitting process.

    Returns:
    - numpy.ndarray: The fitted plane.

    This function fits a plane to the given image data using least squares regression. It constructs a mesh
    grid based on the dimensions of the image and fits a plane model to the data points. If cell masks are
    provided, areas covered by cell masks will be excluded from the fitting process.

    Example:
    >>> image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> result = fit_plane(image)
    >>> print(result)
    [[1. 2. 3.]
     [4. 5. 6.]
     [7. 8. 9.]]

    Note:
    - The 'cell_masks' parameter allows excluding areas covered by cell masks from the fitting process.
    """
    data = np.empty(image.shape)
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])
    xx, yy = np.meshgrid(x, y)
    params = Parameters()
    params.add('a', value=1)
    params.add('b', value=1)
    params.add('c', value=1)
    model = Model(plane, independent_vars=['x', 'y'])
    weights = np.ones_like(xx, dtype=float)
    weights[np.where(cell_masks > 0)] = 0.
    result = model.fit(image,
                       x=xx,
                       y=yy,
                       weights=weights,
                       params=params, max_nfev=3000)
    a = result.params['a'].value
    b = result.params['b'].value
    c = result.params['c'].value
    fitted_plane = plane(xx, yy, a, b, c)
    return fitted_plane


def fit_paraboloid(image, cell_masks=None):
    """
    Fit a paraboloid to the given image data.

    Parameters:
    - image (numpy.ndarray): The input image data.
    - cell_masks (numpy.ndarray, optional): An array specifying cell masks. If provided, areas covered by
                                            cell masks will be excluded from the fitting process.

    Returns:
    - numpy.ndarray: The fitted paraboloid.

    This function fits a paraboloid to the given image data using least squares regression. It constructs
    a mesh grid based on the dimensions of the image and fits a paraboloid model to the data points. If cell
    masks are provided, areas covered by cell masks will be excluded from the fitting process.

    Example:
    >>> image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> result = fit_paraboloid(image)
    >>> print(result)
    [[1. 2. 3.]
     [4. 5. 6.]
     [7. 8. 9.]]

    Note:
    - The 'cell_masks' parameter allows excluding areas covered by cell masks from the fitting process.
    """
    data = np.empty(image.shape)
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])
    xx, yy = np.meshgrid(x, y)
    params = Parameters()
    params.add('a', value=1)
    params.add('b', value=1)
    params.add('c', value=1)
    params.add('d', value=1)
    params.add('e', value=1)
    params.add('g', value=1)
    model = Model(paraboloid, independent_vars=['x', 'y'])
    weights = np.ones_like(xx, dtype=float)
    weights[np.where(cell_masks > 0)] = 0.

    result = model.fit(image,
                       x=xx,
                       y=yy,
                       weights=weights,
                       params=params, max_nfev=3000)
    a = result.params['a'].value
    b = result.params['b'].value
    c = result.params['c'].value
    d = result.params['d'].value
    e = result.params['e'].value
    g = result.params['g'].value
    fitted_paraboloid = paraboloid(xx, yy, a, b, c, d, e, g)
    return fitted_paraboloid


def correct_image(img, cell_masks=None, normalisation_operation=None, clip=False, mode=None):
    """
    Correct an image based on fitted models.

    Parameters:
    - img (numpy.ndarray): The input image data.
    - cell_masks (numpy.ndarray, optional): An array specifying cell masks. If provided, areas covered by
                                            cell masks will be considered during correction.
    - normalisation_operation (str, optional): The normalisation operation ('Subtract' or 'Divide') to apply
                                               during correction.
    - clip (bool, optional): Whether to clip corrected values below zero to a minimum value of 0.00001.
    - mode (str, optional): The mode of correction ('Paraboloid' or 'Plane').

    Returns:
    - tuple: A tuple containing the corrected image and the fitted model parameters.

    This function corrects an image based on fitted models such as paraboloid or plane. It first fits a model
    to the image data based on the specified mode. Then, it performs correction by subtracting or dividing the
    image by the fitted model. Optionally, it clips corrected values below zero to a minimum value of 0.00001.

    Example:
    >>> img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> correction, para = correct_image(img, mode='Paraboloid', normalisation_operation='Subtract', clip=True)
    >>> print(correction)
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    >>> print(para)
    [[1. 2. 3.]
     [4. 5. 6.]
     [7. 8. 9.]]

    Note:
    - The 'cell_masks' parameter allows considering areas covered by cell masks during correction.
    - The 'normalisation_operation' parameter specifies whether to subtract or divide the fitted model from
      the image.
    - If 'clip' is set to True, corrected values below zero will be clipped to a minimum value of 0.00001.
    """
    if mode == "Paraboloid":
        para = fit_paraboloid(img.astype(float), cell_masks=cell_masks).astype(float)
    elif mode == "Plane":
        para = fit_plane(img.astype(float), cell_masks=cell_masks).astype(float)

    if para is not None:
        para = np.array(para)
        if normalisation_operation == 'Subtract':
            correction = img.astype(float) - para.astype(float)  # + 1000.
        else:
            correction = img.astype(float) / para.astype(float)  # + 1000.
        correction = np.array(correction, dtype=float)
        if clip:
            correction[correction <= 0] = 0.00001

    else:
        correction = None

    return correction, para


def field_normalisation(img, threshold, normalisation_operation, clip, mode):
    """
        Perform field normalization on an image.

        Parameters:
        - img (numpy.ndarray): The input image data.
        - threshold (float): The threshold value for determining regions of interest.
        - normalisation_operation (str): The normalization operation ('Subtract' or 'Divide') to apply during correction.
        - clip (bool): Whether to clip corrected values below zero to a minimum value of 0.00001.
        - mode (str): The mode of correction ('Paraboloid' or 'Plane').

        Returns:
        - tuple: A tuple containing the normalized image and the fitted background model.

        This function performs field normalization on an image based on regions of interest determined by the
        specified threshold. It identifies regions with standard deviation above the threshold and considers them
        as areas of interest. Then, it corrects the image using the 'correct_image' function based on the specified
        mode and normalization operation.

        Example:
        >>> img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> fluo_max, bg_fit = field_normalisation(img, threshold=0.5, normalisation_operation='Subtract', clip=True, mode='Paraboloid')
        >>> print(fluo_max)
        [[0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]]
        >>> print(bg_fit)
        [[1. 2. 3.]
         [4. 5. 6.]
         [7. 8. 9.]]

        Note:
        - The 'threshold' parameter determines regions of interest based on standard deviation.
        - The 'normalisation_operation' parameter specifies whether to subtract or divide the fitted model from
          the image during correction.
        - If 'clip' is set to True, corrected values below zero will be clipped to a minimum value of 0.00001.
        """
    std_img = std_filter(gauss_filter(img, 2), 4)
    mask = np.zeros_like(img)
    if threshold=='':
        pass
    mask[np.where(std_img > float(threshold))] = 1.0
    mask_int = mask.astype(int)
    mask_int = binary_fill_holes(mask_int).astype(float)
    # invert_mask = np.zeros_like(mask_int)
    # invert_mask[mask_int == 0] = 1
    # if isinstance(normalisation_operation,bool) and normalisation_operation:
    #     normalisation_operation = 'Subtract'
    # else:
    #     normalisation_operation = 'Divide'
    fluo_max, bg_fit = correct_image(img.astype(float), cell_masks=mask_int,
                                     normalisation_operation=normalisation_operation, clip=clip, mode=mode)
    return fluo_max, bg_fit


def blob_detection(image, label, threshold, diameter):
    """
    Perform blob detection on an image based on labeled regions.

    Parameters:
    - image (numpy.ndarray): The input image data.
    - label (numpy.ndarray): An array specifying labeled regions in the image.
    - threshold (float): The threshold value for blob detection.
    - diameter (float): The expected diameter of blobs.

    Returns:
    - dict: A dictionary containing information about detected blobs.

    This function performs blob detection on an image based on labeled regions. It iterates over each labeled region
    and detects blobs within the region using the Difference of Gaussians (DoG) method. Detected blobs are filtered
    based on the specified threshold and expected diameter. The function returns a dictionary containing the number of
    detected blobs and their mean intensity for each labeled region.

    Example:
    >>> image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> label = np.array([[0, 1, 1], [2, 2, 0], [3, 3, 0]])
    >>> threshold = 0.1
    >>> diameter = 5.0
    >>> result = blob_detection(image, label, threshold, diameter)
    >>> print(result)
    {1: [1, 4.0], 2: [0, nan], 3: [0, nan]}

    Note:
    - Blobs are detected using the Difference of Gaussians (DoG) method.
    - Detected blobs are filtered based on the specified threshold and expected diameter.
    - The returned dictionary contains information about the number of detected blobs and their mean intensity
      for each labeled region.
    """
    blob_labels = {}
    dilated_image = ndimage.grey_dilation(label, footprint=disk(10))
    for mask_index in np.unique(label):
        if mask_index == 0:
            continue
        removed_background = image.copy()
        one_mask = label.copy()
        one_mask[np.where(label != mask_index)] = 0
        dilated_copy = dilated_image.copy()
        dilated_copy[np.where(dilated_image != mask_index)] = 0
        removed_background[np.where(dilated_copy == 0)] = 0
        min_sigma = (1 / (1 + math.sqrt(2))) * diameter
        max_sigma = math.sqrt(2) * min_sigma
        blobs = skimage.feature.blob_dog(removed_background, threshold=threshold, min_sigma=min_sigma,
                                         max_sigma=max_sigma)

        mask = np.array([one_mask[int(y), int(x)] != 0 for y, x, r in blobs])
        if not np.any(mask):
            continue
        blobs_filtered = blobs[mask]
        binary_blobs = np.zeros_like(label)
        for blob in blobs_filtered:
            y, x, r = blob
            rr, cc = dsk((y, x), r, shape=binary_blobs.shape)
            binary_blobs[rr, cc] = 1
        spot_intensity = regionprops_table(binary_blobs, removed_background, ['intensity_mean'])
        blob_labels[mask_index] = [blobs_filtered.shape[0], spot_intensity['intensity_mean'][0]]
    return blob_labels

