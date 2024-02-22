Measure
=======

.. _measure:

Prerequisite
------------

You must segment the cells prior to measurements. The cells can be tracked or not.


I/O
---


The measurement module takes both the segmentation masks and microscopy images as input. If the cells were tracked prior to measurement, the trajectory table is appended with new columns corresponding to the measurements. Otherwise a look-alike table is output by the module, without a ``TRACK_ID`` column (replaced with a ``ID`` column).

Options
-------

Mask-based measurements
~~~~~~~~~~~~~~~~~~~~~~~

The segmentation mask is an obvious starting point to perform single-cell measurements that are tonal, textural and morphological. The mask provides a ROI over which a series of measurements can be performed at each time point. The mask can also be used to define sub sections. 

One practical subsection that can be extracted from the euclidean transform of the mask is to perform a threshold on the distance to the mask boundary, leaving a contour that reflects that of the mask but smaller. With two threshold distances, it is possible to define a slice. This decomposition of the mask can be used to assess the peripherality of a fluorescence signal. 

.. figure:: _static/measurements-ui.png
    :align: center
    :alt: measurement_options
    
    **GUI to pilot single cell measurements, with a highlight on contour intensity measurements.** Mask-based measurements are picked from the list of region properties defined in ``regionprops``. The user can define and visualize contour bands, over which to compute tonal features. Here, the bands are shown for an image of MCF7 cell stained nuclei. The user can enable the computation of Haralick texture features and pilot isotropic measurements.


For morphological and tonal measurements, we rely on the scikit-image library and more specifically ``regionprops`` that provides a fast computation of features from masks.

For texture measurements, we provide several options to measure the texture averaged over cell masks, with refined parameters. You can control carefully the image normalization and play with the distance, scale and # gray levels to make the computation time acceptable while not destroying texture information.

.. figure:: _static/texture-measurements.png
    :align: center
    :alt: texture_options
    
    **GUI to pilot texture measurements.** A section of the measurement configuration window is dedicated to the measurement of the Haralick texture features. As it is computationally expansive, measuring the texture is optional. The user selects the channel of interest within all of the channels available in the loaded experiment. A slider sets the scale parameter to scale down the image before textural computations. The # gray levels field sets the :math:`n_{GL}` parameter. A switch button allows to turn the min/max percentile fields into min/max value fields. A distance field sets the distance over which to compute intensity co-occurrences. On the top right corner, two visualization tools allow to control respectively the histogram of the digitized image and the digitized image itself.

Position-based measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The post-processing operations performed on the trajectories can introduce spatial locations for which there is no associated mask. Indeed, interpolating missing points in trajectories leaves open the question of how and what to measure in these new locations. An even more extreme case is track sustaining, which creates a completely new set of locations where the cell may not even exist. 

In absence of orientational information, the best course of action was to go for an isotropic (circle or ring) measurement of intensities, centered on the positions, irrespective of whether they were interpolated or not. Therefore, for a complete track we could always expect a complete intensity measurement. Obviously, tuning the radius of this circle (or radii for the ring) is an important choice.

.. figure:: _static/iso-measure.png
    :align: center
    :alt: iso_measurements
    
    **GUI to pilot isotropic measurements.** The last section of the measurement configuration window is dedicated to setting up isotropic tonal measurements. The user can define and manage as many circle and rings as desired. Then the operations to be performed on the intensities within the circle or ring are defined right below. By default, all measurements are applied to all available channels in the experiment.

The isotropic measurements are interfaced in almost the same way as the contour measurements, with the exception that the operation to perform over the circle (or ring) ROI has to be defined below (among mean, standard deviation and others). Upon submission a subprocess in launched to take each multichannel frame one by one and perform first the mask measurements and second the isotropic measurements with the kernel defined here. In the example above, if its for three-channel microscopy data then 3 × 2 × 2 = 12 signals will be generated for each tracked single cell.