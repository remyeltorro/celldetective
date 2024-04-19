Measure
=======

.. _measure:

Prerequisite
------------

You must segment the cells prior to measurements. The cells can be tracked or not.


I/O
---


The measurement module takes both the segmentation masks and microscopy images as input. If the cells were tracked prior to measurement, the trajectory table is appended with new columns corresponding to the measurements. Otherwise, a look-alike table is output by the module, without a ``TRACK_ID`` column (replaced with an ``ID`` column).

Options
-------

Background correction
~~~~~~~~~~~~~~~~~~~~~~~

The background correction module allows the user to perform background correction for a specific channel with a choice between two modes: local and field. In local mode, each cell will be corrected individually according to the surrounding background, the distance of the background accounted for is chosen by the user in the outer distance field. If needed, the user cam visualize the selected band around the cell for quality control. The user then can choose whether the correction should be based on the mean or median intensity value of the background, with options to subtract or divide the original cell intensity by the background value.

.. figure:: _static/local_correction.png
    :align: center
    :alt: local_correction

    **GUI to pilot local background correction, with a highlight on the region accounted for during the correction.** The channel on which the correction is to be performed can be selected from the dropdown menu containing all the channels of the experiment. The user can define and visualize contour bands, representing the background that will be used in the correction. Here, the bands are shown for an image of MCF7 cell stained nuclei. The user can then choose whether to perform background subtraction or division based on mean or median background value.


Field background correction assumes non-uniform background distribution. In this case, the user can set the threshold to exclude the cells of interest from the background calculations with the help of a threshold visualization tool. The user must then specify the type of background deformation: paraboloid or plane, which will then be adjusted. Once again, the user must choose whether to subtract or divide by the obtained values. In the case if subtraction, the user can choose whether they wish to remove the obtained negative values by clipping. For field correction, you can preview the corrected image and compare the intensity profiles before and after correction to ensure quality control.

.. figure:: _static/field_correction.png
    :align: center
    :alt: field_correction

    **GUI to pilot field background correction, with a highlight on the threshold interface and corrected image preview** The channel on which the correction is to be performed can be selected from the dropdown menu containing all the channels of the experiment. The user can define the threshold and visualize it with the help of threshold preview interface. The user must choose the type of background distortion present in the image and choose whether to perform background subtraction and division, with or without clipping.


All the background correction parameters that are set to be performed are visible in the list block just below.

Mask-based measurements
~~~~~~~~~~~~~~~~~~~~~~~

The segmentation mask is an obvious starting point to perform single-cell measurements that are tonal, textural and morphological. The mask provides a ROI over which a series of measurements can be performed at each time point. The mask can also be used to define sub sections. 

One practical subsection that can be extracted from the euclidean transform of the mask is to perform a threshold on the distance to the mask boundary, leaving a contour that reflects that of the mask but smaller. With two threshold distances, it is possible to define a slice. This decomposition of the mask can be used to assess the peripherality of a fluorescence signal. 

.. figure:: _static/measurements-ui.png
    :align: center
    :alt: measurement_options
    
    **GUI to pilot single cell measurements, with a highlight on contour intensity measurements.** Mask-based measurements are picked from the list of region properties defined in ``regionprops``. The user can define and visualize contour bands, over which to compute tonal features. Here, the bands are shown for an image of MCF7 cell stained nuclei. The user can enable the computation of Haralick texture features and pilot isotropic measurements.


For morphological and tonal measurements, we rely on the scikit-image library and more specifically ``regionprops`` that provides a fast computation of features from masks.

Based on the this library, a few of the extra measurements were added, such as measurements of the intensity distribution of two types: peripheral and centre of mass displacement. The peripheral intensity calculates the distribution if intensity of each pixel using the Euclidian distance and moving from the edge od the cell mask towards the centre. The obtained intensity values are then fitted linearly, and the slope and intercept of the obtained fit are returned as the output.
The centre of mass displacement is used to define the difference between the geometric centroid of the cell mask and the intensity-weighed centroid. The output is the Euclidian distance between the two centroids as well the orientation in degrees. It is also possible to detect the centre of mass displacement in the outer band only, in cases where the peripheral intensity presents uneven intensity distribution at the edge.

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
    
    **GUI to pilot isotropic measurements.** This section of the measurement configuration window is dedicated to setting up isotropic tonal measurements. The user can define and manage as many circle and rings as desired. Then the operations to be performed on the intensities within the circle or ring are defined right below. By default, all measurements are applied to all available channels in the experiment.

The isotropic measurements are interfaced in almost the same way as the contour measurements, with the exception that the operation to perform over the circle (or ring) ROI has to be defined below (among mean, standard deviation and others). Upon submission, a subprocess is launched to take each multichannel frame one by one and perform, first, the mask measurements, and second, the isotropic measurements with the kernel defined here. In the example above, if its for three-channel microscopy data then 3 × 2 × 2 = 12 signals will be generated for each tracked single cell.

Spot detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spots, characterized by higher intensity values, are circular objects within the cell mask. Users need to input the approximate spot diameter and set a relative threshold value within the range of 0 to 1. Subsequently, they can review the detected spots in their image and fine-tune the threshold or diameter as necessary. The module computes the number of detected spots for each cell along with their mean intensity.

.. figure:: _static/spot_detection.png
    :align: center
    :alt: spot_detection

    **GUI to pilot spot detection** The last section of the measurement configuration window is dedicated to spot detection. The user can choose the channel on which the spot detection is to be performed from dropdown menu. Then an approximate spot diameter and relative spot intensity threshold should be defined. The user has an option to preview the spot detection with the specified parameters and adjust them if needed in the spot preview interface.


Static measurements annotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Celldetective provides a viewer for static data, enabling the visualisation of single-cell measurements. This tool allows the user to visualise the film frame by frame and switch between different channels of the experiment. User can categorize cells based on their characteristics, such as size, and assign phenotypes accordingly.
The tool offers interactive features, allowing users to click on individual cells represented by circles on the plot. Each cell's phenotype is indicated by its color. Upon selection, the measurements associated with the chosen cell are displayed on the graph. The boxplot represents the values of all cells in the film, while the strip plot represents all the cells in the current frame. The selected cell is highlighted as a red dot.
It is possible to represent up to three measurements at a time, with options to normalise and log-rescale the data if the chosen signals exhibit significant differences in quantity. Additionally, users cam opt to display the outliers for the boxplot, which are hidden by default.
The tool is compatible with both static and dynamic data and is especially useful in the cases where a reliable tracking of the cells could not be performed.

.. figure:: _static/measurements_annotator.gif
    :width: 800px
    :align: center
    :alt: measurements_annotator

    Application on an ADCC system of MCF-7 breast cancer cells co-cultured with human primary NK cells.
