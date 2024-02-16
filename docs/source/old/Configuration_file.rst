Configuration file
==================

.. _configuration_file:


The parameters in the configuration file are grouped into sections. 

MovieSettings
-------------

.. code-block:: ini

   [MovieSettings]
   pxtoum = 0.1
   frametomin = 1.0
   len_movie = 60
   shape_x = 2048
   shape_y = 2048
   transmission = 0
   blue_channel = 3
   red_channel = 1
   green_channel = -1
   movie_prefix = Aligned

The ``MovieSettings`` section concentrates all of the parameters related to the movie format.

* ``pxtoum``: pixel to µm spatial calibration (isotropic)
* ``frametomin``: frame to minutes time calibration
* ``len_movie``: total number of frames in the movie
* ``shape_x``: shape in pixels of the movie along the first spatial axis
* ``shape_y``: shape in pixels of the movie along the second spatial axis
* ``transmission``: index of the transmission channel in the channel axis
* ``blue``: index of the alive target cell nuclei channel (Hoescht) in the channel axis
* ``red``: index of the dead target cell nuclei channel (PI) in the channel axis
* optional: ``green``: index of the effector cell channel (CFSE)
* ``movie_prefix``: prefix for the movie stack names in the folders

.. code-block:: ini

   [Thresholds]
   intensity_measurement_radius = 26
   intensity_measurement_radius_nk = 10
   minimum_tracklength = 0
   model_signal_length = 128
   hide_frames_for_tracking = 8,7,15,16

The ``Thresholds`` section contains parameters related to the analysis itself:

* ``intensity_measurement_radius``: the fluorescence intensity of the target cell nuclei in measured inside a circle of radius ``intensity_measurement_radius`` about the centroid of the nucleus, at each time step
* ``intensity_measurement_radius_nk``: the fluorescence intensity of the effector cell nuclei in measured inside a circle of radius ``intensity_measurement_radius_nk`` about the centroid of the nucleus, at each time step
* ``minimum_tracklength``: minimum number of points in a trajectory to keep the trajectory
* ``model_signal_length``: length of the fluorescence signals to send to the lysis detection models (to remove?)
* ``hide_frames_for_tracking``: frames to skip during the tracking of the target cells because the segmentation quality is poor (blur)


The ``Labels`` section is used to make navigation across experiments smoother:

.. code-block:: ini

   [Labels]
   concentrations = 0, 10, 100, 100, 10,0
   cell_types = WT, WT, WT, HER2+, HER2+, HER2+ 

* ``concentrations``: concentration of the antibody per well
* ``cell_types``: cell type per well



