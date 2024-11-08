Preprocessing
=============

.. _preprocessing:

Sometimes, you should correct the images before segmentation to make it easier or obtain more controlled measurements. The preprocessing block of Celldetective allows you to batch-correct your stacks. 

The principle is as follows: 
#. you select a target channel and set the correction parameters either manually or visually
#. you add this correction protocol to the list of corrections
#. you repeat the steps for other channels of interest
#. you press submit to apply these protocols one after the other to your images. If you select multiple positions, protocol 1 will be applied to all the stacks starting with the ``movie_prefix``. The new stacks will be appended the prefix ``Corrected_``. As it shifts to protocol 2, the program automatically selected the ``Corrected_`` stacks and does not add a new prefix. At the very end, it is up to the user to change the ``movie_prefix`` to ``Corrected_`` to use the corrected stacks.


Background correction
---------------------

.. note:: 
    The background correction feature started being available in version 1.1.0

Celldetective includes two techniques for background correction. The first is fit-based, where the idea is to fit a background for each image, where cells are excluded, with a 2D model (plane, paraboloid), and apply it. The second leverages the multiple-position organization of the experiment folder to build a median background, without cells.


Model fit
~~~~~~~~~

The principle is to fit a background model to the image *in-situ*, by masking the non-homogeneous parts first, with a combination of a Gaussian blur and a standard-deviation filter. This preprocessing is done automatically by Celldetective. You have to set a threshold on this transformed image using the viewer. 

You can currently choose between a paraboloid and a plane model to fit the background. 

The background can be subtracted (with or without clipping) from the images or divided to the image. 

Model free
~~~~~~~~~~

For time series, you can select the frame range over which you have the highest chance of observing the background (fewer cells). Otherwise, tick the tiles option (all frames are estimators for the background). 

As above, you set a threshold on the standard deviation transformed image to mask the cells. 

A median projection over the positions is performed to estimate a model-free background per well. The quality check (QC) slider allows you to monitor each generated background. Do not worry if you have some white pixels on the image. These are NaN values. They arise because the background cannot be confidently observed at this image location. If you have too many NaN pixels, go back one step and decrease the threshold. 

This background can be applied to each original image (subtraction or division, clip or not). In addition, an optimization can be performed to minimize the intensity difference between the background part of each image and the generated background. 


Channel offset correction
-------------------------

In some optical microscopy setups, nonnegligible offsets can arise when switching modalities. The offsets are a problem for intensity measurements as we project a cell mask across all channels to extract its intensity. 

With this preprocessing module, you can estimate the offset value between two channels and correct it. You can click on the viewer button to see in grayscale a reference channel and in a blue overlay the channel to correct. Use your keyboard arrows to move the overlay until the superposition is good. Then apply to write the protocol and add it to the list.
