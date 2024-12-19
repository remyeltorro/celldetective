Preprocessing
=============

.. _preprocessing:


Overview
--------

Preprocessing is an essential step to prepare your microscopy data for analysis in Celldetective. It includes both off-software and in-software methods to align, correct, and optimize your stacks for segmentation and downstream measurements.


Off-software preprocessing
--------------------------

Registration
~~~~~~~~~~~~

We highly recommend aligning your movies before using Celldetective. A common tool for this is the **Linear Stack Alignment with SIFT Multichannel** plugin available in Fiji [#]_ , which can be activated by enabling the PTBIOP update site (see discussion here_).

.. _here: https://forum.image.sc/t/registration-of-multi-channel-timelapse-with-linear-stack-alignment-with-sift/50209/16

To facilitate this step, we provide `a macro`_ that can be reused for preprocessing tasks in the ``movie/`` subfolder of each position folder.

.. _`a macro`: align_macro.html


In-software preprocessing
-------------------------

Sometimes, preprocessing your images directly within Celldetective can simplify segmentation and produce more controlled measurements. The **Preprocessing** module allows you to batch-correct stacks through the following steps:


#. **Select a target channel** and define correction parameters (manually or visually).

#. **Add the correction protocol** to the list.

#. Repeat the process for other channels of interest.

#. **Submit** to apply corrections sequentially:

    - Protocol 1 is applied to all stacks matching the movie_prefix.

    - The corrected stacks are saved with the prefix ``Corrected_``.
    
    - For subsequent protocols, Celldetective uses the ``Corrected_`` stacks without adding another prefix.


Background correction
~~~~~~~~~~~~~~~~~~~~~

Celldetective includes two techniques for background correction. The first is fit-based, where the idea is to fit a background for each image, where cells are excluded, with a 2D model (plane, paraboloid), and apply it. The second leverages the multiple-position organization of the experiment folder to build a median background, without cells.


#. **Model fit**: 

    - Fits a 2D background model (plane or paraboloid) to exclude cells.

    - Masks non-homogeneous parts using a Gaussian blur and standard-deviation filter.

    - Allows you to set a threshold on the transformed image using the viewer.

    - The background can be subtracted (with or without clipping) or divided by the image.

#. **Model free**: 

    - Uses the multiple-position structure of the experiment folder to create a median background without cells.

    - For time-series data, select a frame range with minimal cell coverage, or use the tiles option to consider all frames as background estimators.

    - A median projection over the positions is performed to estimate the background for each well.
    
    - A QC slider helps monitor the generated background, including identifying NaN pixels (indicating uncertain areas). If too many NaN pixels are present, reduce the threshold and repeat the process.
    
    - This background can be subtracted or divided (with or without clipping) from the images. Optimization can minimize intensity differences between the background region and the generated background.


Channel offset correction
~~~~~~~~~~~~~~~~~~~~~~~~~

In some optical microscopy setups, offsets between modalities can affect intensity measurements by misaligning channel data.

With the **channel offset correction** module, you can:

#. Estimate the offset between two channels.

#. Use the Viewer to visualize a reference channel in grayscale and the overlayed channel to correct in blue.

#. Adjust the overlay position using keyboard arrows until alignment is satisfactory.

#. Apply the correction and add it to the protocol list.


Bibliography
------------

.. [#] Schindelin, J., Arganda-Carreras, I., Frise, E. et al. Fiji: an open-source platform for biological-image analysis. Nat Methods 9, 676–682 (2012). https://doi.org/10.1038/nmeth.2019
