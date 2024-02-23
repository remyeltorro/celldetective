Train single-cell signal analysis models
========================================

.. _train-signal-models:

.. figure:: _static/train-signal-model.png
    :align: center
    :alt: texture_options
    
    **GUI to pilot texture measurements.** A section of the measurement configuration window is dedicated to the measurement of the Haralick texture features. As it is computationally expansive, measuring the texture is optional. The user selects the channel of interest within all of the channels available in the loaded experiment. A slider sets the scale parameter to scale down the image before textural computations. The # gray levels field sets the :math:`n_{GL}` parameter. A switch button allows to turn the min/max percentile fields into min/max value fields. A distance field sets the distance over which to compute intensity co-occurrences. On the top right corner, two visualization tools allow to control respectively the histogram of the digitized image and the digitized image itself.