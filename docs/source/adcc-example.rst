ADCC experiment in GUI
======================

.. _adcc-example:

Download a demo experiment
--------------------------

We provide two demo experiments in `Zenodo`_ :

.. _Zenodo : https://zenodo.org/records/10650279

#. an ADCC experiment: we imaged a co-culture of MCF-7 and human primary NK cells, in the presence of antibodies
#. a RICM experiment: we imaged human primary NK cells spreading on a surface covered with antibodies 

Unzip the file, extract the content. The resulting folder, containing a well ``W1/`` and a ``config.ini`` file is the experiment folder. Open Celldetective, load this experiment folder. Here, we show the analysis steps for the ADCC demo. 

Segment
-------

At the top part of the control panel, select the ``W1`` well and ``100`` position option to process the first position only.

First, expand the ``PROCESS TARGETS`` block. Tick the ``SEGMENT`` option, select the ``MCF7_h_versatile`` model in the segmentation model zoo. Click on ``Submit`` to segment. The terminal will show the segmentation progress and the GUI will freeze. This step will take a few minutes. Upon completion the eye icon becomes active. Click on it to see the segmentation result in napari.


Track
-----

Click on the settings button of the tracking module. Make sure that features are disabled. Set a minimum tracklength to 10, tick the ``Remove tracks that do not start at the beginning``, ``Interpolate missed detections within tracks`` and ``Sustain last position until the end of the movie`` options. Untick all of the other post-processing options. Click on ``Save``. Tick the ``TRACK`` option. Make sure the ``SEGMENT`` option is disabled and ``Submit`` to track the target cell nuclei. Upon completion, you can view the raw bTrack output in napari, by clicking on the eye button of the tracking module.

Measure
-------

Tick the ``MEASURE`` and ``SIGNAL ANALYSIS`` options. Select ``lysis_PI_area`` in the signal analysis model zoo, to automatically detect lysis events. Submit. At the end of the process, you can configure the signal annotator visualizer by clicking on the settings button of the signal analysis module. Configure the RGB representation and save. Click on the eye to visualize the tracked cells and their signals.

.. figure:: _static/signal-annotator.gif
    :width: 800px
    :align: center
    :alt: signal_annotator

    Visualize single-cell signals with the signal annotator.

Analysis
--------

Once you are satisfied with the classification of lysis events, go to the ``Analyze`` tab of the control panel. Click on the ``plot survival`` button. Select the ``targets`` population, take ``t_lysis`` as the time of interest and ``0`` as the time of reference. Submit. You will see the survival function associated to the position you analyzed. 

To view the collapsed signal response, with respect to the lysis event determined by the signal analysis model, click on the ``plot signals`` button of the ``Analyze`` tab. Select the ``targets`` population, set ``class_lysis`` as the class of interest and ``t_lysis`` as the time of interest. Submit. The visualizer will open. You can see the mean response for all cells or just the ones classified as dead. You can plot the actual single cell traces. 


