Track
=====

.. _track:

Prerequisite
------------

You must segment the cells prior to tracking.


I/O
---

This modules takes the instance segmentation label images, the original microscopy images (if you asked for feature measurements) and outputs a trajectory table where each line represents a cell at a given timepoint.

Adapt the tracker to your cells
-------------------------------

After segmentation, tracking the cells is a necessary step to attribute a unique label, an identity, to each cell in a movie. Since cells exhibit complex motion that often goes well beyond the scope of Brownian motion, we decided to interface the state-of-the art tracking method bTrack [#]_ , exploiting both the motion history and the appearance of the cells to make the best tracking hypotheses. 

bTrack requires a configuration file to set all of its motion and tracklet hypotheses. This configuration can be produced interactively using the bTrack-napari plugin in napari (click on the eye in either the segmentation or tracking module). Each cell passed to the tracker can be attached some features, which can be used to help with the tracking.

.. figure:: _static/tracking-options.png
    :width: 400px
    :align: center
    :alt: tracking_options
    
    **GUI to configure the tracking parameters.** A bTrack configuration can be modified in place, or other configurations can be loaded. Features can be passed to the tracker. Post processing modules clean up the raw trajectories for subsequent analysis.


The tracking configuration window ships a text box to edit directly the current json bTrack configuration. In addition, you can import configurations that were produced with the napari-bTrack plugin, accessible directly in Celldetective, which is the recommended way to optimize a tracking configuration. 

You can decide to pass features to the tracking (choosing among morphological, tonal and textural features). If an intensity feature is chosen, the you can select which channels should be passed to bTrack.

Upon submission, a subprocess loads the multichannel images and the masks one frame at a time, to extract all cell locations. If features were enabled, they are measured along the way. Then the tracking configuration is loaded, as well as all the cells from all time points. The potential features are all normalized independently, using a standard scaler. The tracking mode switches from ``motion`` to ``motion + visual`` depending on the presence of features. The tracking is performed and a ``csv`` table containing the tracks is generated in the ``output/tables`` subfolder of each position.


References
----------

.. [#] Ulicna, K., Vallardi, G., Charras, G. & Lowe, A. R. Automated Deep Lineage Tree Analysis Using a Bayesian Single Cell Tracking Approach. Frontiers in Computer Science 3, (2021).
