Visual inspection
=================

.. _visual_inspection:

Description
-----------

.. figure:: _static/lysis_controller.gif
    :align: center
    :alt: control_lysis_detection
    
    Visual inspection of the cell detection

Clicking on the button ``Control class & regression`` in the Control Panel calls a subprocess that opens a secondary app. 

On the right side of the main window is an accelerated animation thats shows a compressed RGB representation of the selected ADCC movie stack. The stack is min-max normalized using the percentile values written in the :doc:`configuration file <../Configuration_file>`. 

Two scatter plots evolve dynamically with the RGB movie. The circles show the class attributed to each cell (blue is "there is no lysis event", red is "there is a lysis event" and yellow is "lysis already happened or bad tracking"). The crosses show whether or not the cell is dead at the current frame.

Modify a cell lysis status
--------------------------

Each cell is clickable (first double-click selects the cell, the second unselects it). When the user clicks on a cell, the plot of its measured intensity signal is shown on the left side of the window, which a vertical line showing the estimated death time (if any). The user can manually change the class of the selected cell by clicking on the button ``Change class``. Then the user must click on the new color (``red``, ``blue`` or ``yellow``) and submit. If the new color is red, then the death time can be corrected. The signal plot is updated with the new death time, upon submitting.

Exclude cells
-------------

In general, a false positive detection or a cell that is very badly tracked can be classified as yellow. If a debris is on the image, or if a lot of cells in a cluster are badly tracked, the user might want to easily put aside the detections for the higher level analysis. The button ``Exclude ROI`` followed by a drag in the region of interest (from top left to top right) allows the user to "batch classify" those cells as yellow.


Saving options
--------------

If the user clicks on ``Save modifications``, a `visual_table_checked.csv` table is written automatically in the folder of the FOV of interest. The user can also go to ``File > Save As...`` to choose a different path and name. 

Export a training set
---------------------

One feature of this app is that once a movie has been fully controlled/corrected, the user can decide to export an cell lysis training set that can be used to train a new lysis detection model. To do so, go to ``File > Export training set...`` and write the file as `some_name.npy` in a custom dataset folder. With as few as one `.npy` file, you can already fine tune a lysis model of your choice to this data in the Control Panel. 

Shortcuts
---------

- ``Escp``: unselect the currently selected cell
- ``l``: show the last frame of the movie