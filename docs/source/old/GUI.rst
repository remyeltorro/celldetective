GUI
===

.. _gui:

Control panel
-------------

Once an experiment is loaded, a second window, that we call the "Control Panel" opens. The header contains two drop-lists which allow the user to navigate in the experiment. The top list selects the well(s) and the second list the fields of view(s). Any action triggered below will affect the selected wells/FOVs. 

.. figure:: _static/control_panel.png
    :width: 450px
    :align: center
    :alt: control_panel_structure
    
    The Control Panel.


The core of this window contains a two-tab widget. The first tab contains modules that work directly on the movies (segmentation, intensity measurements...). The second tab performs refined analysis on the output of the first tab (survival curves, correlations...).

.. tip::

   All of the modules from the first tab should be run before continuing to tab "Analyze", so that the neighbour analysis can be performed.

Process targets
---------------

Segmentation
************

The first widget in the Process tab is a series of modules related to the target cells. The first step is to segment the target cells in the ADCC movies. We deliver a custom StarDist [#]_ model that relies on Hoescht & PI to isolate the target cells. This model can be selected in the drop-list right below. Tick the "segment TCs" option and press "Submit". A subprocess will be called to load the stack and the model, segment, and export each segmented frame to a ``labels/`` folder inside the position folder.

.. note::

   Make sure that the channels are assigned to the right index in the configuration file. Otherwise, segmentation will be attempted on the wrong channels.

A quick way to control the segmentation quality is to press on "Control segmentation" in the "Process TCs" box. An animation will show the inverted Hoescht channel overlayed with the segmented cells. An inset shows the total number of cells per frames. Spikes in this signal indicate a drop in the segmentation quality, usually due to blurred frames.

.. figure:: _static/control_tc_segmentation.gif
    :width: 400px
    :align: center
    :alt: control_tc_segmentation
    
    Quick animation to control the segmentation quality. Some false positives are triggered by effector cells also intaking the nucleus dye.

Tracking
********

Once a satisfying segmentation is obtained, the user can check the "track TCs" option and press "Submit". The target cells will be automatically tracked using bTrack [#]_ [#]_. bTrack reads a configuration file stored inside ADCCFactory which can be modified. Our standard `configuration file for the target cells tracking`_ is as follows:


.. _`configuration file for the target cells tracking`: https://btrack.readthedocs.io/en/latest/user_guide/configuration.html

.. code-block:: json-object

   {
     "TrackerConfig":
       {
         "MotionModel":
           {
             "name": "cell_motion",
             "dt": 1.0,
             "measurements": 3,
             "states": 6,
             "accuracy": 7.5,
             "prob_not_assign": 0.001,
             "max_lost": 5,
             "A": {
               "matrix": [1,0,0,1,0,0,
                          0,1,0,0,1,0,
                          0,0,1,0,0,1,
                          0,0,0,1,0,0,
                          0,0,0,0,1,0,
                          0,0,0,0,0,1]
             },
             "H": {
               "matrix": [1,0,0,0,0,0,
                          0,1,0,0,0,0,
                          0,0,1,0,0,0]
             },
             "P": {
               "sigma": 150.0,
               "matrix": [0.1,0,0,0,0,0,
                          0,0.1,0,0,0,0,
                          0,0,0.1,0,0,0,
                          0,0,0,1,0,0,
                          0,0,0,0,1,0,
                          0,0,0,0,0,1]
             },
             "G": {
               "sigma": 15.0,
               "matrix": [0.5,0.5,0.5,1,1,1]

             },
             "R": {
               "sigma": 5.0,
               "matrix": [1,0,0,
                          0,1,0,
                          0,0,1]
             }
           },
         "ObjectModel":
           {},
         "HypothesisModel":
           {
             "name": "cell_hypothesis",
             "hypotheses": ["P_FP", "P_init", "P_term", "P_branch","P_link", "P_merge"],
             "lambda_time": 5.0,
             "lambda_dist": 3.0,
             "lambda_link": 10.0,
             "lambda_branch": 1.0,
             "eta": 1e-10,
             "theta_dist": 20.0,
             "theta_time": 5.0,
             "dist_thresh": 35.0,
             "time_thresh": 20.0,
             "apop_thresh": 10,
             "segmentation_miss_rate": 0.05,
             "apoptosis_rate": 0.0005,
             "relax": true
           }
       }
   }   

The tracking modules exports a "trajectories.csv" table in each position folder. 

.. csv-table:: Trajectory table
   :file: /home/limozin/Documents/GitHub/ADCCFactory/docs/source/_static/small_trajectory_table.csv

Lysis detection
***************

The last module, "measure TCs" uses the tracking table and the movie to convert each individual cell into flurescence intensity signals. 

Those signals are sent first to a classifier model that discriminates cells that exhibit a death event from the ones that do not. The dead cells are redirected to a second model, a regressor, that finds the actual death time. Check the :doc:`Models <../Models>`  page to understand which model to choose.  


.. figure:: _static/combined_model_figure.png
    :width: 500px
    :align: center
    :alt: detect_lysis_models_figure
    
    Two models run sequentially to detect and quantify cell lysis.

The "measure TCs" module outputs a new table, which adds intensity and lysis information to the trajectory table. This table called "visual_table.csv" can be read by the :doc:`"Control class & regression" tool <../Visual_inspection_tool>`.

.. csv-table:: Visual table
   :file: /home/limozin/Documents/GitHub/ADCCFactory/docs/source/_static/small_visual_table.csv

.. note::

   Inspection of the lysis detection quality is extremely recommended. The tool allows the user to quickly modify any detected mistake. Saving the modified table after inspection creates a "visual_table_checked.csv" file that can be read by subsequent modules.

.. figure:: _static/lysis_controller.gif
    :align: center
    :alt: control_lysis_detection
    
    The controller window must be used to inspect the quality of the lysis detection and correct mistakes.

Finishing the target cell analysis unlocks the computation of the survival curves in the "Analyze" tab, but that is as far as one can get without analyzing the effector cells.

Process NKs
-----------

Segmentation
************

Similarly to the target cells, the first option is to segment the effector cells. We provide two models for this segmentation:

* segment the NKs directly from bright-field, ignoring target cells and erythrocytes
* segment the effector cells from a fluorescence channel, when available (i.e. CFSE)


.. figure:: _static/control_nk_segmentation.gif
    :width: 400px
    :align: center
    :alt: control_nk_segmentation
    
    Quick animation to control the effector cell segmentation quality.

Filter out RBCs
***************

.. note::

   It is highly recommended to run this module if you segmented the NKs using a brightfield based model.

Separating the NKs from the RBCs in brightfield can be a difficult task. Fortunately, the NKs have a nucleus and intake the target nucelus dye, unlike the RBCs. This module shows a histogram of the Hoescht intensity at the beginning (blue) and the end of the movie (red). If a lot of RBCs are on the image and have been mistakenly detected by the model then there should be a peak at almost zero Hoescht intensities. Move the slider to the end of the peak at zero and submit to remove those cells from the table. You can directly perform another visual control of the segmentation to ensure that the RBCs are no longer marked by a red cross.


Death classification
********************

The third module is used to set an intensity threshold for the dead effector cells. We show the histogram of the PI intensity in the first frame (no death) and the last frame (maximum death). The user can then use a slider to set an intensity threshold and attribute all effector cells with an intensity higher than the threshold as dead. 

.. figure:: _static/dead_nk_threshold.gif
    :width: 400px
    :align: center
    :alt: control_nk_segmentation
    
    Interactive selection for the PI intensity threshold associated to a dead effector cell.

Once the threshold is well known for a given experiment the user can simply toggle "set PI threshold" and type directly a value to be applied.


Neighbourhood
-------------

Once all of the modules of tab "Process" have been executed for a given position, we can switch to the "Analyze" tab and run the first module, "match neighbours (effector & target)". This module reads the parameters from the ``SearchRadii`` section of the configuration file. 

.. code-block:: ini

   [SearchRadii]
   search_radius_tc = 100
   search_radius_nk = 75

Two methods are successively called:

* ``find_nk_neighbours``: for each time step, register effector cells within the circle of radius "search_radius_nk" around a given target cell nucleus as being neighbours to that target cell

* ``find_tc_neighbours``: for each time step, register target cells within the circle of radius "search_radius_tc" around a given target cell nucleus as being neighbours to that target cell

The tables ``visual_table_checked.csv`` and ``table_nks_w_deaths.csv`` are merged into a single ``tc_w_neighbours.csv`` table.

.. csv-table:: Frozen Delights!
   :file: /home/limozin/Documents/GitHub/ADCCFactory/docs/source/_static/small_tc_w_neighbours_table.csv

Survival
--------

The module ``plot survival`` will compute the Kaplan-Meier estimate from a ``visual_table_checked.csv`` table. 

References
----------

.. [#] `Cell Detection with Star-convex Polygons`, Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers, International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.

.. [#] `Automated deep lineage tree analysis using a Bayesian single cell tracking approach`, Ulicna K, Vallardi G, Charras G and Lowe AR. Front in Comp Sci (2021).

.. [#] `Local cellular neighbourhood controls proliferation in cell competition`, Bove A, Gradeci D, Fujita Y, Banerjee S, Charras G and Lowe AR. Mol. Biol. Cell (2017).