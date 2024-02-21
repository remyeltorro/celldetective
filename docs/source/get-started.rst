Get started
===========

.. _get_started:


Running the GUI
---------------

Once the pip installation is complete, open a terminal and run:

.. code-block:: console

	$ python -m adccfactory

A first window of the GUI will open, asking for the path to the ADCC experiment folder to be loaded.


Configure your first experiment folder
--------------------------------------

ADCCFactory requires a specific folder tree, that mimics the organization of a `glass slide`_ into wells (main folders) and positions within the wells (subfolders). A configuration file, common to the whole experiment, is read to provide the relevant information unique to each experiment. 

.. _`glass slide`: Microscopy

.. figure:: _static/glass-slide.png
    :align: center
    :alt: exp_folder_mimics_glass_slide
    
    The experiment folder mimics the organization of the glass slide into wells and fields of view within wells.

To generate automatically such a folder tree, open ADCCFactory and go to File>New experiment... or press Ctrl+N.

.. figure:: _static/startup_new_exp.gif
    :width: 400px
    :align: center
    :alt: startup_new_experiment
    
    Press Ctrl+N or go to File>New experiment... to configure a new experiment folder
   
A dialog window will ask you where you want to create the experiment folder. Then a second window will ask for complementary information needed to fill the configuration file.     
   
.. image:: _static/configure_experiment.png
    :width: 350px
    :align: center
    :alt: configure_experiment

Once you press "Submit", these parameters create the experiment folder named "ExpLambda" in home/. At the root of the experiment folder is a configuration file that looks as follows:

.. code-block:: ini

   # Configuration for ExpLambda/ following user input
   
    [MovieSettings]
    pxtoum = 0.3112
    frametomin = 2.75
    len_movie = 44
    shape_x = 2048
    shape_y = 2048
    movie_prefix = Aligned

    [Channels]
    brightfield_channel = 0
    live_nuclei_channel = 3
    dead_nuclei_channel = 1
    effector_fluo_channel = 2
    adhesion_channel = nan
    fluo_channel_1 = nan
    fluo_channel_2 = nan

    [Labels]
    cell_types = MCF7-HER2+primary NK,MCF7-HER2+primary NK
    antibodies = None,Ab
    concentrations = 0,100
    pharmaceutical_agents = None,None


Detailed information about the role of each parameter is provided in "Configuration file".

Drag and drop movies
--------------------

.. note::

   Unfortunately, putting the movies in their respective folders is a manual task

The user can now drag and drop the movie associated to each field of view of each well in its respective folder (typical path: "ExpFolder/well/fov/movie/"). The movie should be in TIF format and be organized in time-X-Y-channel or channel-time-X-Y order. 

We highly recommend that you align the movie beforehand using for example, the "Linear Stack Alignment with SIFT Multichannel" tool available in Fiji, when activating the PTBIOP update site [#]_ (see discussion here_). We also put `a macro`_ at your disposal to facilitate this preliminary step.

.. _`a macro`: Align_Macro


.. _here: https://forum.image.sc/t/registration-of-multi-channel-timelapse-with-linear-stack-alignment-with-sift/50209/16

Usually, the alive target nucleus florescence channel works as a great reference for alignment, since the target cells are quasi-static. 


.. figure:: _static/align-stack-sift.gif
    :align: center
    :alt: sift_align
    
    Demonstration of the of the SIFT multichannel tool on FIJI



Load an experiment folder
-------------------------

Once you have filled up an experiment folder with some ADCC movies, you can open ADCCFactory, browse to the folder and press "Submit" to open the Control Panel.


References
----------

.. [#] https://www.epfl.ch/research/facilities/ptbiop/
