Datasets
========

.. |br| raw:: html

   <br />

.. list-table:: Datasets
   :widths: 50 50 50
   :header-rows: 1

   * - 
     - input type
     - output type
   * - cell signals
     - .. figure:: _static/vignette_signals.png
           :align: center
           :width: 150px
           
           two-channel intensity signals
     - .. figure:: _static/dead_cell_signal.png
           :align: center
           :width: 150px
           
           cell class & death time
   * - target nuclei
     - .. figure:: _static/rgb_adcc.png
           :align: center
           :width: 150px
           
           Multichannel brightfield |br| & fluorescence images
     - .. figure:: _static/label_target_adcc.png
           :align: center
           :width: 150px
           
           instance mask
   * - effectors
     - .. figure:: _static/rgb_adcc.png
           :align: center
           :width: 150px
           
           Multichannel brightfield |br| & fluorescence images
     - .. figure:: _static/label_nk_adcc.png
           :align: center
           :width: 150px
           
           instance mask

Cell signals
************

The process TCs module from ADCCFactory measures the fluorescence intensity of each target cell at each time step. As a result, each target cell is represented by a couple of intensity signals from which we can infer the lysis state. The lysis state is a couple of value:

* a class (0: red, 1: blue or 2: yellow) corresponding to ("there is a lysis event", "there is no lysis event", "lysis already happened or bad tracking")
* a death time, corresponding to the moment the nuclear pores allow PI to enter i.e. the inflexion point of the PI intensity signal

This annotation task is performed interactively using the "Control class & regression" tool, accounting for the fact that sometimes cells are not perfectly tracked and the intensity signal is not always sharp. 

The data is stored in a ``.npy`` file, each corresponding to a movie of an ADCC experiment. The data can be simply loaded using:

.. code:: python
    
    dataset = np.load("OctExp_501.npy",allow_pickle=True)
    print(len(datatset)) # number of cells
    # 284
    print(dataset[0].shape) # information for first cell
    # (5,)

The information per cell consists of:

* the cell class
* the alive nucleus signal
* the dead nucleus signal
* a cell index
* the lysis time

Target nuclei
*************

The ``target nuclei`` set contains annotations for the target cell nuclei from many ADCC movies with varying focus quality and effector to target cell ratios. 

This dataset is split into two folders:

* ``images/``: the multichannel ADCC images
* ``labels/``: the paired instance segmentation images

The names of the labels match the names of the respective ADCC image with an added ``_labelled.tif`` suffix. Each image and mask can be loaded in python using the package ``tifffile``:

.. code:: python
    
    from tifffile import imread

    image = imread("target_nuclei/images/110_snap_0_1_10.tif")
    label = imread("target_nuclei/labels/110_snap_0_1_10_labelled.tif")
    
    print(image.shape)
    # (4,512,512)
    print(label.shape)
    # (512,512)

Effectors
*********

The ``effector`` dataset is split into two sub-datasets corresponding to the effector cells annotated from brightfield (in their full shape) or from a nucleus fluorescence channel (like CFSE).   
