======
Models
======

.. _installation:


.. |br| raw:: html

   <br />

.. list-table:: Models
   :widths: 50 50 50
   :header-rows: 1
   :align: center

   * - 
     - dataset
     - model architecture
   * - ``lysis_detection``
     - **cell signals** |br| |br| Hoescht & PI |br| PI
     - .. figure:: _static/modified_resnet.png
           :align: center
           :width: 150px
           
           |br| 2x modified ResNet
   * - ``segmentation_tc``
     - **target nuclei** |br| |br| Hoescht & PI |br| |br| Hoescht
     - .. figure:: _static/stardist_logo.jpg
           :align: center
           :width: 150px
           
           StarDist
   * - ``segmentation_nk``
     - **effectors** |br| |br| Brightfield |br| CFSE
     - .. figure:: _static/stardist_logo.jpg
           :align: center
           :width: 150px
           
           StarDist

----------------------
Lysis detection models
----------------------

Architecture
============

The **lysis detection model** consists of two sequential models with the same ResNet-like backbone but a different prediction head. The backbone consists in a series of three 1D `residual blocks`_ [#]_, a `1D max-pooling layer`_, another series of three 1D residual blocks and a `1D global average pooling layer`_:

.. _`residual blocks`: https://en.wikipedia.org/wiki/Residual_neural_network

.. _`1D max-pooling layer`: https://keras.io/api/layers/pooling_layers/max_pooling1d/

.. _`1D global average pooling layer`: https://keras.io/api/layers/pooling_layers/global_average_pooling1d/

.. code-block::

   from tensorflow.keras.layers import Input, Dense, MaxPooling1D, GlobalAveragePooling1D
   from adccfactory.utils import residual_block

   inputs = Input(shape=(128, nbr_channels))

   for i in range(3):
       if i==0:
           x = residual_block(inputs,64)
       else:
           x = residual_block(x,128)
   x = MaxPooling1D()(x)

   for i in range(3):
       x = residual_block(x,128)

   x = GlobalAveragePooling1D()(x)
   
   # and finally the head


The first model is a classifier network, whose task is to classify the cell signals in one of three classes:

* there is a death event (0)
* there is no death event (1)
* the death event already occured or the signal is anomalous (2)

The prediction head for the classifier is a dense layer with three neurons and a softmax activation function.

.. code-block::

   # Classifier head
   x = Dense(3, activation="softmax", name="classifier")(x)

The cell signals that have been classified as showing a death event are sent to the second model which performs a regression to determine the time of death. This second model is called the regressor. With the PI dye, the time of death is defined as the inflexion point of the PI intensity signal (the moment the nucleus becomes permeable to the dye). 

.. code-block::
   
   # Regressor head
   x = Dense(1, activation="linear", name="regressor")(x)


Data preparation
================

Input shape
........... 

**H_PI model**: each couple of intensity signals :math:`I(t) = [I_{H}(t),I_{PI}(t)]` from the dataset ``lysis_detection`` is padded with zeros along the time axis to reach the shape :math:`(128,2)`. 

.. tip:: 
   If you have both a Hoescht and a PI channel (or equivalent) then this model is recommended.

**PI model**: same as above except that only the red channel is sent to the model: the target shape is :math:`(128,1)`. 

.. tip:: 
   If you have an anomaly on your Hoescht signal (e.g. signal going up when the cell dies, faint signal...), but PI is fine, then this model is recommended.


**NucSpot model**: this is an exact copy of the PI model but prediction is performed on the blue channel (NucSpot®, Incucyte®Nuclight) instead of the red channel (PI). 

.. tip:: 
   This model is recommended when using NucSpot®, Incucyte®Nuclight, and to some extent if the Hoescht signal goes significantly up when the cell dies.

Normalization
.............

**Hoescht & PI model**: The couple of intensity signals :math:`I(t) = [I_{H}(t),I_{PI}(t)]` are normalized with respect to the initial Hoescht (or equivalent) intensity:

.. math::
   I(t) = \frac{I(t)}{I_{H}(0)}

in such a way that the relative amplitudes of the two colors is conserved. A cell that remains alive has a signal :math:`I_{H}(t) \sim 1`.


**PI model** & **NucSpot model**: The PI/Nuclight intensity signal is normalized with respect to the first intensity and offset to zero. 

.. math::
   I_{PI}(t) = \frac{I_{PI}(t)}{I_{PI}(0)} - I_{PI}(0)

Negative values are clipped to zero. A cell that remains alive has a signal :math:`I_{PI}(t) \sim 0`.

The target death times are rescaled from the interval :math:`[0,128]` to the interval :math:`[0,1]` with a min-max operation. These two models are, by construction, unable to detect target cells that are already dead at the beginning of the movie.

Augmentation
............

A large augmentation is performed on the intensity signals to account for most different exposures, dye concentration ratios and unexpected death times within the :math:`T = 128` frame range. 

Some death times are rarely observed experimentally in our setup, which create a bias in the training sets for the regression task. To correct for this biais, we shift the cell signals associated to a death event in order to reach a target death time pulled randomly from a uniform distribution (:math:`t_{\dagger}^{target} \in [0,128]`). 

We also introduce a time independent white noise and modify randomly the amplitudes of :math:`I_{H}(t)` and :math:`I_{PI}(t)`. 


Hyperparameters
===============

**Classifier**: we use a categorical crossentropy loss function with the Adam optimizer (learning rate :math:`= 10^{-3}`, :math:`\beta_1 = 0.9`, :math:`\beta_2 = 0.999`, :math:`\varepsilon = 10^{-7}`). We use the precision as a metric and balance the classes with the weights: :math:`\{0: 1.85, 1: 0.4, 2: 8.8 \}`. A callback reduces the learning rate by a factor of :math:`0.5` if there is no decrease of the validation loss for :math:`80` epochs. Another callback saves the model with the best validation precision. The batch size :math:`= 128` and the model runs until no improvement of the validation loss is observed.

**Regressor**: we use a mean square error loss function with the Adam optimizer (learning rate :math:`= 10^{-3}`, :math:`\beta_1 = 0.9`, :math:`\beta_2 = 0.999`, :math:`\varepsilon = 10^{-7}`). A callback reduces the learning rate by a factor of :math:`0.5` if there is no decrease of the validation loss for :math:`80` epochs. Another callback saves the model with the lowest validation error. The batch size :math:`= 128` and the model runs until no improvement of the validation loss is observed.

-------------------
Segmentation models
-------------------

To upload a Cellpose model, the user must provide an effective spatial calibration of the training images. Cellpose usually up or downscales the training images to reach a cell diameter of 30 px. Starting with images with a spatial calibration 1 px :math:`= 0.1 \ \mu m`, and a median cell size of 20 px, the effective spatial calibration of the input images becomes :math:`s = 0.1 \ \mu m \times 20 px / 30 px`.


All of the segmentation models proposed here are StarDist [#]_ models trained on variations of our ADCC images dataset (``target_nuclei`` & ``effectors``). The training procedure follows very closely the `example notebook`_ provided by StarDist's team. The spatial calibration of all of our training images is 1 px :math:`= 0.3112 \ \mu m`. The have cell nuclei of roughly the same size as in the training set, each frame to segment is up or downscaled to conserve the same spatial scale.

.. _`example notebook`: https://github.com/stardist/stardist/blob/master/examples/2D/2_training.ipynb

Target segmentation models
==========================

These models segment the target cells on crowded images where the effector cells are usually also visible.

Data preparation
................

**H_PI model**: we isolate the Hoescht & PI channels (or equivalent) from the ``target_nuclei`` training set, to reach the shape :math:`(512,512,2)` for each sample (PI is the first channel, Hoescht the second). 

Each channel is normalized independently with a `percentile rescaling`_:

.. _`percentile rescaling`: https://github.com/CSBDeep/CSBDeep/blob/ad20e6d235efa205f175d63fb7c81b2c5e442922/csbdeep/utils/utils.py#L51

.. code-block::
   
   from csbdeep.utils import normalize

   inputs.shape
   # (nbr_samples, 512, 512, nbr_channels)
   
   lower_percentile = 0.0 # do not clip at faint intensities
   upper_percentile = 99.9 # clip just the brightest intensities

   inputs_normalized = [normalize(x,lower_percentile,upper_percentile,
                                  axis=axis_norm,clip=True) for x in inputs]

The data augmentation is performed on the fly and consists of random flips, random intensity changes and white noise. 

Hyperparameters
...............

We set the number of rays to :math:`32`. The loss function is the mean absolute error, measuring the star-convex polygon distances. The UNET-backbone dropout is set to :math:`0.1` and batch normalization is disabled. The number of convolutions per stage of the UNET is set to :math:`3`. The depth of the UNET is set to :math:`3`. The learning rate :math:`= 10^{-3}`. A callback reduces the learning rate by a factor of :math:`0.1` if there is no decrease of the validation loss for :math:`50` epochs. The model takes patches of size :math:`(256,256)` as its input. The training batch size :math:`= 8` and we train the model until the loss stops decreasing.

Effector segmentation models
============================

These models segment the effector cells on crowded images, which usually include target cells and red blood cells. 

Data preparation
................

**bf model**: we isolate the brightfield channel from the ``effectors/brightfield`` training set, to reach the shape :math:`(512,512,1)` for each sample. 

**cfse model**: we isolate the CFSE channel (or equivalent) from the ``effectors/cyto`` training set, to reach the shape :math:`(512,512,1)` for each sample. 

The channel is normalized with the percentile rescaling decribed above.

The data augmentation is identical to the one of the target cells.

Hyperparameters
...............

Identical to the target cell segmentation.


----------
References
----------

.. [#] `Deep Residual Learning for Image Recognition`, He, Kaiming; Zhang, Xiangyu; Ren, Shaoqing; Sun, Jian, IEEE Conference on Computer Vision and Pattern Recognition, Las Vegas, NV, USA (2016)

.. [#] `Cell Detection with Star-convex Polygons`, Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers, International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.
