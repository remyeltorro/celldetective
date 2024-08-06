Troubleshooting
===============

.. _troubleshooting:

Napari issues
-------------

Black screen
~~~~~~~~~~~~

*napari* opens to a black screen with the following error messages:

.. code-block:: bash

    WARNING: QOpenGLWidget: Failed to create context
    parallel.py (371): The TBB threading layer requires TBB version 2021 update 6 or later i.e., TBB_INTERFACE_VERSION >= 12060. Found TBB_INTERFACE_VERSION = 12050. The TBB threading layer is disabled.
    WARNING: QOpenGLWidget: Failed to create context
    WARNING: composeAndFlush: QOpenGLContext creation failed
    WARNING: composeAndFlush: makeCurrent() failed

A potential fix is to create a symbolic link to libraries not found by *napari* in your python environment (here a conda environment named celldetective):

.. code-block:: bash

    cd ~/anaconda3/envs/celldetective/lib
    mkdir backup 
    mv libstd* backup
    cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6  ./ 
    ln -s libstdc++.so.6 libstdc++.so
    ln -s libstdc++.so.6 libstdc++.so.6.0.19

Deep learning libraries
-----------------------

Pytorch
~~~~~~~

On older hardware, Pytorch may yield the following error:

.. code-block:: bash

    Pytorch: [W NNPACK.cpp:64] Could not initialize NNPACK! Reason: Unsupported hardware.

A potential fix is to install Pytorch through `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_:

.. code-block:: bash

    mamba remove pytorch
    mamba install pytorch

Tensorflow
~~~~~~~~~~

On older hardware, Tensorflow may yield the following error:

.. code-block:: bash
    Tensorflow: Illegal instruction (core dumped)
    The TensorFlow library was compiled to use AVX instructions, but these aren't available on your machine.

A potential fix is to install Tensorflow through `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_:

.. code-block:: bash

    conda remove tensorflow
    mamba remove tensorflow
    mamba install tensorflow

StarDist
~~~~~~~~

When training a StarDist model on an older CPU the following error can be triggered:

.. code-block:: bash
    pyopencl._cl.LogicError: clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR

Try to install the missing pocl library as:

.. code-block:: bash

    pip install pocl-binary-distribution
