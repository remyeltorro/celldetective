Celldetective
=============

.. image:: celldetective/icons/logo-large.png
   :align: center
   :width: 33%

Celldetective is a package and software for single-cell image analysis in python.

- **Documentation:** https://celldetective.readthedocs.io
- **Source code:** https://github.com/remyeltorro/celldetective
- **Bug reports:** https://github.com/remyeltorro/celldetective/issues/new/choose

Overview
--------

Despite notable efforts in the development of user-friendly softwares that integrate state-of-the-art solutions to perform single cell analysis, very few are designed for time-lapse data and even less for multimodal problems where cells populations are mixed and can only be separated through the use of multimodal information. Few software solutions provide, to our knowledge, the extraction of response functions from single cell events such as the dynamic survival of a population directly in the GUI, as coding skills are usually required to do so. We want to study complex data which is often multimodal time lapse microscopy images of interacting cell populations, without loss of generality. With a high need for an easy-to-use, no-coding-skill-required software adapted to images and intended for biologists, we introduce **Celldetective**, an open-source python-based software with the following highlight features:

* **Comprehensive single-cell analysis** : segmentation, tracking, and measurement modules, event detection from single-cell signals, for up to two populations of interest.
* **Integration of state-of-the-art solutions** : Celldetective harnesses state-of-the-art segmentation techniques (StarDist [155], Cellpose [131, 169]) and tracking algorithm (bTrack [176]), as well as the napari viewer [1] where applicable. We interfaced these algorithms to make them well integrated and user-friendly for the target audience, in the context of complex biological applications.
* **A framework for event description and annotations** : a broad and intuitive framework to annotate and automate the detection of events from single-cell signals through Deep Learning signal classification and regression. Use event formulation to define global survival responses.
* **A neighborhood scheme to study cell-cell interactions** : a neighborhood scheme to relate the spatio-temporal distribution and measurements of two cell populations, allowing the study of how cell-cell interactions affect global responses.
* **Deep Learning customization in GUI** : Celldetective facilitates the specialization of Deep Learning models or the creation of new ones adapted to user data, by facilitating the creation of training sets and the training of such models, without having to write a single line of code.
* **In-software analysis** : Celldetective ships visualization tools to collapse single-cell signals with respect to an event, build survival curves, compare measurement distributions across biological conditions.
* **A library of segmentation and signal models**: we created specific models to investigate a co-culture of MCF-7 cells and primary NK cells, that are available directly is the software with a large collection of generalist models developed by the StarDist and Cellpose teams, which are a perfect starting point to segment single cells in a new biological system. 
* **Accessible and open source** : Celldetective does not require any coding skills. The software, its models and datasets are made fully open source to encourage transparency and reproducibility.

.. image:: article/figures/Figure1.png
    :width: 60%
    :align: center
    :alt: pipeline

System requirements
===================

Hardware requirements
---------------------

RAM needed (8+? 16+?)
CPU needed, GPU needed...
GPU functionalities tested on NVIDIA RTX 3070 with 8 Gb of memory. 

Software requirements
---------------------



Linux:
Windows:
MacOS: 

Installation
============

Stable release
--------------

Explain here how to install release...

Development version
-------------------

If you want to run the latest development version, you can clone the repository to your local machine and install celldetective in “development” mode. This means that any changes to the cloned repository will be immediately available in the python environment:

.. code-block:: bash

    # creates "celldetective" folder
    git clone git://github.com/remyeltorro/celldetective.git
    cd celldetective

    # install the celldetective package in editable/development mode
    pip install -e .

To run the latest development version without cloning the repository, you can also use this line:

.. code-block:: bash

    pip install git+https//github.com/remyeltorro/celldetective.git

Documentation
=============

Read the tutorial here:

https://celldetective.readthedocs.io/
