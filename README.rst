Celldetective
=============

.. raw:: html

	<embed>
		<p align="center">
		<img src="https://github.com/remyeltorro/celldetective/blob/main/celldetective/icons/logo-large.png" width="33%" />
		</p>
	</embed>


Celldetective is a python package and software to perform single-cell analysis on multimodal time lapse microscopy images.

- **Documentation:** https://celldetective.readthedocs.io
- **Source code:** https://github.com/remyeltorro/celldetective
- **Bug reports:** https://github.com/remyeltorro/celldetective/issues/new/choose
- **Datasets, models and demos:** https://zenodo.org/records/10650279

Overview
--------

.. raw:: html

	<embed>
		<p align="center">
		<img src="https://github.com/remyeltorro/celldetective/blob/main/docs/source/_static/celldetective-blocks.png" width="90%" />
		</p>
	</embed>



Despite notable efforts in the development of user-friendly softwares that integrate state-of-the-art solutions to perform single cell analysis, very few are designed for time-lapse data and even less for multimodal problems where cells populations are mixed and can only be separated through the use of multimodal information. Few software solutions provide, to our knowledge, the extraction of response functions from single cell events such as the dynamic survival of a population directly in the GUI, as coding skills are usually required to do so. We want to study complex data which is often multimodal time lapse microscopy images of interacting cell populations, without loss of generality. With a high need for an easy-to-use, no-coding-skill-required software adapted to images and intended for biologists, we introduce **Celldetective**, an open-source python-based software with the following highlight features:

* **Comprehensive single-cell image analysis** : Celldetective ships segmentation, tracking, and measurement modules, as well as event detection from single-cell signals, for up to two populations of interest.
* **Integration of state-of-the-art solutions** : Celldetective harnesses state-of-the-art segmentation techniques (StarDist [#]_, Cellpose [#]_ , [#]_) and tracking algorithm (bTrack [#]_), as well as the napari viewer [#]_ where applicable. These algorithms are interfaced to be well integrated and accessible for the target audience, in the context of complex biological applications.
* **A framework for event description and annotations** : we propose a broad and intuitive framework to annotate and automate the detection of events from single-cell signals through Deep Learning signal classification and regression. The event formulation is directly exploited to define population survival responses.
* **A neighborhood scheme to study cell-cell interactions** : we introduce a neighborhood scheme to relate the spatio-temporal distribution and measurements of two cell populations, allowing the study of how cell-cell interactions affect single-cell and population responses.
* **Deep Learning customization in GUI** : Celldetective facilitates the specialization of Deep Learning models or the creation of new ones adapted to user data, by facilitating the creation of training sets and the training of such models, without having to write a single line of code.
* **In-software analysis** : Celldetective ships visualization tools to collapse single-cell signals with respect to an event, build survival curves, compare measurement distributions across biological conditions.
* **A library of segmentation and signal models**: we created specific models to investigate a co-culture of MCF-7 cells and primary NK cells, that are available directly is the software with a large collection of generalist models developed by the StarDist and Cellpose teams, which are a perfect starting point to segment single cells in a new biological system. 
* **Accessible and open source** : Celldetective does not require any coding skills. The software, its models and datasets are made fully open source to encourage transparency and reproducibility.



System requirements
===================

Hardware requirements
---------------------

The software was tested on several machines, including:

- An Intel(R) Core(TM) i9-10850K CPU @ 3.60GHz, with a single NVIDIA GeForce RTX 3070 (8 Gb of memory) and 16 Gb of memory
- An Intel(R) Core(TM) i7-9750H CPU @ 2.60 GHz, with 16 Gb of memory

In GPU mode, succesive segmentation and DL signal analysis could be performed without saturating the GPU memory thanks to the subprocess formulation for the different modules. The GPU can be disabled in the startup window. The software does not require a GPU (but model inference will be longer). A typical analysis of a single movie with a GPU takes between 5 to 15 minutes. Depending on the number of cells and frames on the images, this computation time can increase to the order of half an hour on a CPU. 

The memory must be sufficient to load a movie stack at once in order to visualize it in napari. Otherwise, processing is performed frame by frame, therefore the memory required is extremely low. 

Software requirements
---------------------

The software was developed simulateously on Ubuntu 20.04 and Windows 11. It was tested on MacOS, but Tensorflow installation can rquire extra steps. 

- Linux: Ubuntu 20.04.6 LTS (Focal Fossa) (not tested on ulterior versions)
- Windows: Windows 11 Home 23H2

To use the software, you must install python, *e.g.* through `Anaconda <https://www.anaconda.com/download>`_. We developed and tested the software in Python 3.9.18. 


Installation
============


Stable release
--------------

The first release will be available once we open the GitHub repository to the public.


Development version
-------------------

From GitHub
~~~~~~~~~~~

Cloning or installing from the GitHub repository will be available once we open the repository to the public.


If you want to run the latest development version, you can clone the repository to your local machine and install Celldetective in “development” mode. This means that any changes to the cloned repository will be immediately available in the python environment:

.. code-block:: bash

	# creates "celldetective" folder
	git clone git://github.com/remyeltorro/celldetective.git
	cd celldetective

	# install the celldetective package in editable/development mode
	pip install -e .

To run the latest development version without cloning the repository, you can also use this line:

.. code-block:: bash

	pip install git+https//github.com/remyeltorro/celldetective.git

From a zip file
~~~~~~~~~~~~~~~

You can also download the repository as a compressed file. Unzip the file and open a terminal at the root of the folder (same level as the file requirements.txt). We recommend that you create a python environment as Celldetective relies on many packages that may interfere with package requirements for other projects. Run the following lines to create an environment named "celldetective":

.. code-block:: bash

	conda create -n celldetective python=3.9.18 pyqt
	conda activate celldetective
	pip install -r requirements.txt
	pip install .

The installation of the dependencies will take a few minutes (up to half an hour if the network is bad). The Celldetective package itself is light and installs in a few seconds.

Before launching the software, move to a different directory as running the package locally can create some bugs when locating the models.


Documentation
=============

Read the tutorial here:

https://celldetective.readthedocs.io/

How to cite?
============

If you use this code in your research, please cite the `Celldetective <https://www.biorxiv.org/content/10.1101/2024.03.15.585250v1>`_  paper (currently preprint):

.. code-block:: raw

	@article {Torro2024.03.15.585250,
		author = {R{\'e}my Torro and Beatriz D{\`\i}az-Bello and Dalia El Arawi and Lorna Ammer and Patrick Chames and Kheya Sengupta and Laurent Limozin},
		title = {Celldetective: an AI-enhanced image analysis tool for unraveling dynamic cell interactions},
		elocation-id = {2024.03.15.585250},
		year = {2024},
		doi = {10.1101/2024.03.15.585250},
		publisher = {Cold Spring Harbor Laboratory},
		abstract = {A current key challenge in bioimaging is the analysis of multimodal and multidimensional data reporting dynamic interactions between diverse cell populations. We developed Celldetective, a software that integrates AI-based segmentation and tracking algorithms and automated signal analysis into a user-friendly graphical interface. It offers complete interactive visualization, annotation, and training capabilities. We demonstrate it by analyzing original experimental data of spreading immune effector cells as well as antibody-dependent cell cytotoxicity events using multimodal fluorescence microscopy.Competing Interest StatementThe authors have declared no competing interest.},
		URL = {https://www.biorxiv.org/content/early/2024/03/17/2024.03.15.585250},
		eprint = {https://www.biorxiv.org/content/early/2024/03/17/2024.03.15.585250.full.pdf},
		journal = {bioRxiv}
	}




Bibliography
============

.. [#] Schmidt, U., Weigert, M., Broaddus, C. & Myers, G. Cell Detection with Star-Convex Polygons. in Medical Image Computing and Computer Assisted Intervention – MICCAI 2018 (eds. Frangi, A. F., Schnabel, J. A., Davatzikos, C., Alberola-López, C. & Fichtinger, G.) 265–273 (Springer International Publishing, Cham, 2018). doi:10.1007/978-3-030-00934-2_30.

.. [#] Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. Cellpose: a generalist algorithm for cellular segmentation. Nat Methods 18, 100–106 (2021).

.. [#] Pachitariu, M. & Stringer, C. Cellpose 2.0: how to train your own model. Nat Methods 19, 1634–1641 (2022).

.. [#] Ulicna, K., Vallardi, G., Charras, G. & Lowe, A. R. Automated Deep Lineage Tree Analysis Using a Bayesian Single Cell Tracking Approach. Frontiers in Computer Science 3, (2021).

.. [#] Ahlers, J. et al. napari: a multi-dimensional image viewer for Python. Zenodo https://doi.org/10.5281/zenodo.8115575 (2023).
