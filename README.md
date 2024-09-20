# Celldetective

<embed>
    <p align="center">
    <img src="https://github.com/remyeltorro/celldetective/blob/main/celldetective/icons/logo-large.png" width="33%" />
    </p>
</embed>

![ico1](https://img.shields.io/readthedocs/celldetective?link=https%3A%2F%2Fcelldetective.readthedocs.io%2Fen%2Flatest%2Findex.html)
![ico17](https://github.com/remyeltorro/celldetective/actions/workflows/test.yml/badge.svg)
![ico4](https://img.shields.io/pypi/v/celldetective)
![ico6](https://img.shields.io/github/downloads/remyeltorro/celldetective/total)
![ico5](https://img.shields.io/pypi/dm/celldetective)
![GitHub repo size](https://img.shields.io/github/repo-size/remyeltorro/celldetective)
![GitHub License](https://img.shields.io/github/license/remyeltorro/celldetective?link=https%3A%2F%2Fgithub.com%2Fremyeltorro%2Fcelldetective%2Fblob%2Fmain%2FLICENSE)
![ico2](https://img.shields.io/github/forks/remyeltorro/celldetective?link=https%3A%2F%2Fgithub.com%2Fremyeltorro%2Fcelldetective%2Fforks)
![ico3](https://img.shields.io/github/stars/remyeltorro/celldetective?link=https%3A%2F%2Fgithub.com%2Fremyeltorro%2Fcelldetective%2Fstargazers)

Celldetective is a python package and graphical user interface to perform single-cell
analysis on multimodal time lapse microscopy images.

-  [Check the full documentation](https://celldetective.readthedocs.io)
-  [Report a bug or request a new feature](https://github.com/remyeltorro/celldetective/issues/new/choose)
-  [Explore the datasets, models and demos](https://zenodo.org/records/10650279)

## Overview

![Pipeline](https://github.com/remyeltorro/celldetective/blob/main/docs/source/_static/celldetective-blocks.png)


Celldetective was designed to analyze time-lapse microscopy images in difficult situations: mixed cell populations that are only separable through multimodal information. This software provides a toolkit for the analysis of cell population interactions. 


**Key features**: 
- Achieve single-cell description (segment / track / measure) for up to two populations of interest
- Signal annotation and traditional or Deep learning automation
- Mask annotation in napari[^5] and retraining of Deep learning models
- Neighborhood linking within and across populations and interaction annotations
- Everything is done graphically, no coding is required!
  
Check out the [highlights](https://celldetective.readthedocs.io/en/latest/overview.html#description) in the documentation! 

Instead of reinventing the wheel and out of respect for the amazing work done by these teams, we chose to build around StarDist[^1] & Cellpose[^2][^3] for the Deep-learning segmentation and the Bayesian tracker bTrack[^4] for tracking. If you use these models or methods in your Celldetective workflow, don't forget to cite the respective papers!

**Target Audience**: The software is targeted to scientists who are interested in quantifying dynamically (or not) cell populations from microscopy images. Experimental scientists who produce such images can also analyze their data, thanks to the graphical interface, that completely removes the need for coding, and the many helper functions that guide the user in the analysis steps. Finally, the modular structure of Celldetective welcomes users with a partial need. 

![Signal analysis](https://github.com/remyeltorro/celldetective/blob/main/docs/source/_static/signal-annotator.gif)


# System requirements

## Hardware requirements

The software was tested on several machines, including:

-   An Intel(R) Core(TM) i9-10850K CPU @ 3.60GHz, with a single NVIDIA
    GeForce RTX 3070 (8 Gb of memory) and 16 Gb of memory
-   An Intel(R) Core(TM) i7-9750H CPU @ 2.60 GHz, with 16 Gb of memory

In GPU mode, succesive segmentation and DL signal analysis could be
performed without saturating the GPU memory thanks to the subprocess
formulation for the different modules. The GPU can be disabled in the
startup window. The software does not require a GPU (but model inference
will be longer). 

A typical analysis of a single movie with a GPU takes
between 3 to 15 minutes. Depending on the number of cells and frames on
the images, this computation time can increase to the order of half an
hour on a CPU.

Processing is performed frame by frame, therefore the memory requirement is extremely low. The main bottleneck is in the visualization of segmentation and tracking output. Whole stacks (typically 1-9 Gb) have to be loaded in memory at once to be viewed in napari. 

## Software requirements

The software was developed simulateously on Ubuntu 20.04 and Windows 11.
It was tested on MacOS, but Tensorflow installation can require extra
steps.

-   Linux: Ubuntu 20.04.6 LTS (Focal Fossa)
-   Windows: Windows 11 Home 23H2

To use the software, you must install python, *e.g.* through
[Anaconda](https://www.anaconda.com/download). Celldetective is routinely tested on both Ubuntu and Windows for Python versions 3.9, 3.10 and 3.11.

# Installation

## Stable release

Celldetective requires a version of Python between 3.9 and 3.11 (included). If your Python version is older or more recent, consider using `conda` to create an environment as described below.

With the proper Python version, Celldetective can be directly installed with `pip`:

``` bash
pip install celldetective
```

We recommend that you create an environment to use Celldetective, to protect your package versions and fix the Python version *e.g.*
with `conda`:

``` bash
conda create -n celldetective python=3.11 pyqt
conda activate celldetective
pip install celldetective
```

Need an update? Simply type the following in the terminal (in your
environment):

``` bash
pip install --upgrade celldetective
```

For more installation options, please check the [documentation](https://celldetective.readthedocs.io/en/latest/get-started.html#installation).


# Quick start

You can launch the GUI by 1) opening a terminal and 2) typing the
following:

``` bash
# conda activate celldetective
python -m celldetective
```

For more information about how to get started, please check the [documentation](https://celldetective.readthedocs.io/en/latest/get-started.html#launching-the-gui).

# How to cite?

If you use this software in your research, please cite the
[Celldetective](https://www.biorxiv.org/content/10.1101/2024.03.15.585250v1)
paper (currently preprint):

``` raw
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
```

Make sure you to cite the papers of any segmentation model (StarDist,
Cellpose) or tracker (bTrack) you used through Celldetective.

# Bibliography

[^1]: Schmidt, U., Weigert, M., Broaddus, C. & Myers, G. Cell Detection
    with Star-Convex Polygons. in Medical Image Computing and Computer
    Assisted Intervention -- MICCAI 2018 (eds. Frangi, A. F., Schnabel,
    J. A., Davatzikos, C., Alberola-López, C. & Fichtinger, G.) 265--273
    (Springer International Publishing, Cham, 2018).
    <doi:10.1007/978-3-030-00934-2_30>.

[^2]: Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. Cellpose: a
    generalist algorithm for cellular segmentation. Nat Methods 18,
    100--106 (2021).

[^3]: Pachitariu, M. & Stringer, C. Cellpose 2.0: how to train your own
    model. Nat Methods 19, 1634--1641 (2022).

[^4]: Ulicna, K., Vallardi, G., Charras, G. & Lowe, A. R. Automated Deep
    Lineage Tree Analysis Using a Bayesian Single Cell Tracking
    Approach. Frontiers in Computer Science 3, (2021).

[^5]: Ahlers, J. et al. napari: a multi-dimensional image viewer for
    Python. Zenodo <https://doi.org/10.5281/zenodo.8115575> (2023).
