Neighborhood
============

.. _neighborhood:

Prerequisites
-------------

You must perform the segmentation and measurements of the cell populations for which you want to compute the neighborhood (at least one).

Principle
---------

Celldetective allows a complete and independent characterization of two cell populations that evolve simultaneously on the microscopy images. In order to study the effect of one population on the other, we developed a simple neighborhood scheme. In the most general case, it is always possible to define an isotropic neighborhood around the center of mass of each object in the system. The only control parameter is the radius of the circle :math:`R_{\textrm{neigh}}` , which determines the largest distance over which two cells can be matched as neighbors. 


This kind of neighborhood can be defined between one population and another but also within a population, *e.g.* to describe local cell density. In the next chapter, we will show that the isotropic neighborhood is the only practical solution on systems of interacting cells when the cell shape of the reference population is not easily accessible.

Most of the difficulties are displaced to the quantification step, as the isotropic neighborhood rarely reflects the true geometry of the system and is therefore subject to over or underestimation of the real number of neighboring cells. To face these difficulties, we introduced three different counting methods:

#. inclusive: all cells inside the circle are counted as neighbors to the reference cell
#. exclusive: attribute each neighbor to the closest reference cell only
#. intermediate: all cells are counted as neighbors but with a weight (the attention weight) that is :math:`1/M` with :math:`M` the number of neighborhoods the neighbor cell is engaged in

.. figure:: _static/neighborhoods.png
    :align: center
    :alt: neighborhoods
    
    **Proposed neighborhood counting methods.** Notice the weights attributed in (c).



Configure a neighborhood measurement
------------------------------------

If you want to compute a neighborhood you can go the ``NEIGHBORHOOD``  section of the control panel and open the settings window associated to the ``Distance cut`` option. 

.. figure:: _static/neigh-ui.png
    :align: center
    :alt: neigh-ui
    
    **GUI for neighborhood configuration.** After setting the reference and neighbor populations, which can be identical, the user defines as many radii as there are neighborhood distances of interest.

You must define the reference and neighbor populations. The neighbor population is associated a status, in order to decompose the neighborhood into sub-populations (*e.g.* dead and alive neighbor cells). A NOT gate on the side can be used to switch the 0 and 1 in the status column. An option can be ticked to compute the cumulated presence of a neighbor in a neighborhood. The second option is to symmetrize the neighborhood written in the table of the reference cells to that of the neighbor cells. The event time option defines a pre-event window for the reference cells over which to compute the average number of neighboring cells, using the three methods described above. The idea is to have an estimator of the average neighbor presence before an event occurred to the reference cell.

You can define as many neighborhood distances as needed, simply by adding radii one by one.

When you submit, the tables for both populations are loaded, and for each time point, the distance for all cell pairs across the populations is computed. The matrix is thresholded to the neighborhood radius. Then it is scanned column-wise to determine the attention weights of the neighbors. Finally it is run row-wise, to attribute to each reference cell its neighbors, as a dictionary containing basic information about each neighbor (identity, attention weight, status) at this neighborhood size. 

This process is repeated for each time step and for each neighborhood radius, yield as many “neighborhood” columns as radii were set. Then for each neighborhood size, reference cells that are too close to the image edge and for which the neighborhood is incomplete are masked from the neighborhood analysis. Finally, the neighborhood counts are performed, using each of the three techniques described before, and decomposing by the status of the neighbor cells, yielding 9 counting metrics. In addition, the event time of the reference cell information is exploited to measure the mean neighborhood before the event. Therefore, 12 counting metrics are obtained for each neighborhood. The complete tables that include a neighborhood column with dictionaries in each cell are saved as ``pickle`` files. The counting metrics are equivalent to the single cell signals measured before and are written in the csv tables, in such a way that the can be exploited by the signal annotator.