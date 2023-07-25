Population model
================

.. _population_model:

Simple model
------------

We assume that the target cells' population can be written in simple terms as:

.. math::
   
   \frac{d T}{dt} = - k_L N(t) T(t) + p T(t) - d T(t)

where :math:`T(t)` is the number of target cells alive at time :math:`t`, :math:`N(t)` is the number of effector cells around a target cell at time :math:`t`, :math:`k_L` is a antibody-dependent lysis constant, :math:`p` a proliferation term for the target cells and :math:`d` an effector-independent death rate for the targets. 

Since in our analysis we do not include target cell tracks that start after the beginning of the movie, we neglect the :math:`p` term. We also chose to neglect the :math:`d` term that is very weak at early times, in which we measure the ADCC. Thus, the population equation simplifies to:

.. math::
   
   \frac{d T}{dt} = - k_L N(t) T(t)

Solving for :math:`T(t)`, we obtain:

.. math::
   
   \log{\frac{T(t)}{T_0}} = - k_L \int_0^t N(t) dt \\
   S(t) = \frac{T(t)}{T_0} = \exp{\left(- k_L \int_0^t N(t) dt \right) }

We quantify :math:`N(t)` as the instantaneous number of effector cells in the neighbourhood of a target cell. A typical profile for :math:`N(t)` shows a saturation after a short amount of time. Imposing that those effector cells must be alive to be counted as predators, we still end up with an almost saturating profile. Thus the survival :math:`S(t)` should look like an exponential with a delay corresponding to the increasing phase of :math:`N(t)`. 

.. figure:: _static/simple_model_illustration.png
    :align: center
    :alt: simple_population_model
    
    The population model is too simple to fit the experimental data.


Refined model
-------------

Experimental survival curves are unfortunately never quite like this. A recurrent feature is a plateau of the survival after some times, which could only be explained with :math:`N(t) \rightarrow 0`. Another feature is that the emergence of this plateau is usually common to all conditions within an experiment. To model both features, we propose to introduce the notion of active effector cell :math:`N^a(t) = \alpha(t) N(t)` where :math:`\alpha(t)` is common to all antibody concentrations. 

.. math::
   
   \frac{d T}{dt} = - k_L N^a(t) T(t) = - k_L \alpha(t) N(t) T(t)

Isolating the quantities we can measure experimentally:

.. math::
   
   K(c,t) = k_L(c) \alpha(t) = - \frac{1}{N(t)} \frac{1}{T(t)} \frac{d T}{dt}

Where we assume that the lysis term :math:`k_L` is really independent of time. Comparing multiple antibody concentrations, we can estimate :math:`\alpha(t)` up to a constant multiplicative factor. Constraining the max of :math:`\alpha` to be 1, we can isolate :math:`k_L(t) =  K(c,t) / \alpha(t)`. If :math:`k_L(t)` is independent of time then our variable separation assumption is valid. 

.. math::
 
   S(t) = \frac{T(t)}{T_0} = \exp{\left(- k_L \int_0^t \alpha(t) N(t) dt \right) }


Target cell organization dependence
***********************************

The recovered :math:`k_L` is defined for a group of cells. The grouping of the cell can be spatial (per field of view), condition-based (per concentration of antibody and cell type) but we could also group the cells by cellular organization. Our metric for the cellular organization is the target neighbour density :math:`d` defined as the number of target cell neighbours in a circle centered around the nucleus of radius 31 :math:`\mu m`. 

Grouping cells by :math:`d` and performing the normalization procedure described above, we can obtain one :math:`k_L` value per group. Differences in this value across different :math:`d` groups should be independent of the effector cell concentration. Indeed, we observe that, in general, :math:`k_L(d)` is non flat. 
