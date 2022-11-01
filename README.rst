PFC simulation in python
==============================
created by michael in 2022/04

pfc_util is a python package for PFC (phase field crystal) simulations.

Dependencies
======================
* numpy
* scipy
* matplotlib
* pyfftw
* torusgrid
* rich


Modules
========
:code:`pfc_util.core` - Core module; definitions of PFC free energy functional, state functions, as well as minimizers

:code:`pfc_util.extra` - Extensions to vanilla PFC. Currently includes :code:`pfc6`, which is a 6th-order generalization to the PFC functional

:code:`pfc_util.toolkit` - Static preminimized solid/liquid profiles & utility functions.



Minimizers
============
The package provides three main means of minimizing/evolving a PFC system:

* **Constant chemical potential** - The grand potential :math:`\Omega \equiv F - \mu \int d\mathbf{r} \psi` is minimized with fixed :math:`\mu`

* **Stress relaxer** - The grand potential **density** :math:`\omega \equiv \Omega / V` is minimized with respect to the density field and system size with fixed :math:`\mu`

* **Conserved dynamics** - The free energy :math:`F` is minimized with fixed :math:`\bar\psi`


