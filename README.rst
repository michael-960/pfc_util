PFC simulation in python
==============================
created by michael in 2022/04

pfc_util is a python package for PFC (phase field crystal) simulations.


Documentation
======================

Documentation is available on `<https://pfc-util.readthedocs.io>`_ .



Dependencies
======================
* numpy
* scipy
* matplotlib
* pyfftw
* torusgrid
* rich

Installation
====================
Install via pip:

.. code-block:: bash

    pip install pfc-util



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



Example Code Snippet
=======================

The following code snippet will retrieve a preminimized solid unit cell, evolve
it under :math:`\epsilon=0.05` and :math:`\mu=0.08` until :math:`\bar\psi` barely changes, and save the newly minimized 
unit cell to :code:`solid_eps0.05_mu0.08.field`

.. code-block:: python

    import pfc_util as pfc
    import torusgrid as tg

    '''Get a preminimized solid profile at epsilon = 0.1'''
    solid = pfc.toolkit.get_unit_cell(eps='0.1')


    '''Minimize this profile at epsilon = 0.05 and mu = 0.08'''
    pfc.ConstantMuMinimizer(solid, dt=0.001, eps=0.05, mu=0.08).run(
            n_steps=31, # n_steps means the number of evolution steps between hook calls
                        # hooks are invoked to update display, monitor values, etc
            hooks= (
                tg.dynamics.Display() # add display capability
                + tg.dynamics.Panel() # add a panel to live display
                + tg.dynamics.MonitorValues(
                    {'psibar': lambda e: e.field.psi.mean()},
                    period=8
                ) # tell the minimizer that psi.mean() should be monitored and stored as "psibar"
                  # here e refers to the minimizer instance
                  # period = 8 means that the values are logged every 8 hook calls
                  # so in this case we would be calculating the values every 31*8 = 248 time steps
                + tg.dynamics.Text('psibar={psibar:.8f}')
                + tg.dynamics.DetectSlow(
                    'psibar', rtol=1e-9, atol=0, period=8, patience=200
                ) # make the minimizer stop if psibar varies sufficiently slowly
            )
        )

    tg.save(solid, './solid_eps0.05_mu0.08.field') # save the minimized field


