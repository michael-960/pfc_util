Minimizers
===========

In PFC simulations, one often wishes to find the field configuration
:math:`\psi(\mathbf{r})` that minimizes some free energy fucntional :math:`F[\psi]`.
This is done by *minimizers*, which are subclasses of :code:`torusgrid.dynamics.Evolver`.

To use a minimizer, instantiate a minimizer class (for example
:py:class:`pfc_util.ConstantMuMinimizer`) by passing a
:code:`torusgrid.RealField` object and relevant parameters:


.. code-block:: python
    
    minim = pfc.ConstantMuMinimizer(solid, dt=0.01, eps=0.1, mu=0.19)


Now, to actually run the minimization:

.. code-block:: python

    minim.run(31) 


Note that during the minimization, the data of the field passed into the
minizer's constructor will be modified. That is, the minimization is done
in-place.


By default, this would run until manually interrupted (by pressing
:code:`Ctrl+C` for example). This can be changed by passing a
:code:`torusgrid.dynamics.EvolverHooks` object:


.. code-block:: python

    hooks = (
        tg.dynamics.MonitorValues({'psibar': lambda e: e.field.psi.mean()}) 
        + tg.dynamics.DetectSlow('psibar', rtol=1e-7, atol=0, patience=200) 
    )

    minim.run(31, hooks=hooks)


This will execute a callback every :code:`n_steps` (31 in this case) time
steps to monitor the mean density :math:`\bar\psi`, and stop the process
if it changes too slow.
