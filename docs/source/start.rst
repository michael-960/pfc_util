Getting Started
=================



Example Code Snippet
-----------------------

The following code snippet will retrieve a preminimized solid unit cell, evolve
it under :math:`\epsilon=0.05` and :math:`\mu=0.08` until :math:`\bar\psi` barely changes, and save the newly minimized 
unit cell to :code:`solid_eps0.05_mu0.08.field`

.. code-block:: python

    import pfc_util as pfc
    import torusgrid as tg

    '''Get a preminimized solid profile at epsilon = 0.1'''
    solid = pfc.toolkit.get_unit_cell(eps='0.1')


    '''Minimize this profile at epsilon = 0.2 and mu = 0.26'''
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


