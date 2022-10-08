import torusgrid as tg


def evolve_and_elongate_interface(
        ifc: tg.RealField2D, 
        delta_sol: tg.RealField2D, 
        delta_liq: tg.RealField2D,

        minimizer_supplier: Callable[[PFC], PFCMinimizer]=None,

        minimizer: str=None, dt: float=None, eps: float=None, mu: float=None, 
        N_steps: int=31, N_epochs: Optional[int]=None, interrupt_handler: Optional[Callable]=None,
        display_format: Optional[str]=None, callbacks: List[Callable]=[],
        verbose=False):

    try:
        assert delta_sol.Ny == delta_liq.Ny == ifc.Ny
        assert delta_sol.Ly == delta_liq.Ly == ifc.Ly
    except AssertionError:
        raise ValueError('delta_sol, delta_liq, and ifc must have the same Y dimensions')


    if verbose:
        print(f'Lx={ifc.Lx}, Ly={ifc.Ly}')
        print(f'evolving interface')
    model = PFC(ifc)

    model.evolve(
            minimizer_supplier=minimizer_supplier,
            minimizer=minimizer, dt=dt, eps=eps, mu=mu, N_steps=N_steps, N_epochs=N_epochs, 
            custom_keyboard_interrupt_handler=interrupt_handler,
            display_format=display_format, callbacks=callbacks
    )

    if verbose:
        print(f'elongating interface')


    left = fu.crop(model.field, 0, 0.5, 0, 1)
    right = fu.crop(model.field, 0.5, 1, 0, 1)

    tmp = fu.concat(delta_liq, left)
    tmp = fu.concat(tmp, delta_sol)    
    tmp = fu.concat(tmp, delta_sol)    
    tmp = fu.concat(tmp, right)   
    ifc_elongated = fu.concat(tmp, delta_liq)

    if verbose:
        print(f'Lx={ifc_elongated.Lx}, Ly={ifc_elongated.Ly}')
    return model, ifc_elongated


 
