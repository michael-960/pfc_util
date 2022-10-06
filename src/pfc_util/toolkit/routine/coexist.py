import torusgrid as tg


def find_coexistent_mu_general(
    solid_field: tg.RealField2D,
    mu_min: float, mu_max: float,
    fef: tg.FreeEnergyFunctional,

    minimizer_supplier: Callable[[RealField2D, float], PFCMinimizer], 
    relaxer_supplier: Optional[Callable[[RealField2D, float], PFCMinimizer]]=None,
    max_iters: Optional[int]=None, precision: float=0.,
    relax_callbacks: List[EvolverCallBack]=[],
    minim_callbacks: List[EvolverCallBack]=[],
    **minimizer_kwargs):
    '''
    Find mu such that omega_l == omega_s by running the following:
        (0) Liquefy solid to get liquid profile
        (1) Let mu = (mu_min + mu_max)
        (2) Relax & minimize solid with const. mu
        (3) Minimize liquid with const. mu
        (4) Compare omeag_l & omega_s.
            - If omega_l > omega_s then mu_min = mu
            - If omega_l < omega_s then mu_max = mu
        (5) Repeat (1)~(4) until mu_max - mu_min is smaller than the desired
            precision, or until maximum iteration number is reached
    '''
    if mu_min >= mu_max:
        raise ValueError('mu_min must be smaller than mu_max')
    
    if max_iters is None and precision == 0.:
        raise ValueError('binary search will not stop with max_iters=None and precision=0') 

    sol = solid_field
    liq = tg.liquefy(solid_field)

    
    rec = {'mu': [], 'omega_s': [], 'omega_l': [], 'mu_min_initial': mu_min, 'mu_max_initial': mu_max}
    
    print('===========================================================================')

    for i in range(max_iters):
        if mu_max - mu_min <= precision:
            break
        mu = (mu_min + mu_max) / 2
        print(f'current bounds: {mu_min} ~ {mu_max}')

        if relaxer_supplier:
            # relax solid and resize liquid accordingly
            relaxer = relaxer_supplier(sol, mu)
            relaxer.run_nonstop(31, callbacks=relax_callbacks, **minimizer_kwargs)
            liq.set_size(sol.Lx, sol.Ly)

        # evolve solid under constant mu
        minim = minimizer_supplier(sol, mu)
        minim.run_nonstop(31, callbacks=minim_callbacks, **minimizer_kwargs)

        # calculate solid mean grand potential
        omega_s = fef.mean_grand_potential_density(sol, mu)

        if is_liquid(sol.psi, tol=1e-4):
            print(f'solid field was liquefied during minimization with mu={mu}')
            break

        # evolve liquid under constant mu
        minim = minimizer_supplier(liq, mu)
        minim.run_nonstop(31, callbacks=minim_callbacks, **minimizer_kwargs)
        # calculate liquid mean grand potential
        omega_l = fef.mean_grand_potential_density(liq, mu)

        rec['mu'].append(mu)
        rec['omega_s'].append(omega_s)
        rec['omega_l'].append(omega_l)

        if omega_s < omega_l:
            mu_min = mu

        if omega_s > omega_l:
            mu_max = mu


        print('------------------------------------------------------------------------')

    rec['mu_min_final'] = mu_min
    rec['mu_max_final'] = mu_max
    return rec

