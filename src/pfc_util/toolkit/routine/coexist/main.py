from __future__ import annotations

from typing import Literal, Optional, List, Tuple, TypedDict
import torusgrid as tg
import numpy as np
from torusgrid.dynamics import Evolver, EvolverHooks, FieldEvolver


import rich
from ....core import FreeEnergyFunctionalBase
from ....utils import is_liquid

from ..base import MuMinimizerSupplier
from .base import MuSearchRecord



def find_coexistent_mu(
    solid_field: tg.RealField2D,
    mu_min: tg.FloatLike, mu_max: tg.FloatLike,
    fef: FreeEnergyFunctionalBase, *,
    
    relaxer_supplier: MuMinimizerSupplier,
    relaxer_nsteps: int = 31,
    relaxer_hooks: Optional[EvolverHooks[FieldEvolver[tg.RealField2D]]]=None,

    const_mu_supplier: MuMinimizerSupplier,
    const_mu_nsteps: int = 31,
    const_mu_hooks: Optional[EvolverHooks[FieldEvolver[tg.RealField2D]]]=None,
    
    max_iters: Optional[int]=None,
    precision: tg.FloatLike=0.,

    verbose: bool = True,
    liquid_tol: tg.FloatLike = 1e-4,

    search_method: Literal['binary', 'interpolate'] = 'binary'
):
    """
    Given: 
        - a solid profile 
        - minimum and maximum values of chemical potential
        - a free energy functional,

    find the chemical potential that satisfies:

            Omega[solid] = Omega[liquid]
    <=>  F[solid] - mu * N_s = F[liquid] - mu * N_l

    via binary search or linear interpolation.
    """

    dtype = tg.get_real_dtype(solid_field.precision)

    mu_min = dtype(mu_min)
    mu_max = dtype(mu_max)

    digits = round(-np.log10(precision+1e-22))

    if mu_min >= mu_max:
        raise ValueError('mu_min must be smaller than mu_max')

    if max_iters is None and precision == 0.:
        raise ValueError('binary search will not stop with max_iters=None and precision=0') 

    if search_method not in ['binary', 'interpolate']:
        raise ValueError(f'Invalid search method: {search_method}')

    if max_iters is None: max_iters = int(2**32-1)


    sol = solid_field
    liq = tg.const_like(solid_field)

    mu_rec = MuSearchRecord(initial_range=(mu_min, mu_max), search_method=search_method)
   
    console = rich.get_console()

    for _ in range(max_iters):
        if (np.abs(mu_rec.upper_bound - mu_rec.lower_bound) 
            <= precision * (np.abs(mu_rec.upper_bound)
                            +np.abs(mu_rec.lower_bound)) / 2):
            break

        mu = mu_rec.next()

        console.log(f'Computed mu: {mu}')

        if verbose:
            console.rule()
            mu_min_str = tg.highlight_last_digits(tg.float_fmt(mu_rec.lower_bound, digits), 2, 'red')
            mu_max_str = tg.highlight_last_digits(tg.float_fmt(mu_rec.upper_bound, digits), 2, 'red')
            console.log(f'current mu bounds: {mu_min_str} ~ {mu_max_str}')

        '''relax solid'''
        relaxer = relaxer_supplier(sol, mu)
        relaxer.run(relaxer_nsteps, hooks=relaxer_hooks)

        '''resize liquid'''
        liq.set_size(sol.lx, sol.ly)

        '''calculate solid mean grand potential'''
        omega_s = fef.mean_free_energy_density(sol) - mu*sol.psi.mean()

        if is_liquid(sol.psi, tol=liquid_tol):
            if verbose:
                console.log(f'solid field was liquefied during minimization with mu={mu}')
            break

        '''evolve liquid under constant mu'''
        minim = const_mu_supplier(liq, mu)
        minim.run(const_mu_nsteps, hooks=const_mu_hooks)

        '''calculate liquid mean grand potential'''
        omega_l = fef.mean_free_energy_density(liq) - mu*liq.psi.mean()
        
        mu_rec.append(mu, omega_l, omega_s)

        if verbose:
            console.log(f'omega_l={omega_l}')
            console.log(f'omega_s={omega_s}')
            console.log(f'omega_s-omega_l={omega_s - omega_l}')

        if mu_rec.zero is not None:
            console.log(f'omega_l and omega_s are numerically indistinguishable under the current floating point precision')
            break

    if verbose:
        console.rule()
        console.log('Summary:')
        mu_min_str = tg.highlight_last_digits(tg.float_fmt(mu_rec.lower_bound, digits), 2, 'red')
        mu_max_str = tg.highlight_last_digits(tg.float_fmt(mu_rec.lower_bound, digits), 2, 'red')
        final_mu_str = tg.highlight_last_digits(tg.float_fmt(mu_rec.mu[-1], digits), 2, 'red')

        console.log(f'mu min   = {mu_min_str}')
        console.log(f'mu max   = {mu_max_str}')
        console.log(f'final mu = {final_mu_str}')

    return mu_rec

