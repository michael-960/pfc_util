from __future__ import annotations

from typing import Literal, Optional, List, TypedDict
import torusgrid as tg
import numpy as np
from torusgrid.dynamics import Evolver, EvolverHooks, FieldEvolver

from .base import MuMinimizerSupplier

import rich
from ...core import FreeEnergyFunctionalBase
from ...utils import is_liquid


class MuSearchRecord(TypedDict):
    mu: List[tg.FloatLike]
    omega_s: List[tg.FloatLike]
    omega_l: List[tg.FloatLike]

    mu_min_initial: tg.FloatLike
    mu_max_initial: tg.FloatLike
    
    mu_min_final: tg.FloatLike
    mu_max_final: tg.FloatLike



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

    if max_iters is None: max_iters = 2**32


    sol = solid_field
    liq = tg.const_like(solid_field)

    rec: MuSearchRecord = {
            'mu': [], 
            'omega_s': [], 
            'omega_l': [], 
            'mu_min_initial': mu_min, 
            'mu_max_initial': mu_max,
            'mu_min_final': -1,
            'mu_max_final': -1,
            }

   
    console = rich.get_console()

    for iter in range(max_iters): # type: ignore

        if np.abs(mu_max - mu_min) <= precision * (np.abs(mu_max)+np.abs(mu_min))/2:
            break

        if search_method == 'binary':
            mu = (mu_min + mu_max) / 2
        
        else: # interpolate
            if iter > 2:
                mu1 = rec['mu'][-1]
                omega_l_1 = rec['omega_l'][-1]
                omega_s_1 = rec['omega_s'][-1]

                delta_omega_1 = omega_s_1 - omega_l_1

                mu2 = rec['mu'][-2]
                omega_l_2 = rec['omega_l'][-2]
                omega_s_2 = rec['omega_s'][-2]
                delta_omega_2 = omega_s_2 - omega_l_2

                mu = (-mu1 * delta_omega_2 + mu2 * delta_omega_1) / (delta_omega_1 - delta_omega_2)

                if verbose:
                    console.log(f'Computed mu: {mu2}, {mu1} -> {mu}')

            else:
                mu = (mu_min + mu_max) / 2

        if verbose:
            console.rule()
            if search_method == 'binary':
                mu_min_str = tg.highlight_last_digits(tg.float_fmt(mu_min, digits), 2, 'red')
                mu_max_str = tg.highlight_last_digits(tg.float_fmt(mu_max, digits), 2, 'red')
                console.log(f'current mu bounds: {mu_min_str} ~ {mu_max_str}')

        '''relax solid and resize liquid accordingly'''
        relaxer = relaxer_supplier(sol, mu)
        relaxer.run(relaxer_nsteps, hooks=relaxer_hooks)
        liq.set_size(sol.lx, sol.ly)
        
        '''evolve solid under constant mu'''

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

        rec['mu'].append(mu)
        rec['omega_s'].append(omega_s)
        rec['omega_l'].append(omega_l)

        if verbose:
            console.log(f'omega_l={omega_l}')
            console.log(f'omega_s={omega_s}')

        if omega_s < omega_l:
            mu_min = mu

        elif omega_s > omega_l:
            mu_max = mu

        else:
            console.log('Solid and liquid grand potentials are numerically indistinguishable under current floating point precision.')
            console.log('Aborted.')
            break

    rec['mu_min_final'] = mu_min
    rec['mu_max_final'] = mu_max

    if verbose:
        console.rule()
        console.log('Results:')
        mu_min_str = tg.highlight_last_digits(tg.float_fmt(mu_min, digits), 2, 'red')
        mu_max_str = tg.highlight_last_digits(tg.float_fmt(mu_max, digits), 2, 'red')
        final_mu_str = tg.highlight_last_digits(tg.float_fmt(mu, digits), 2, 'red')

        console.log(f'mu min   = {mu_min_str}')
        console.log(f'mu max   = {mu_max_str}')
        console.log(f'final mu = {final_mu_str}')

    return rec

