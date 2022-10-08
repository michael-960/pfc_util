from __future__ import annotations

from typing import Optional, Callable, List, TypeVar, TypedDict
import torusgrid as tg
import numpy as np
from torusgrid.dynamics import Evolver, EvolverHooks

import rich

class MuSearchRecord(TypedDict):
    mu: List[float]
    omega_s: List[float]
    omega_l: List[float]

    mu_min_initial: float
    mu_max_initial: float
    
    mu_min_final: float
    mu_max_final: float


def is_liquid(psi: np.ndarray, tol=1e-5):
    return np.max(psi) - np.min(psi) <= tol


def find_coexistent_mu(
    solid_field: tg.RealField2D,
    mu_min: float, mu_max: float,
    fef: tg.FreeEnergyFunctional, *,
    
    relaxer_supplier: Callable[[tg.RealField2D,float], Evolver[tg.RealField2D]],
    relaxer_nsteps: int = 31,
    relaxer_hooks: Optional[EvolverHooks]=None,

    const_mu_supplier: Callable[[tg.RealField2D, float], Evolver[tg.RealField2D]],
    const_mu_nsteps: int = 31,
    const_mu_hooks: Optional[EvolverHooks]=None,
    
    max_iters: Optional[int]=None,
    precision: float=0.,

    verbose: bool = True
    ):
    '''
    Given: 
        - a solid profile 
        - minimum and maximum values of chemical potential
        - a free energy functional,

    find the chemical potential that satisfies:

            Omega[solid] = Omega[liquid]
    <=>  F[solid] - mu * N_s = F[liquid] - mu * N_l

    via binary search. 
    '''
    if mu_min >= mu_max:
        raise ValueError('mu_min must be smaller than mu_max')
    if max_iters is None and precision == 0.:
        raise ValueError('binary search will not stop with max_iters=None and precision=0') 

    if max_iters is None: max_iters = 2**32

    sol = solid_field
    liq = tg.liquefy(solid_field)

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

    for _ in range(max_iters): # type: ignore
        if mu_max - mu_min <= precision:
            break
        mu = (mu_min + mu_max) / 2
        if verbose:
            console.rule()
            console.log(f'current bounds: {mu_min} ~ {mu_max}')

        '''relax solid and resize liquid accordingly'''
        relaxer = relaxer_supplier(sol, mu)
        relaxer.run(relaxer_nsteps, hooks=relaxer_hooks)
        liq.set_size(sol.Lx, sol.Ly)
        
        '''evolve solid under constant mu'''
        # minim = const_mu_supplier(sol, mu)
        # minim.run(const_mu_nsteps, hooks=const_mu_hooks)

        '''calculate solid mean grand potential'''
        omega_s = fef.mean_free_energy_density(sol) - mu*sol.psi.mean()


        if is_liquid(sol.psi, tol=1e-4):
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

    return rec

