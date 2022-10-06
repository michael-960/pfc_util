from typing import Optional, Callable, List

import numpy as np

from torusgrid.fields import RealField2D
from torusgrid import field_util as fu
from torusgrid.dynamics import EvolverCallBack, FreeEnergyFunctional2D

from ..core.base import PFCFreeEnergyFunctional
from ..core.evolution import PFCMinimizer
from ..pfc import PFC



def find_coexistent_mu(solid_field: RealField2D, dt: float, eps: float, mu_min: float, mu_max: float, 
        relax=True, max_iters: int=None, precision: float=0., **minimizer_kwargs) -> dict:

    if mu_min >= mu_max:
        raise ValueError('mu_min must be smaller than mu_max')
    
    if max_iters is None and precision == 0.:
        raise ValueError('binary search will not stop with max_iters=None and precision=0') 

    fef = PFCFreeEnergyFunctional(eps)
    model_s = PFC(solid_field)
    model_l = PFC(fu.liquefy(solid_field))

    solid_psi = model_s.field.psi.copy()
    liquid_psi = model_l.field.psi.copy()
    
    rec = {'mu': [], 'omega_s': [], 'omega_l': [], 'mu_min_initial': mu_min, 'mu_max_initial': mu_max}
    
    print('===========================================================================')

    for i in range(max_iters):
        if mu_max - mu_min <= precision:
            break
        mu = (mu_min + mu_max) / 2
        print(f'current bounds: {mu_min} ~ {mu_max}')

        if relax:
            model_s.evolve(minimizer='relax', dt=dt, eps=eps, mu=mu, expansion_rate=1., **minimizer_kwargs)
            model_l.field.set_size(model_s.field.Lx, model_s.field.Ly)

        model_s.evolve(minimizer='mu', dt=dt, eps=eps, mu=mu, **minimizer_kwargs)
        omega_s = fef.mean_grand_potential_density(model_s.field, mu)

        if is_liquid(model_s.field.psi, tol=1e-4):
            print(f'solid field was liquefied during minimization with mu={mu}')
            break
            #mu_max = mu
            #model_s.field.set_psi(solid_psi)
            #continue

        model_l.evolve(minimizer='mu', dt=0.001, eps=eps, mu=mu, **minimizer_kwargs)
        omega_l = fef.mean_grand_potential_density(model_l.field, mu)

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


# works for a generic free energy functional
# fef: free energy functional
def find_coexistent_mu_general(
    solid_field: RealField2D,
    mu_min: float, mu_max: float,
    fef: FreeEnergyFunctional2D,
    minimizer_supplier: Callable[[RealField2D, float], PFCMinimizer], 
    relaxer_supplier: Optional[Callable[[RealField2D, float], PFCMinimizer]]=None,
    max_iters: Optional[int]=None, precision: float=0.,
    relax_callbacks: List[EvolverCallBack]=[],
    minim_callbacks: List[EvolverCallBack]=[],
    **minimizer_kwargs):

    if mu_min >= mu_max:
        raise ValueError('mu_min must be smaller than mu_max')
    
    if max_iters is None and precision == 0.:
        raise ValueError('binary search will not stop with max_iters=None and precision=0') 

    sol = solid_field
    liq = fu.liquefy(solid_field)

    #solid_psi = model_s.field.psi.copy()
    #liquid_psi = model_l.field.psi.copy()
    
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


def is_liquid(psi: np.ndarray, tol=1e-5):
    return np.max(psi) - np.min(psi) <= tol


# [interface] -> [left][right]
# elongation: [delta_liq][------left------][delta_sol][delta_sol][------right------][delta_liq]
def evolve_and_elongate_interface(
        ifc: RealField2D, delta_sol: RealField2D, delta_liq: RealField2D,
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


     
