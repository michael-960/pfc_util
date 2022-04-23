import numpy as np
from torusgrid.fields import RealField2D
from ..pfc import PFC
from torusgrid.field_util import liquefy
from ..pfc_core.base import PFCFreeEnergyFunctional


def find_coexistent_mu(solid_field: RealField2D, eps: float, mu_min: float, mu_max: float, 
        relax=True, max_iters: int=None, precision: float=0., **minimizer_kwargs) -> dict:

    if mu_min >= mu_max:
        raise ValueError('mu_min must be smaller than mu_max')
    
    if max_iters is None and precision == 0.:
        raise ValueError('binary search will not stop with max_iters=None and precision=0') 

    fef = PFCFreeEnergyFunctional(eps)
    model_s = PFC(solid_field)
    model_l = PFC(liquefy(solid_field))

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
            model_s.evolve('relax', 0.001, eps, mu, expansion_rate=1., **minimizer_kwargs)
            model_l.field.set_size(model_s.field.Lx, model_s.field.Ly)

        model_s.evolve('mu', 0.001, eps, mu, **minimizer_kwargs)
        omega_s = fef.mean_grand_potential_density(model_s.field, mu)

        if is_liquid(model_s.field.psi, tol=1e-4):
            print(f'solid field was liquefied during minimization with mu={mu}')
            break
            #mu_max = mu
            #model_s.field.set_psi(solid_psi)
            #continue

        model_l.evolve('mu', 0.001, eps, mu, **minimizer_kwargs)
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

def is_liquid(psi: np.ndarray, tol=1e-5):
    return np.max(psi) - np.min(psi) <= tol
