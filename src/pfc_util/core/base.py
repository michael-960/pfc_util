import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import rfft2, irfft2
import pyfftw

from pprint import pprint
import tqdm
import time
import threading
import sys
from typing import List, Optional

from michael960lib.math import fourier
from michael960lib.common import overrides, ModifyingReadOnlyObjectError, IllegalActionError
from torusgrid.fields import RealField2D, import_field, FieldStateFunction
from torusgrid.dynamics import FreeEnergyFunctional2D, NoiseGenerator2D


class PFCFreeEnergyFunctional(FreeEnergyFunctional2D):
    def __init__(self, eps):
        self.eps = eps

    @overrides(FreeEnergyFunctional2D)
    def free_energy_density(self, field: RealField2D):
        kernel = 1-2*field.K2+field.K4
        psi_k = rfft2(field.psi)
        psi_k_o = kernel * psi_k
        f = 1/2 * field.psi * irfft2(psi_k_o) + field.psi**4/4 - self.eps/2 * field.psi**2
        return np.real(f)

    @overrides(FreeEnergyFunctional2D)
    def derivative(self, field: RealField2D):
        field.fft()
        linear_term = ((1-field.K2)**2 - self.eps) * field.psi_k
        field.ifft()

        #local_mu = (1-self.eps) * field.psi + field.psi**3 + 2*D2psi + D4psi
        local_mu = irfft2(linear_term) + field.psi**3
        return local_mu

    def grand_potential_density(self, field: RealField2D, mu: float):
        return self.free_energy_density(field) - mu*field.psi

    def mean_grand_potential_density(self, field: RealField2D, mu: float):
        return np.mean(self.grand_potential_density(field, mu))

    def grand_potential(self, field: RealField2D, mu: float):
        return np.sum(self.grand_potential_density(field, mu)) * field.dV

class PFCStateFunction(FieldStateFunction):
    def __init__(self, Lx, Ly, f, F, psibar, omega=None, Omega=None):
        super().__init__()
        self.Lx = self._content['Lx'] = Lx
        self.Ly = self._content['Ly'] = Ly
        self.f = self._content['f'] = f
        self.F = self._content['F'] = F
        self.psibar = self._content['psibar'] = psibar
        self.omega = self._content['omega'] = omega
        self.Omega = self._content['Omega'] = Omega

    def is_grand_canonical(self) -> bool:
        return not (self.omega is None)

    def to_string(self, state_string_format: Optional[str]=None,
            float_fmt: str='.3f', pad: int=1, delim: str='|') -> str:
        if not state_string_format is None:
            return state_string_format.format(
                Lx=self.Lx, Ly=self.Ly, f=self.f, F=self.F, 
                omega=self.omega, Omega=self.Omega, psibar=self.psibar
            )

        delim_padded = ' '*pad + delim + ' '*pad
        items = ['Lx', 'Ly', 'f', 'F', 'omega', 'Omega', 'psibar']
        state_func_list = []

        for item_name in items:
            item = self.item_dict[item_name]
            if not item is None: 
                state_func_list.append(f'{item_name}={item:{float_fmt}}')
        return delim_padded.join(state_func_list)


def import_state_function(state: dict) -> PFCStateFunction:
    sf = PFCStateFunction(state['Lx'], state['Ly'], state['f'], state['F'], state['psibar'], state['omega'], state['Omega'])
    return sf

def get_latex(item_name) -> str:
    if not item_name in _item_latex_dict.keys():
        raise ValueError(f'{item_name} is not a valid state function function')
    return _item_latex_dict[item_name]

_item_latex_dict = {
    'Lx': r'$Lx$',
    'Ly': r'$Ly$',
    'f': r'$f$',
    'F': r'$F$',
    'omega': r'$\omega$',
    'Omega': r'$\Omega$',
    'psibar': r'$\bar\psi$'
}


