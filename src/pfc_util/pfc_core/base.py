import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, rfft2, irfft2, set_global_backend
from pprint import pprint
import tqdm

import pyfftw
import time
import threading
import sys
from typing import List

from michael960lib.math import fourier
from michael960lib.common import overrides

from torusgrid.fields import FreeEnergyFunctional2D, FieldMinimizer, RealField2D, NoiseGenerator2D, import_field, real_convolution_2d
from michael960lib.common import ModifyingReadOnlyObjectError, IllegalActionError


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
        D2psi = irfft2(-field.K2*rfft2(field.psi))
        D4psi = irfft2(field.K4*rfft2(field.psi))
        local_mu = (1-self.eps) * field.psi + field.psi**3 + 2*D2psi + D4psi
        return local_mu

    def grand_potential_density(self, field: RealField2D, mu: float):
        return self.free_energy_density(field) - mu*field.psi

    def mean_grand_potential_density(self, field: RealField2D, mu: float):
        return np.mean(self.grand_potential_density(field, mu))

    def grand_potential(self, field: RealField2D, mu: float):
        return np.sum(self.grand_potential_density(field, mu)) * field.dV

class PFCStateFunction:
    def __init__(self, Lx, Ly, f, F, psibar, omega=None, Omega=None):
        self.Lx = Lx
        self.Ly = Ly

        self.f = f
        self.F = F
        self.psibar = psibar

        self.omega = omega
        self.Omega = Omega

        self.item_dict = {
            'Lx': self.Lx,
            'Ly': self.Ly,
            'f': self.f,
            'F': self.F,
            'omega': self.omega,
            'Omega': self.Omega,
            'psibar': self.psibar
        } 

    def is_grand_canonical(self):
        return not (self.omega is None)


    def get_item(self, item_name):
        if not item_name in self.item_dict.keys():
            raise ValueError(f'{item_name} is not a valid PFC state function')
        return self.item_dict[item_name]

    def to_string(self, float_fmt='.7f', pad=1, delim='|'):
        delim_padded = ' '*pad + delim + ' '*pad
        items = ['Lx', 'Ly', 'f', 'F', 'omega', 'Omega', 'psibar']
        state_func_list = []

        for item_name in items:
            item = self.item_dict[item_name]
            if not item is None: 
                state_func_list.append(f'{item_name}={item:{float_fmt}}')
        return delim_padded.join(state_func_list)

    def export(self) -> str:
        return self.item_dict

def import_state_function(state: dict) -> PFCStateFunction:
    sf = PFCStateFunction(state['Lx'], state['Ly'], state['f'], state['F'], state['psibar'], state['omega'], state['Omega'])
    return sf

def get_latex(item_name):
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


