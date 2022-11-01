from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from ..utils.fft import rfft2, irfft2 

from typing import Dict, Generic, List, Optional, Tuple, TypeVar, final
from typing_extensions import Self

import torusgrid as tg

import torusgrid as tg

import numpy.typing as npt

from .abc import FieldStateFunction2D, FreeEnergyFunctionalBase, StateFunctionBase



class FreeEnergyFunctional(FreeEnergyFunctionalBase[tg.RealField2D]):
    """
    PFC free energy fuctional
    """
    def __init__(self, eps: tg.FloatLike):
        self.eps = eps

    def free_energy_density(self, field: tg.RealField2D) -> npt.NDArray[np.floating]:
        kernel = 1-2*field.k2+field.k4
        psi_k = rfft2(field.psi)
        psi_k_o = kernel * psi_k
        
        f = 1/2 * field.psi * irfft2(psi_k_o) + field.psi**4/4 - self.eps/2 * field.psi**2
        return np.real(f)

    def derivative(self, field: tg.RealField2D) -> npt.NDArray[np.floating]:
        field.fft()
        linear_term = ((1-field.k2)**2 - self.eps) * field.psi_k
        field.ifft()

        local_mu = irfft2(linear_term) + field.psi**3

        return local_mu


@final
class StateFunction(FieldStateFunction2D):
    """
    An object representing a state function of a PFC field
    """
    def __init__(self, 
            Lx: tg.FloatLike, Ly: tg.FloatLike,
            f: tg.FloatLike, F: tg.FloatLike, psibar: tg.FloatLike,
            omega: Optional[tg.FloatLike]=None, Omega: Optional[tg.FloatLike]=None):

        self._data: Dict[str, tg.FloatLike|None] = {}

        self._data['Lx'] = Lx
        self._data['Ly'] = Ly
        self._data['f'] = f
        self._data['F'] = F
        self._data['psibar'] = psibar
        self._data['omega'] = omega
        self._data['Omega'] = Omega

    @staticmethod
    def environment_params() -> Tuple[List[str], List[str]]:
        return ['eps'], ['mu']

    @property
    def is_grand_canonical(self) -> bool: return not (self.omega is None)

    @classmethod
    def free_energy_functional(cls, *, eps: tg.FloatLike) -> FreeEnergyFunctionalBase[tg.RealField2D]:
        return FreeEnergyFunctional(eps)
        

def import_state_function(state: dict) -> StateFunction:
    sf = StateFunction(state['Lx'], state['Ly'], state['f'], state['F'], state['psibar'], state['omega'], state['Omega'])
    return sf

def get_latex(item_name) -> str:
    if not item_name in _item_latex_dict.keys():
        raise ValueError(f'{item_name} is not a valid state function function')
    return _item_latex_dict[item_name]

_item_latex_dict = {
    'Lx': r'$L_x$',
    'Ly': r'$L_y$',
    'f': r'$f$',
    'F': r'$F$',
    'omega': r'$\omega$',
    'Omega': r'$\Omega$',
    'psibar': r'$\bar\psi$'
}


