from __future__ import annotations
from abc import ABC, abstractmethod
from os import wait
import numpy as np


from ..utils.fft import rfft2, irfft2 

from typing import Dict, Generic, Optional, TypeVar, final
from typing_extensions import Self

from torusgrid.misc.typing import generic
import torusgrid as tg

import numpy.typing as npt




T  = TypeVar('T', bound=tg.RealField)
  
  
@generic
class FreeEnergyFunctionalBase(ABC, Generic[T]):
    '''
    Base class for free energy functional
    Only makes sense when the field.psi can be interpreted as a density.
    '''
    @abstractmethod
    def derivative(self, field: T):
        raise NotImplementedError()

    @abstractmethod
    def free_energy_density(self, field: T) -> npt.NDArray[np.floating]:
        raise NotImplementedError()

    def free_energy(self, field: T) -> np.floating:
        return np.sum(self.free_energy_density(field)) * field.dv

    def mean_free_energy_density(self, field: T) -> np.floating:
        return np.mean(self.free_energy_density(field))

    def grand_potential_density(self, field: T, mu: tg.FloatLike) -> npt.NDArray[np.floating]:
        return self.free_energy_density(field) - mu * field.psi

    def grand_potential(self, field: T, mu: tg.FloatLike) -> np.floating:
        return np.sum(self.grand_potential_density(field, mu)) * field.dv

    def mean_grand_potential_density(self, field: T, mu: tg.FloatLike) -> np.floating:
      return np.mean(self.free_energy_density(field))


class FreeEnergyFunctional(FreeEnergyFunctionalBase[tg.RealField]):
    '''
    PFC free energy fuctional
    '''
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


class StateFunction:
    '''
    An object representing a state function of a PFC field
    '''
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

    @property
    def Lx(self): 'system width'; return self._data['Lx']
    @property
    def Ly(self): 'system height'; return self._data['Ly']
    @property
    def f(self): 'mean free energy density'; return self._data['f']
    @property
    def F(self): 'free energy'; return self._data['F']
    @property
    def psibar(self): 'mean density'; return self._data['psibar']
    @property
    def omega(self): 'mean grand potential density'; return self._data['omega']
    @property
    def Omega(self): 'grand potential'; return self._data['Omega']

    @property
    def data(self) -> Dict[str,tg.FloatLike|None]:
        return self._data.copy()

    @classmethod
    def from_field(cls, 
            field: tg.RealField2D, eps: float, 
            mu: Optional[tg.FloatLike]=None) -> Self:
        '''
        Alternative constructor for fields.
        '''
        fef = FreeEnergyFunctional(eps)
        f = fef.mean_free_energy_density(field)
        F = fef.free_energy(field)
        psibar = field.psi.mean()
        
        omega = None
        Omega = None
        
        if mu is not None:
            omega = fef.mean_grand_potential_density(field, mu)
            Omega = fef.grand_potential(field, mu)

        return cls(field.lx, field.ly, f, F, psibar, omega, Omega)

    @property
    def is_grand_canonical(self) -> bool: return not (self.omega is None)

    
    def to_string(self, 
            state_string_format: Optional[str]=None,
            float_fmt: str='.3f', pad: int=1, delim: str='|') -> str:
        if state_string_format is not None:
            return state_string_format.format(
                Lx=self.Lx, Ly=self.Ly, f=self.f, F=self.F, 
                omega=self.omega, Omega=self.Omega, psibar=self.psibar
            )

        delim_padded = ' '*pad + delim + ' '*pad
        items = ['Lx', 'Ly', 'f', 'F', 'omega', 'Omega', 'psibar']
        state_func_list = []

        for item_name in items:
            item = self._data[item_name] # ! modified
            if not item is None: 
                state_func_list.append(f'{item_name}={item:{float_fmt}}')
        return delim_padded.join(state_func_list)

    def __repr__(self) -> str:
        if self.is_grand_canonical:

            return self.to_string(
                    'PFCStateFunction(Lx={Lx:.5f}, Ly={Ly:.5f}, f={f:.5f}, F={F:.5f}, ' +
                        'omega={omega:.5f}, Omega={Omega:.5f}'
            )
        else:
            return self.to_string(
                'PFCStateFunction(Lx={Lx:.5f}, Ly={Ly:.5f}, f={f:.5f}, F={F:.5f})'
            )
        


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


