from __future__ import annotations
from abc import ABC, abstractmethod
from typing_extensions import Self
import numpy as np



from typing import Dict, Generic, List, Optional, Tuple, TypeVar

import torusgrid as tg

import torusgrid as tg

import numpy.typing as npt



T  = TypeVar('T', bound=tg.RealField)


class FreeEnergyFunctionalBase(ABC, Generic[T]):
    """
    Abstract base class for free energy functional
    Only makes sense when the field.psi can be interpreted as a density.
    """
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
      return np.mean(self.grand_potential_density(field, mu))


T_Any = TypeVar('T_Any')

class StateFunctionBase(ABC, Generic[T_Any]):
    """
    Abstract base class for field state functions
    """
    @property
    @abstractmethod
    def data(self) -> Dict[str,tg.FloatLike|None]: ...

    @classmethod
    @abstractmethod
    def from_(
            cls,
            obj: T_Any, /, **environment
    ) -> Self:
        """
        Given a target object and environment, return a state function
        """

    @staticmethod
    @abstractmethod
    def environment_params() -> Tuple[List[str], List[str]]:
        """
        Return a tuple of two lists of parameter names used to calculate the state function

        The first list contains required parameters while the ones in the second list are optional
        """


class FieldStateFunction2D(StateFunctionBase[tg.RealField2D]):
    """
    Base class for 2D field state functions
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

    @classmethod
    def from_(
        cls, 
        field: tg.RealField2D, /, 
        mu: tg.FloatLike|None=None, **environment
    ) -> Self:
        """
        Alternative constructor for fields.
        """
        fef = cls.free_energy_functional(**environment)

        f = fef.mean_free_energy_density(field)
        F = fef.free_energy(field)
        psibar = field.psi.mean()

        
        omega = None
        Omega = None
        
        if mu is not None:
            omega = fef.mean_grand_potential_density(field, mu)
            Omega = fef.grand_potential(field, mu)

        return cls(field.lx, field.ly, f, F, psibar, omega, Omega)

    @classmethod
    def from_field(
        cls, field: tg.RealField2D, /,
        mu: tg.FloatLike|None=None, **environment
    ) -> Self:
        """
        Alias of from_
        """
        return cls.from_(field, mu=mu, **environment)

    @classmethod
    @abstractmethod
    def free_energy_functional(cls, **environment) -> FreeEnergyFunctionalBase[tg.RealField2D]: ...


    @property
    def data(self) -> Dict[str, tg.FloatLike | None]:
        return self._data.copy()

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
                    type(self).__name__
                    + '(Lx={Lx:.5f}, Ly={Ly:.5f}, f={f:.5f}, F={F:.5f}, '
                    + 'omega={omega:.5f}, Omega={Omega:.5f}'
            )
        else:
            return self.to_string(
                type(self).__name__ + '(Lx={Lx:.5f}, Ly={Ly:.5f}, f={f:.5f}, F={F:.5f})'
            )





