import numpy as np

from ..utils.fft import rfft2, irfft2 

from typing import Dict, Optional, final
from typing_extensions import Self

from torusgrid.fields import RealField2D

from torusgrid.fields import FreeEnergyFunctional as _FreeEnergyFunctional
import numpy.typing as npt



class FreeEnergyFunctional(_FreeEnergyFunctional[RealField2D]):
    '''
    PFC free energy fuctional
    '''
    def __init__(self, eps: float):
        self.eps = eps

    def free_energy_density(self, field: RealField2D) -> npt.NDArray[np.float_]:
        kernel = 1-2*field.K2+field.K4
        psi_k = rfft2(field.psi)
        psi_k_o = kernel * psi_k
        
        f = 1/2 * field.psi * irfft2(psi_k_o) + field.psi**4/4 - self.eps/2 * field.psi**2
        return np.real(f)

    def derivative(self, field: RealField2D) -> npt.NDArray[np.float_]:
        field.fft()
        linear_term = ((1-field.K2)**2 - self.eps) * field.psi_k
        field.ifft()

        #local_mu = (1-self.eps) * field.psi + field.psi**3 + 2*D2psi + D4psi
        local_mu = irfft2(linear_term) + field.psi**3
        return local_mu

    def grand_potential_density(self, field: RealField2D, mu: float) -> npt.NDArray[np.float_]:
        return self.free_energy_density(field) - mu*field.psi

    def mean_grand_potential_density(self, field: RealField2D, mu: float) -> float:
        return np.mean(self.grand_potential_density(field, mu))

    def grand_potential(self, field: RealField2D, mu: float) -> float:
        return np.sum(self.grand_potential_density(field, mu)) * field.dV


class StateFunction:
    '''
    An object representing a state function of a PFC field
    '''
    def __init__(self, 
            Lx: float, Ly: float, f: float, F: float, psibar: float,
            omega: Optional[float]=None, Omega: Optional[float]=None):
        self._content: Dict[str, float|None] = {}

        self._content['Lx'] = Lx
        self._content['Ly'] = Ly
        self._content['f'] = f
        self._content['F'] = F
        self._content['psibar'] = psibar
        self._content['omega'] = omega
        self._content['Omega'] = Omega

    @property
    def Lx(self): 'system width'; return self._content['Lx']
    @property
    def Ly(self): 'system height'; return self._content['Ly']
    @property
    def f(self): 'mean free energy density'; return self._content['f']
    @property
    def F(self): 'free energy'; return self._content['F']
    @property
    def psibar(self): 'mean density'; return self._content['psibar']
    @property
    def omega(self): 'mean grand potential density'; return self._content['omega']
    @property
    def Omega(self): 'grand potential'; return self._content['Omega']

    @classmethod
    def from_field(cls, 
            field: RealField2D, eps: float, 
            mu: Optional[float]=None) -> Self:
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

        return cls(field.Lx, field.Ly, f, F, psibar, omega, Omega)

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
            item = self._content[item_name] # ! modified
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


