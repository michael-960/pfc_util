from __future__ import annotations
from typing import Tuple, final
import numpy as np

from torusgrid.fields import RealField2D
from torusgrid.dynamics import SecondOrderRK4, FirstOrderRK4, TemporalEvolver
import torusgrid as tg


import numpy.typing as npt

from .base import MinimizerMixin, NonlocalConservedRK4Base



class NonlocalConservedMinimizer(TemporalEvolver[RealField2D], MinimizerMixin):
    """
    Nonlocal conserved dynamics by forcing mean density to be constant
    """
    def __init__(self,
            field: RealField2D, 
            dt: tg.FloatLike, eps: tg.FloatLike):

        super().__init__(field, dt)

        self.init_pfc_variables(eps)

        self._kernel = 1-2*self.field.k2+self.field.k4 - self.eps
        self._exp_dt_kernel = np.exp(-dt*self._kernel)

        self.psibar = np.mean(self.field.psi)
        self.initialize_fft()
    
    def step(self):
        self.set_age(self.age + self.dt)
        
        self.field.psi[...] /= np.sqrt(1+self.field.psi**2*self.dt)
        self.field.psi[...] += - np.mean(self.field.psi) + self.psibar

        self.field.fft()
        self.field.psi_k[...] *= self._exp_dt_kernel
        self.field.ifft()

        self.field.psi[...] += - np.mean(self.field.psi) + self.psibar

        self.field.psi[...] /= np.sqrt(1+self.field.psi**2*self.dt)
        self.field.psi[...] += - np.mean(self.field.psi) + self.psibar

    def start(self) -> None:
        super().start()
        self.data['minimizer'] = 'nonlocal'


# class NonlocalConservedRK4(SecondOrderRK4[RealField2D], MinimizerMixin):
class NonlocalConservedRK4(NonlocalConservedRK4Base, MinimizerMixin):
    """
    RK4 nonlocal conserved dynamics with inertia
    """
    def __init__(self, 
            field: RealField2D, 
            dt: tg.FloatLike, eps: tg.FloatLike, *,
            k_regularizer: tg.FloatLike=0.1, 
            inertia: tg.FloatLike=100):

        super().__init__(field, dt, 
                         k_regularizer=k_regularizer, inertia=inertia)
        self.init_pfc_variables(eps)
        self.initialize_fft()

    def derivative(self, f: RealField2D) -> npt.NDArray:
        return self.fef.derivative(f)

    def start(self) -> None:
        super().start()
        self.data['minimizer'] = 'nonlocal-rk4'


class NonlocalConservedRK4Plain(FirstOrderRK4[RealField2D], MinimizerMixin):
    """
    RK4 nonlocal conserved dynamics without inertia
    """

    def __init__(self, 
            field: RealField2D, dt: float, eps: float, *,
            k_regularizer=0.1):
        
        super().__init__(field, dt)
        self.init_pfc_variables(eps)

        self.R = k_regularizer

        self.data['minimizer'] = 'nonlocal-rk4'
        self.data['R'] = k_regularizer
        self.data['label'] = self.label = f'nonlocal-rk4 eps={eps} dt={dt} R={k_regularizer}'

        self.initialize_fft()

        self._deriv = RealField2D(field.lx, field.ly, field.nx, field.ny)
        self._deriv.initialize_fft()

    def psi_dot(self) -> npt.NDArray:
        self._deriv.psi[:,:] = -self.fef.derivative(self.grid_tmp)
        self._deriv.fft()
        self._deriv.psi_k[...] *= np.exp(-self.R*self._deriv.k2)
        self._deriv.ifft()
        F = self._deriv.psi - np.mean(self._deriv.psi)
        return F


class NonlocalDescent(TemporalEvolver[RealField2D], MinimizerMixin):
    def __init__(self, 
            field: RealField2D, dt: float, eps: float, *,
            k_regularizer: float=0.1):

        super().__init__(field, dt)
        self.init_pfc_variables(eps)

        self.R = k_regularizer
        
        self.initialize_fft()

        self.field_tmp = self.field.copy()
        self.field_tmp.initialize_fft()

        self._deriv = RealField2D(
                field.lx, field.ly, field.nx, field.ny,
                precision=field._precision
                )
        self._deriv.initialize_fft()

    def psi_dot(self):
        self._deriv.psi[:,:] = -self.fef.derivative(self.field_tmp)
        self._deriv.fft()
        self._deriv.psi_k[...] *= np.exp(-self.R*self._deriv.k2)
        self._deriv.ifft()
        F = self._deriv.psi - np.mean(self._deriv.psi)
        return F

    def step(self):
        self.set_age(self.age + self.dt)
        self.field_tmp.psi[:,:] = self.field.psi[:,:]
        F = self.psi_dot()
        cutoff = 0.01 * self.dt

        dT = self.dt / (np.sum(F**2)*self.field.dv + cutoff)
        self.field.psi[...] += F * dT

    def start(self) -> None:
        super().start()
        self.data['minimizer'] = 'nonlocal-desc'
        self.data['R'] = self.R


