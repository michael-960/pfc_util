from __future__ import annotations
from typing import Tuple
import numpy as np

from torusgrid.fields import RealField2D
from torusgrid.dynamics import SecondOrderRK4, FirstOrderRK4
import torusgrid as tg


import numpy.typing as npt

from ..base import MinimizerMixin


class NonlocalConservedMinimizer(MinimizerMixin):
    '''
    Nonlocal conserved dynamics by forcing mean density to be constant
    '''
    def __init__(self,
            field: RealField2D, 
            dt: tg.FloatLike, eps: tg.FloatLike):
        super().__init__(field, dt)
        self.init_pfc_variables(eps)

        self.info['minimizer'] = 'nonlocal'
        self.info['label'] = self.label = f'nonlocal eps={eps:.5f} dt={dt:.5f}'

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



class NonlocalConservedRK4(SecondOrderRK4[RealField2D], MinimizerMixin):
    '''
    RK4 nonlocal conserved dynamics with inertia
    '''
    def __init__(self, 
            field: RealField2D, dt: float, eps: float, *,
            k_regularizer=0.1, inertia=100):
        super().__init__(field, dt)
        self.init_pfc_variables(eps)

        self.R = k_regularizer
        self.inertia = inertia

        self.info['minimizer'] = 'nonlocal-rk4'
        self.info['M'] = inertia
        self.info['R'] = k_regularizer
        self.info['label'] = self.label = f'nonlocal-rk4 eps={eps} dt={dt} R={k_regularizer} M={inertia}'

        self.initialize_fft()

        self._deriv = RealField2D(
                field.lx, field.ly, field.nx, field.ny,
                precision=field._precision)

        self._deriv.initialize_fft()



    def psi_dot(self) -> Tuple[npt.NDArray, npt.NDArray]:
        self._deriv.psi[...] = -self.fef.derivative(self.grid_tmp)
        self._deriv.fft()
        self._deriv.psi_k[...] *= np.exp(-self.R*self._deriv.k2)
        self._deriv.ifft()
        F = self._deriv.psi - np.mean(self._deriv.psi)
        return self.dgrid_tmp.psi*self.inertia, -(self.dgrid_tmp.psi - F)


class NonlocalConservedRK4Plain(FirstOrderRK4[RealField2D], MinimizerMixin):
    '''
    RK4 nonlocal conserved dynamics without inertia
    '''

    def __init__(self, 
            field: RealField2D, dt: float, eps: float, *,
            k_regularizer=0.1):
        
        super().__init__(field, dt)
        self.init_pfc_variables(eps)

        self.R = k_regularizer

        self.info['minimizer'] = 'nonlocal-rk4'
        self.info['R'] = k_regularizer
        self.info['label'] = self.label = f'nonlocal-rk4 eps={eps} dt={dt} R={k_regularizer}'

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


class NonlocalDescent(MinimizerMixin):
    def __init__(self, 
            field: RealField2D, dt: float, eps: float, *,
            k_regularizer: float=0.1):

        super().__init__(field, dt)
        self.init_pfc_variables(eps)

        self.R = k_regularizer
        self.info['minimizer'] = 'nonlocal-desc'
        self.info['R'] = k_regularizer
        self.info['label'] = self.label = f'nonlocal-desc eps={eps} dt={dt} R={k_regularizer}'

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




