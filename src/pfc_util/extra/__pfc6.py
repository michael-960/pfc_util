raise ImportError('Module is outdated and deprecated')

from __future__ import annotations
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
from scipy.fft import rfft2, irfft2

from michael960lib.common import overrides, deprecated, experimental
from torusgrid.fields import RealField2D
from torusgrid.dynamics import FreeEnergyFunctional2D, NoiseGenerator2D
from ..core.base import PFCFreeEnergyFunctional
from ..core.evolution import ConstantChemicalPotentialMinimizer, StressRelaxer, PFCMinimizer, NonlocalConservedRK4



class PFCFunctional6(PFCFreeEnergyFunctional):
    def __init__(self, eps: float, alpha: float):
        self.eps = eps
        self.alpha = alpha

    @overrides(FreeEnergyFunctional2D)
    def free_energy_density(self, field: RealField2D):
        kernel = (1-field.K2)**2 +\
                self.alpha*(field.K4*field.K2 - 2*field.K4 + field.K2)
        psi_k = rfft2(field.psi)
        psi_k_o = kernel * psi_k
        f = 1/2 * field.psi * irfft2(psi_k_o) + field.psi**4/4 - self.eps/2 * field.psi**2
        return np.real(f)

    @overrides(FreeEnergyFunctional2D)
    def derivative(self, field):
        field.fft()
        linear_term = ((1-field.K2)**2 + self.alpha*field.K2*(1-field.K2)**2 - self.eps) * field.psi_k
        field.ifft()

        local_mu = irfft2(linear_term) + field.psi**3
        return local_mu

class PFC6Minimizer(ConstantChemicalPotentialMinimizer):
    def __init__(self,
            field: RealField2D, dt: float, 
            eps: float, alpha: float, mu: float, 
            noise_generator:Optional[NoiseGenerator2D]=None):

        super().__init__(field, dt, eps, mu, noise_generator)
        self.alpha = alpha
        self.info['system'] = 'pfc6'
        self.info['alpha'] = alpha
        self.info['label'] = self.label = f'mu6 eps={eps:.5f} mu={mu:.5f} alpha={alpha:.5f} dt={dt:.5f}'
        self.fef = PFCFunctional6(eps, alpha)

        _kernel = 1 - 2*field.K2 + field.K4 +\
                alpha*(field.K4*field.K2 -2*field.K4 + field.K2)- eps

        self._exp_dt_kernel = np.exp(-dt*_kernel)
        self._mu_dt_half = dt * mu / 2


class PFC6NonlocalConservedRK4(NonlocalConservedRK4):
    def __init__(self, field: RealField2D, dt: float, eps: float, alpha: float,
            noise_generator: Optional[NoiseGenerator2D]=None,
            k_regularizer=0.1, inertia=100):
        super().__init__(field, dt, eps, noise_generator, k_regularizer, inertia)
        self.alpha = alpha
        self.info['system'] = 'pfc6'
        self.info['alpha'] = alpha
        self.info['label'] = self.label = f'pfc6 nonlocal-rk4 eps={eps} alpha={alpha} dt={dt} R={k_regularizer} M={inertia}'
        self.fef = PFCFunctional6(eps, alpha)



class PFC6Relaxer(StressRelaxer):
    def __init__(self, field: RealField2D, dt: float, eps: float, alpha: float, mu: float, expansion_rate: float=1.):
        super().__init__(field, dt, eps, mu, expansion_rate=expansion_rate)
        self.info['system'] = 'pfc6'
        self.info['alpha'] = alpha
        self.info['label'] = self.label = f'stress-relax6 eps={eps:.5f} mu={mu:.5f} alpha={alpha:.5f} dt={dt:.5f}'

        self.alpha = alpha
        self.fef = PFCFunctional6(eps, alpha)

    def _prepare_minimization(self):
        f = self.field
        self._mu_dt_half = self.dt * self.mu / 2
        self._2_NN = 2 / self.NN

        self._domega_kernels = np.array([
            2/f.Lx*f.Kx2*(1-f.K2), 
            2/f.Ly*f.Ky2*(1-f.K2), 
            -2/f.Lx**2*f.Kx2*(3-5*f.Kx2-3*f.Ky2),
            -2/f.Ly**2*f.Ky2*(3-5*f.Ky2-3*f.Kx2),
            2/f.Lx/f.Ly * 2*f.Kx2*f.Ky2
        ])

        self.Lx0 = f.Lx
        self.Ly0 = f.Ly

        self.Lx = f.Lx
        self.Ly = f.Ly
        self.set_size_scale(1., 1.)

    @overrides(PFCMinimizer)
    def step(self):
        f = self.field
        self.age += self.dt
        # if self.noise_generator is not None:
        #     f.psi += self.dt * self.noise_generator.generate() 

        f.psi += self._mu_dt_half
        f.psi /=np.sqrt(1+f.psi**2*self.dt)
        
        f.fft()

        _kernel = 1-2*self.K2+self.K4 - self.eps +\
                self.alpha*(self.K6 -2*self.K4 + self.K2)

        _exp_dt_kernel = np.exp(-self.dt*_kernel/2)
        f.psi_k *= _exp_dt_kernel

        self.relax_stress_full()

        _kernel = 1-2*self.K2+self.K4 - self.eps +\
                self.alpha*(self.K6 -2*self.K4 + self.K2)

        _exp_dt_kernel = np.exp(-self.dt*_kernel/2)
        f.psi_k *= _exp_dt_kernel


        f.ifft()

        f.psi /= np.sqrt(1+self.field.psi**2*self.dt)
        f.psi += self._mu_dt_half

    def relax_stress_full(self):
        f = self.field
        dT = self.dt

        self._domega_kernels[:,:,:] = [
                # domega/dLx
                1/self.Lx * self.Kx2 * (1-self.K2) * (2-self.alpha*(1-3*self.K2)),
                # domega/dLy
                1/self.Ly * self.Ky2 * (1-self.K2) * (2-self.alpha*(1-3*self.K2)),
                # d2omega/dLx2
                -1/self.Lx**2*self.Kx2 *\
                (6-10*self.Kx2-6*self.Ky2 - self.alpha*(3 - 20*self.Kx2 - 12*self.Ky2 + 21*self.Kx4 + 30*self.Kx2*self.Ky2 + 9*self.Ky4)),
                # d2omega/dLy2
                -1/self.Ly**2*self.Ky2 *\
                (6-10*self.Ky2-6*self.Kx2 - self.alpha*(3 - 20*self.Ky2 - 12*self.Kx2 + 21*self.Ky4 + 30*self.Ky2*self.Kx2 + 9*self.Kx4)),
                # d2omega/dLxdLy
                1/self.Lx/self.Ly * 4*self.Kx2*self.Ky2 * (1-self.alpha*(2-3*self.K2))
        ]

        omega_list = self.real_convolution_2d(np.abs(f.psi_k**2), self._domega_kernels)

        dLx = -omega_list[0]*dT + (omega_list[0] * omega_list[2] + omega_list[1] * omega_list[4]) * dT**2/2
        dLy = -omega_list[1]*dT + (omega_list[1] * omega_list[3] + omega_list[0] * omega_list[4]) * dT**2/2

        dfx = dLx / self.Lx0
        dfy = dLy / self.Ly0

        #f.set_size(f.Lx+dLx, f.Ly+dLy)
        self.set_size_scale(self.fx + dfx, self.fy + dfy)

    # size scale factor compared to the original dimensions
    def set_size_scale(self, fx, fy):
        self.fx = fx
        self.fy = fy

        self.Kx2 = self.field.Kx2 / fx**2
        self.Ky2 = self.field.Ky2 / fy**2
        self.K2 = self.Kx2 + self.Ky2

        self.Kx4 = self.Kx2**2
        self.Ky4 = self.Ky2**2
        self.K4 = self.K2**2

        self.K6 = self.K2**3

        self.Lx = self.field.Lx * self.fx
        self.Ly = self.field.Ly * self.fy



__all__ = ['PFCFunctional6', 'PFC6Minimizer']





