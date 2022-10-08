from typing import Callable, List

import numpy as np

from torusgrid.fields import RealField2D

from ..base import MuMinimizerMixin

from ..abc import StressRelaxerBase



class StressRelaxer(StressRelaxerBase, MuMinimizerMixin):
    '''
    Constant mu stress relaxer
    '''
    def __init__(self, field: RealField2D, 
            dt: float, eps: float, mu: float, *,
            expansion_rate: float=1., resize_cycle: int = 31):


        super().__init__(
                field, dt,
                expansion_rate=expansion_rate, 
                resize_cycle=resize_cycle)

        self.init_pfc_variables(eps, mu)

        self.mu = mu
        self.info['mu'] = mu
        self.info['label'] = self.label = f'stress-relax eps={eps:.5f} mu={mu:.5f} dt={dt:.5f}'
        self.info['minimizer'] = 'stress-relax'
        self.info['expansion_rate'] = expansion_rate

        self.display_format = '[{label}][{age:.4f}] Lx={Lx:.5f} Ly={Ly:.5f} f={f:.5f} F={F:.5f} omega={omega:.5f} Omega={Omega:.5f} '

        self.initialize_fft()

    def on_size_changed(self):
        self.Kx2 = self.field.Kx2 / self.fx**2
        self.Ky2 = self.field.Ky2 / self.fy**2
        self.K2 = self.Kx2 + self.Ky2
        self.K4 = self.K2**2
        self._kernel = 1-2*self.K2+self.K4 - self.eps

    def get_realspace_steps(self) -> List[Callable[[float], None]]:
        f = self.field
        def step_mu(dt: float):
            f.psi += dt * self.mu

        def step_nonlin(dt: float):
            f.psi /=np.sqrt(1+f.psi**2*dt*2)

        return [step_mu, step_nonlin]

    def step_kernel(self, dt: float):
        f = self.field
        _exp_dt_kernel = np.exp(-dt*self._kernel)
        f.psi_k *= _exp_dt_kernel

    def update_domega_kernels(self):
        self._domega_kernels[:,:,:] = [
                2/self.Lx*self.Kx2*(1-self.K2), 
                2/self.Ly*self.Ky2*(1-self.K2), 
                -2/self.Lx**2*self.Kx2*(3-5*self.Kx2-3*self.Ky2),
                -2/self.Ly**2*self.Ky2*(3-5*self.Ky2-3*self.Kx2),
                2/self.Lx/self.Ly * 2*self.Kx2*self.Ky2
        ]

