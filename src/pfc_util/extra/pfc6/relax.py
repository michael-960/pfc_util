
from typing import List, final
from torusgrid.dynamics import Step
from torusgrid.fields import RealField2D
import numpy as np
from ... import core

from .base import MinimizerMixin



@final
class StressRelaxer(core.StressRelaxerBase, MinimizerMixin, core.MuMinimizerMixin):
    def __init__(self, 
            field: RealField2D, dt: float, eps: float, alpha: float, mu: float, *,
            expansion_rate: float=1., resize_cycle: int=31):

        super().__init__(field, dt, expansion_rate=expansion_rate, resize_cycle=resize_cycle)
        self.init_pfc_variables(eps, mu)
        self.init_pfc6_variables(alpha)

        self.info['label'] = self.label = f'stress-relax6 eps={eps:.5f} mu={mu:.5f} alpha={alpha:.5f} dt={dt:.5f}'

        self._mu_dt_half = dt * mu / 2

        self.initialize_fft()

    def on_size_changed(self):

        self.Kx2 = self.field.Kx2 / self.fx**2
        self.Ky2 = self.field.Ky2 / self.fy**2
        self.K2 = self.Kx2 + self.Ky2

        self.Kx4 = self.Kx2**2
        self.Ky4 = self.Ky2**2
        self.K4 = self.K2**2

        self.K6 = self.K2**3
        self._kernel = 1-2*self.K2+self.K4 - self.eps +\
                       self.alpha*(self.K6 -2*self.K4 + self.K2)

    def get_realspace_steps(self) -> List[Step]:
        f = self.field
        def step_mu(dt: float):
            f.psi += self._mu_dt_half

        def step_nonlin(dt: float):
            f.psi /=np.sqrt(1+f.psi**2*self.dt)

        return [step_mu, step_nonlin]

    def step_kernel(self, dt: float):
        _exp_dt_kernel = np.exp(-dt*self._kernel)
        self.field.psi_k *= _exp_dt_kernel

    def update_domega_kernels(self):
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

