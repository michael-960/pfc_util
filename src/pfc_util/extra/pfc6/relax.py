from typing import List, final
from torusgrid import FloatLike
from torusgrid.dynamics import Step
from torusgrid.fields import RealField2D
import torusgrid as tg

import numpy as np
from ... import core

from .base import MinimizerMixin



@final
class StressRelaxer_(core.StressRelaxerBase, MinimizerMixin, core.MuMinimizerMixin):
    def __init__(self, 
            field: RealField2D,
                 dt: FloatLike,
                 eps: FloatLike, alpha: FloatLike, beta: FloatLike,
                 mu: FloatLike, *,
            expansion_rate: float=1., resize_cycle: int=31):

        super().__init__(field, dt, expansion_rate=expansion_rate, resize_cycle=resize_cycle)
        self.init_pfc_variables(eps, mu)
        self.init_pfc6_variables(alpha, beta)
        self.info['label'] = self.label = f'stress-relax6 eps={eps:.5f} mu={mu:.5f} alpha={alpha:.5f} dt={dt:.5f}'

        self._mu_dt_half = dt * mu / 2
        self.initialize_fft()

    def on_size_changed(self):

        self.Kx2 = self.field.kx**2 / self.fx**2
        self.Ky2 = self.field.ky**2 / self.fy**2

        self.K2 = self.Kx2 + self.Ky2

        self.Kx4 = self.Kx2**2
        self.Ky4 = self.Ky2**2
        self.K4 = self.K2**2

        self.K6 = self.K2**3
        self._kernel = 1-2*self.K2+self.K4 - self.eps +\
                       self.alpha*(self.K6 -2*self.K4 + self.K2)

    def get_realspace_steps(self) -> List[Step]:
        f = self.field
        def step_mu(dt: FloatLike):
            f.psi[...] += self._mu_dt_half

        def step_nonlin(dt: FloatLike):
            f.psi[...] /= np.sqrt(1+f.psi**2*self.dt)

        return [step_mu, step_nonlin]

    def step_kernel(self, dt: float):
        _exp_dt_kernel = np.exp(-dt*self._kernel)
        self.field.psi_k[...] *= _exp_dt_kernel

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



@final
class StressRelaxer(
        core.StressRelaxerBase,
        MinimizerMixin,
        core.MuMinimizerMixin):
    """
    PFC6 stress relaxer
    """
    def __init__(self, 
            field: RealField2D,
            dt: FloatLike,
            eps: FloatLike, alpha: FloatLike, beta: FloatLike,
            mu: FloatLike, *,
            expansion_rate: float=1., resize_cycle: int=31):

        super().__init__(field, dt, expansion_rate=expansion_rate, resize_cycle=resize_cycle)
        self.init_pfc6_variables(eps, alpha, beta)
        self.init_mu(mu)


        self._dtype = tg.get_real_dtype(field.precision)

        self._mu_dt_half = dt * mu / 2
        self.initialize_fft()

    def start(self) -> None:
        super().start()
        self.data['minimizer'] = 'stress-relax'

    def on_size_changed(self):

        self.Kx2 = self.field.kx**2 / self.fx**2
        self.Ky2 = self.field.ky**2 / self.fy**2

        self.K2 = self.Kx2 + self.Ky2

        self.Kx4 = self.Kx2**2
        self.Ky4 = self.Ky2**2
        self.K4 = self.K2**2

        self.K6 = self.K2**3
        self._kernel = 1-2*self.K2+self.K4 - self.eps +\
                       self.alpha*(self.K6 -2*self.K4 + self.K2)

    def get_realspace_steps(self) -> List[Step]:
        f = self.field
        def step_mu(dt: FloatLike):
            f.psi[...] += self._mu_dt_half

        def step_nonlin_4(dt: FloatLike):
            f.psi[...] /= np.sqrt(1+f.psi**2*self.dt)

        def step_nonlin_6(dt: tg.FloatLike):
            self.field.psi[...] /= np.power(
                    1+2*self.beta*self.field.psi**4*self.dt,
                    1 / self._dtype('4'))

        return [step_mu, step_nonlin_4, step_nonlin_6]

    def step_kernel(self, dt: float):
        _exp_dt_kernel = np.exp(-dt*self._kernel)
        self.field.psi_k[...] *= _exp_dt_kernel

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

