from typing import Callable, List

import numpy as np

import torusgrid as tg

from .base import MinimizerMixin, MuMinimizerMixin

from .relax_base import StressRelaxerBase



class StressRelaxer(
        StressRelaxerBase, 
        MinimizerMixin,
        MuMinimizerMixin):
    """
    Constant mu stress relaxer
    """
    def __init__(self, field: tg.RealField2D, 
            dt: tg.FloatLike, eps: tg.FloatLike, mu: tg.FloatLike, *,
            expansion_rate: tg.FloatLike=1., resize_cycle: int = 31):


        super().__init__(
                field, dt,
                expansion_rate=expansion_rate, 
                resize_cycle=resize_cycle)

        self.init_pfc_variables(eps)
        self.init_mu(mu)

        self.mu = mu
        
        self.initialize_fft()

    def on_size_changed(self):
        self.Kx2 = self.field.kx**2 / self.fx**2
        self.Ky2 = self.field.ky**2 / self.fy**2

        self.K2 = self.Kx2 + self.Ky2
        self.K4 = self.K2**2
        self._kernel = 1-2*self.K2+self.K4 - self.eps

    def get_realspace_steps(self) -> List[Callable[[float], None]]:
        f = self.field
        def step_mu(dt: float):
            f.psi[...] += dt * self.mu

        def step_nonlin(dt: float):
            f.psi[...] /=np.sqrt(1+f.psi**2*dt*2)

        return [step_mu, step_nonlin]

    def step_kernel(self, dt: float):
        f = self.field
        _exp_dt_kernel = np.exp(-dt*self._kernel)
        f.psi_k[...] *= _exp_dt_kernel

    def update_domega_kernels(self):
        self._domega_kernels[:,:,:] = [
                2/self.Lx*self.Kx2*(1-self.K2), 
                2/self.Ly*self.Ky2*(1-self.K2), 
                -2/self.Lx**2*self.Kx2*(3-5*self.Kx2-3*self.Ky2),
                -2/self.Ly**2*self.Ky2*(3-5*self.Ky2-3*self.Kx2),
                2/self.Lx/self.Ly * 2*self.Kx2*self.Ky2
        ]

    def start(self) -> None:
        super().start()    
        self.data['minimizer'] = 'stress-relax'
        self.data['expansion_rate'] = self.expansion_rate

