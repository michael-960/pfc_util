from typing import List
from torusgrid import FloatingPointPrecision, RealField2D, FloatLike
import torusgrid as tg
import numpy as np

from ... import core

from .base import MinimizerMixin


class ConstantMuMinimizer_(core.ConstantMuMinimizer, MinimizerMixin):
    def __init__(self, 
            field: RealField2D, 
            dt: FloatLike, eps: FloatLike, 
            alpha: FloatLike, 
            beta: FloatLike,
            mu: FloatLike):

        super().__init__(field, dt, eps, mu)
        self.init_pfc_variables(eps, alpha, beta)

        self.info['label'] = self.label = f'mu6 eps={eps:.5f} mu={mu:.5f} alpha={alpha:.5f} dt={dt:.5f}'

        self._kernel = 1 - 2*field.k2 + field.k4 +\
                alpha*(field.k4*field.k2 -2*field.k4 + field.k2) - eps
       
        self._exp_dt_kernel = np.exp(-dt*self._kernel)
        self._mu_dt_half = dt * mu / 2

        self.initialize_fft()


class ConstantMuMinimizer(
        tg.dynamics.SplitStep[tg.RealField2D],
        MinimizerMixin,
        core.MuMinimizerMixin):

    def __init__(self, 
            field: RealField2D, 
            dt: FloatLike, 
            eps: FloatLike, 
            alpha: FloatLike, 
            beta: FloatLike,
            mu: FloatLike):

        super().__init__(field, dt)
        self.init_pfc6_variables(eps, alpha, beta)
        self.init_mu(mu)

        self._dtype = tg.get_real_dtype(field.precision)

        self._kernel = 1 - 2*field.k2 + field.k4 +\
                alpha*(field.k4*field.k2 -2*field.k4 + field.k2) - eps
       
        self._exp_dt_kernel = np.exp(-dt*self._kernel)
        self._mu_dt_half = dt * mu / 2

        self.initialize_fft()

    def start(self) -> None:
        super().start()
        self.data['minimizer'] = 'mu'

    def get_realspace_steps(self) -> List[tg.dynamics.Step]:
        def evolve_mu_(dt: tg.FloatLike):
            self.field.psi[...] += self._mu_dt_half

        def evolve_nonlin_4(dt: tg.FloatLike): 
            self.field.psi[...] /= np.sqrt(1+self.field.psi**2*self.dt)
            
        def evolve_nonlin_6(dt: tg.FloatLike):
            self.field.psi[...] /= np.power(
                    1+2*self.beta*self.field.psi**4*self.dt,
                    1 / self._dtype('4'))

        return [evolve_mu_, evolve_nonlin_4, evolve_nonlin_6]

    def get_kspace_steps(self) -> List[tg.dynamics.Step]:
        def evolve_k_(dt: tg.FloatLike):
            self.field.psi_k[...] *= self._exp_dt_kernel
        return [evolve_k_]

