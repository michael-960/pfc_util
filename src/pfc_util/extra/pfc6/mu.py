from torusgrid import RealField2D
import numpy as np

from ... import core

from .base import MinimizerMixin


class ConstantMuMinimizer(core.ConstantMuMinimizer, MinimizerMixin):

    def __init__(self, 
            field: RealField2D, dt: float, 
            eps: float, alpha: float, mu: float):
        super().__init__(field, dt, eps, mu)
        self.init_pfc6_variables(alpha)

        self.alpha = alpha
        self.info['label'] = self.label = f'mu6 eps={eps:.5f} mu={mu:.5f} alpha={alpha:.5f} dt={dt:.5f}'

        self._kernel = 1 - 2*field.K2 + field.K4 +\
                alpha*(field.K4*field.K2 -2*field.K4 + field.K2)- eps
       
        self._exp_dt_kernel = np.exp(-dt*self._kernel)
        self._mu_dt_half = dt * mu / 2

        self.initialize_fft()
