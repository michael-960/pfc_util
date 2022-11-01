from __future__ import annotations

from typing import List

import numpy as np

import torusgrid as tg
from torusgrid.dynamics import FieldEvolver, SplitStep, Step

from .base import MinimizerMixin, MuMinimizerMixin


class ConstantMuMinimizer(
        SplitStep[tg.RealField2D], 
        MinimizerMixin,
        MuMinimizerMixin):
    """
    PFC constant chemical potential minimizer with split-step FFT
    """
    def __init__(self, 
            field: tg.RealField2D, 
            dt: tg.FloatLike, eps: tg.FloatLike, mu: tg.FloatLike
        ):

        super().__init__(field, dt)
        self.init_pfc_variables(eps)
        self.init_mu(mu)

        self._kernel = 1-2*self.field.k2+self.field.k4 - self.eps

        self._exp_dt_kernel = np.exp(-dt*self._kernel)
        self._mu_dt_half = self.dt * self.mu / 2

        self.initialize_fft()

    def get_realspace_steps(self) -> List[Step]:
        def evolve_mu_(dt: tg.FloatLike):
            self.field.psi[...] += self._mu_dt_half

        def evolve_nonlin_(dt: tg.FloatLike): 
            self.field.psi[...] /= np.sqrt(1+self.field.psi**2*self.dt)

        return [evolve_mu_, evolve_nonlin_]

    def get_kspace_steps(self) -> List[Step]:
        def evolve_k_(dt: tg.FloatLike):
            self.field.psi_k[...] *= self._exp_dt_kernel
        return [evolve_k_]

    def start(self) -> None:
        super().start()
        self.data['minimizer'] = 'mu'
