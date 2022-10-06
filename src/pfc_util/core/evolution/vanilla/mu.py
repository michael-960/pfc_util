from __future__ import annotations

from typing import List, Type

import numpy as np

from torusgrid.dynamics import SplitStep, Step

from torusgrid.fields import RealField2D

from ..base import MuMinimizerMixin


class ConstantMuMinimizer(
        SplitStep[RealField2D], MuMinimizerMixin
    ):
    '''
    PFC constant chemical potential minimizer with split-step FFT
    '''
    def __init__(self, 
            field: RealField2D, 
            dt: float, eps: float, mu: float):

        super().__init__(field, dt)
        self.init_pfc_variables(eps, mu)

        self.info['minimizer'] = 'mu'
        self.info['mu'] = self.mu = mu
        self.info['label'] = self.label = f'mu eps={eps:.5f} mu={mu:.5f} dt={dt:.5f}'

        self._kernel = 1-2*self.field.K2+self.field.K4 - self.eps

        self._exp_dt_kernel = np.exp(-dt*self._kernel)
        self._mu_dt_half = self.dt * self.mu / 2

        self.initialize_fft()

    def get_realspace_steps(self) -> List[Step]:
        def evolve_mu_(dt: float):
            self.field.psi[:] += self._mu_dt_half

        def evolve_nonlin_(dt: float): 
            self.field.psi /= np.sqrt(1+self.field.psi**2*self.dt)

        return [evolve_mu_, evolve_nonlin_]

    def get_kspace_steps(self) -> List[Step]:
        def evolve_k_(dt: float):
            self.field.psi_k *= self._exp_dt_kernel
        return [evolve_k_]



# Deprecated
ConstantChemicalPotentialMinimizer = ConstantMuMinimizer



