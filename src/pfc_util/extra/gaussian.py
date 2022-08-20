import numpy as np
from scipy.fft import rfft2, irfft2

from michael960lib.common import overrides, deprecated, experimental
from torusgrid.fields import RealField2D
from torusgrid.dynamics import FreeEnergyFunctional2D, NoiseGenerator2D
from ..core.base import PFCFreeEnergyFunctional
from ..core.evolution import ConstantChemicalPotentialMinimizer


class GaussianPFCFucntional(PFCFreeEnergyFunctional):
    def __init__(self, eps):
        self.eps = eps

    @overrides(FreeEnergyFunctional2D)
    def free_energy_density(self, field: RealField2D):
        kernel = np.exp(1-field.K2) + field.K2 + 1 - np.e
        psi_k = rfft2(field.psi)
        psi_k_o = kernel * psi_k
        f = 1/2 * field.psi * irfft2(psi_k_o) + field.psi**4/4 - self.eps/2 * field.psi**2
        return np.real(f)

    @overrides(FreeEnergyFunctional2D)
    def derivative(self, field: RealField2D):
        kernel = np.exp(1-field.K2) + field.K2 + 1 - np.e
        field.fft2()
        linear_term = (kernel - self.eps) * field.psi_k
        field.ifft2()
        local_mu = irfft2(linear_term) + field.psi**3
        return local_mu



class GaussianPFCMinimizer(ConstantChemicalPotentialMinimizer):
    def __init__(self, field: RealField2D, dt: float, eps: float, mu: float, noise_generator:NoiseGenerator2D=None):
        super().__init__(field, dt, eps, mu, noise_generator)
        self.info['system'] = 'gauss'
        self.fef = GaussianPFCFucntional(eps)

        self._kernel = np.exp(1-field.K2) + field.K2 + 1 - np.e - self.eps
        self._exp_dt_kernel = np.exp(-dt*self._kernel)
        self._mu_dt_half = self.dt * self.mu / 2




