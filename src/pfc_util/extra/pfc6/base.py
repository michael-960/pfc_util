from torusgrid import RealField2D
from ...utils.fft import rfft2, irfft2
import numpy as np

from ... import core


class FreeEnergyFunctional(core.FreeEnergyFunctional):
    def __init__(self, eps: float, alpha: float):
        self.eps = eps
        self.alpha = alpha

    def free_energy_density(self, field: RealField2D):
        kernel = (1-field.K2)**2 +\
                self.alpha*(field.K4*field.K2 - 2*field.K4 + field.K2)
        psi_k = rfft2(field.psi)
        psi_k_o = kernel * psi_k
        f = 1/2 * field.psi * irfft2(psi_k_o) + field.psi**4/4 - self.eps/2 * field.psi**2
        return np.real(f)

    def derivative(self, field):
        field.fft()
        linear_term = ((1-field.K2)**2 + self.alpha*field.K2*(1-field.K2)**2 - self.eps) * field.psi_k
        field.ifft()

        local_mu = irfft2(linear_term) + field.psi**3
        return local_mu


class MinimizerMixin(core.MinimizerMixin):
    def init_pfc6_variables(self, alpha: float):
        self.alpha = alpha
        self.info['system'] = 'pfc6'
        self.info['alpha'] = alpha
        self.fef = FreeEnergyFunctional(self.eps, alpha)
