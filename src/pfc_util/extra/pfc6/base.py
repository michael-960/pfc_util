from __future__ import annotations
from typing import List, Tuple, final
import torusgrid as tg
import numpy as np
from ...utils.fft import rfft2, irfft2
from ... import core


class FreeEnergyFunctional(core.FreeEnergyFunctionalBase[tg.RealField2D]):
    """
    PFC6 free energy functional
    """
    def __init__(self, eps: tg.FloatLike, alpha: tg.FloatLike, beta: tg.FloatLike):
        """
        Parameters:
            eps: PFC epsilon
            alpha: 6th order derivative coefficient
            beta: psi^6/6 coefficient
        """
        self.eps = eps
        self.alpha = alpha
        self.beta = beta

    def free_energy_density(self, field: tg.RealField):
        kernel = (1-field.k2)**2 +\
                self.alpha*(field.k4*field.k2 - 2*field.k4 + field.k2)

        psi_k = rfft2(field.psi)
        psi_k_o = kernel * psi_k
        f = 1/2 * field.psi * irfft2(psi_k_o) + field.psi**4/4 - self.eps/2 * field.psi**2 + self.beta * field.psi**6 / 6
        return np.real(f)

    def derivative(self, field: tg.RealField):
        field.fft()
        linear_term = ((1-field.k2)**2 + self.alpha*field.k2*(1-field.k2)**2 - self.eps) * field.psi_k
        field.ifft()
        local_mu = irfft2(linear_term) + field.psi**3 + self.beta * field.psi**5
        return local_mu


@final
class StateFunction(core.FieldStateFunction2D):
    @classmethod
    def free_energy_functional(
        cls, *, 
        eps: tg.FloatLike, alpha: tg.FloatLike, beta: tg.FloatLike
    ) -> core.FreeEnergyFunctionalBase[tg.RealField2D]:
        return FreeEnergyFunctional(eps, alpha, beta)

    @staticmethod
    def environment_params() -> Tuple[List[str], List[str]]:
        return ['eps', 'alpha', 'beta'], ['mu']

class MinimizerMixin(core.MinimizerMixin):
    """
    Mixin for PFC6 minimizers

    Subclasses should call init_pfc6_variables instead of init_pfc_variables
    """
    def init_pfc6_variables(self, 
                            eps: tg.FloatLike,
                            alpha: tg.FloatLike, beta: tg.FloatLike):

        super().init_pfc_variables(eps)
        self.fef = FreeEnergyFunctional(self.eps, alpha, beta)
        self.alpha = alpha
        self.beta = beta

    def start(self) -> None:
        super().start()
        self.data['system'] = 'pfc6'
        self.data['alpha'] = self.alpha
        self.data['beta'] = self.beta


