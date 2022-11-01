from abc import abstractmethod
from typing import Tuple, final
import numpy as np

from torusgrid.fields import RealField2D
from torusgrid.dynamics import SecondOrderRK4, FieldEvolver
import torusgrid as tg
from warnings import warn
import numpy.typing as npt

from ..base import FreeEnergyFunctional


class MinimizerMixin(FieldEvolver[RealField2D]):
    """
    A mix-in class for PFC minimizers.
    """
    def init_pfc_variables(self, eps: tg.FloatLike):
        """
        Setup variables related to PFC
        """
        self.fef = FreeEnergyFunctional(eps)
        self.eps = eps

    @property
    def info(self):
        """
        using .info is deprecated, use .data directly instead
        """
        warn('Using .info is deprecated, use .data directly instead')
        return self.data

    def start(self) -> None:
        super().start()
        self.data['system'] = 'pfc'
        self.data['eps'] = self.eps


class MuMinimizerMixin(tg.dynamics.Evolver[RealField2D]):
    """
    A mix-in class for constant chemical potential minimizers.
    """
    def init_mu(self, mu: tg.FloatLike):
        self.mu = mu

    def start(self):
        super().start()
        self.data['mu'] = self.mu


class NonlocalConservedRK4Base(SecondOrderRK4[RealField2D]):
    """
    A base class for second-order RK4 with regularizer and inertia

    Subclasses must implement:
        - derivative()
    """
    def __init__(self, 
            field: RealField2D, 
            dt: tg.FloatLike, *,
            k_regularizer: tg.FloatLike, 
            inertia: tg.FloatLike):

        super().__init__(field, dt)

        self.R = k_regularizer
        self.inertia = inertia

        self._deriv = RealField2D(
                field.lx, field.ly, field.nx, field.ny,
                precision=field.precision)

        self._deriv.initialize_fft()

    @final
    def psi_dot(self) -> Tuple[npt.NDArray, npt.NDArray]:
        self._deriv.psi[...] = -self.derivative(self.grid_tmp)
        self._deriv.fft()
        self._deriv.psi_k[...] *= np.exp(-self.R*self._deriv.k2)
        self._deriv.ifft()
        F = self._deriv.psi - np.mean(self._deriv.psi)
        return self.dgrid_tmp.psi*self.inertia, -(self.dgrid_tmp.psi - F)

    @abstractmethod
    def derivative(self, f: RealField2D) -> npt.NDArray:
        """
        Return the derivative of the free energy functional w.r.t. f
        """

    def start(self) -> None:
        super().start()
        self.data['M'] = self.inertia
        self.data['R'] = self.R


