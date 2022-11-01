import numpy as np

from torusgrid.fields import RealField2D
from torusgrid.dynamics import TemporalEvolver, FieldEvolver
import torusgrid as tg
from warnings import warn

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


