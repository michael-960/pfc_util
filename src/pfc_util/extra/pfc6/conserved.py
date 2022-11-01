from typing import final
import torusgrid as tg
from ... import core
from .base import FreeEnergyFunctional
from .base import MinimizerMixin
import numpy.typing as npt


@final
class NonlocalConservedRK4(
        core.NonlocalConservedRK4Base,
        MinimizerMixin):
    """
    PFC6 Nonlocal conserved dynamics with RK4
    """
    def __init__(
        self, 
        field: tg.RealField2D, 
        dt: tg.FloatLike, 
        eps: tg.FloatLike, alpha: tg.FloatLike, beta: tg.FloatLike, *,
        k_regularizer: tg.FloatLike=0.1,
        inertia: tg.FloatLike=100
    ):
        super().__init__(field, dt,
                         k_regularizer=k_regularizer, 
                         inertia=inertia)
        self.init_pfc6_variables(eps, alpha, beta)

    def derivative(self, f: tg.RealField2D) -> npt.NDArray:
        return self.fef.derivative(f)

    def start(self) -> None:
        super().start()
        self.data['minimizer'] = 'nonlocal-rk4'



