from typing import final
import torusgrid as tg
from ... import core
from .base import FreeEnergyFunctional
from .base import MinimizerMixin
import numpy.typing as npt


@final
class NonlocalConservedRK4(
    core.NonlocalConservedRK4Base,
    MinimizerMixin
):
    r"""
    PFC6 RK4 nonlocal conserved dynamics with inertia

    The evolution equations are

        .. math::

            \dot\psi = M \phi

        .. math::

            \dot\phi = - \left(\phi + \frac{\delta F}{\delta \psi}\Big\vert_{k\neq 0}\right)
    """

    def __init__(
        self, 
        field: tg.RealField2D, 
        dt: tg.FloatLike, 
        eps: tg.FloatLike, alpha: tg.FloatLike, beta: tg.FloatLike, *,
        k_regularizer: tg.FloatLike=0.1,
        inertia: tg.FloatLike=100
    ):
        r"""
        :param field: the PFC field to be minimized
        :param dt: time step
        :param eps: PFC :math:`\epsilon`
        :param alpha: PFC6 :math:`\alpha`
        :param beta: PFC6 :math:`\beta`

        :param k_regularizer: k-space regulator to suppress high frequency modes from blowing up
        :param inertia: coefficient of 1st time derivative
        """

        super().__init__(field, dt,
                         k_regularizer=k_regularizer, 
                         inertia=inertia)
        self.init_pfc6_variables(eps, alpha, beta)

    def derivative(self, f: tg.RealField2D) -> npt.NDArray:
        return self.fef.derivative(f)

    def start(self) -> None:
        super().start()
        self.data['minimizer'] = 'nonlocal-rk4'



