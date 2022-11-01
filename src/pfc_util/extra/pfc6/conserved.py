import torusgrid as tg
from ... import core
from .base import FreeEnergyFunctional
from .base import MinimizerMixin


class NonlocalConservedRK4(
        core.NonlocalConservedRK4,
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
        inertia: tg.FloatLike=100):
        
        raise NotImplementedError

        super().__init__(field, dt, eps, 
                         k_regularizer=k_regularizer, inertia=inertia)

        self.init_pfc6_variables(eps, alpha, beta)
        self.alpha = alpha
        self.info['system'] = 'pfc6'
        self.info['alpha'] = alpha
        self.info['label'] = self.label = f'pfc6 nonlocal-rk4 eps={eps} alpha={alpha} dt={dt} R={k_regularizer} M={inertia}'
        self.fef = FreeEnergyFunctional(eps, alpha, beta)


