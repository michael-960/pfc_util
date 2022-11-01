import torusgrid as tg
from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np

from ...core import StateFunctionBase, FieldStateFunction2D



def get_pfc_hooks(*,
    state_function_cls: Type[FieldStateFunction2D],

    display_digits: int,

    extra_display_digits: int = 0,
    extra_digits_color='yellow',

    title_params: List[str] = ['eps', 'mu'],
    display_params: List[str] = ['Lx', 'Ly', 'psibar', 'f', 'F', 'omega', 'Omega'],
    refresh_interval: int = 8,

    detect_slow: Optional[Tuple[str,tg.FloatLike,int]] = ('psibar', 1e-17, 1000)

    ) -> tg.dynamics.EvolverHooks[tg.dynamics.FieldEvolver[tg.RealField2D]]:
    """
    A convenience factor function.
    Build an EvolverHooks object appropriate for PFC simulations
    """

    def float_fmt(x: tg.FloatLike, digits=None):
        if digits is None:
            digits = display_digits
        s = tg.highlight_last_digits(
                tg.float_fmt(x, digits),
                extra_display_digits, highlight=extra_digits_color)
        return s

    def get_title(data: Dict[str,Any]) -> str:
        return '{system} {minimizer} ' + ' '.join(
                [f'{k}={np.format_float_scientific(data[k], precision=display_digits)}'
                 for k in title_params])

    length = max([len(p) for p in display_params]) + 1
    def get_display_info(data: Dict[str, Any]) -> str:
        return '\n'.join([f'{"t":{length}} = {float_fmt(data["age"])}'] + 
                         [f'{k:{length}} = {float_fmt(data[k])}' for k in display_params])


    def monitor(evolver: tg.dynamics.FieldEvolver[tg.RealField2D]):
        environment = {p: evolver.data[p] for p in state_function_cls.environment_params()[0]}
        environment.update({p: evolver.data.get(p, None) for p in state_function_cls.environment_params()[1]})
        sf = state_function_cls.from_field(evolver.field, **environment)
        return sf.data

    hooks = (  tg.dynamics.Display()

             + tg.dynamics.Panel(title=get_title)

             + tg.dynamics.MonitorValues[tg.dynamics.FieldEvolver[tg.RealField2D]](
                 monitor, period=refresh_interval)

             + tg.dynamics.Text(get_display_info, period=refresh_interval)
             )

    if detect_slow is not None:
        target = detect_slow[0]
        precision = detect_slow[1]
        patience = detect_slow[2]
        hooks = hooks + tg.dynamics.DetectSlow(
                 target, 
                 rtol=precision, atol=0, 
                 monotone='ignore',
                 patience=patience, period=refresh_interval)

    hooks = hooks + tg.dynamics.ExitOnInterrupt()
    return hooks

