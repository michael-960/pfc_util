from typing import Optional
import torusgrid as tg
import rich
from torusgrid.dynamics import EvolverHooks, FieldEvolver

from .base import MinimizerSupplier


def elongate_interface(
    ifc: tg.RealField2D, 
    delta_sol: tg.RealField2D, 
    delta_liq: tg.RealField2D,
):
    """
    Elongate the interface with delta_sol and delta_liq
    
        IIII -> LIISSIIL

    where

        IIII: original interface
        S: delta_sol
        L: delta_liq

    """

    left = tg.crop(ifc, 0, 0, ifc.nx//2)

    right = tg.crop(ifc, 0, ifc.nx//2, ifc.nx)

    ifc_elongated = tg.concat(
        delta_liq, left, delta_sol, delta_sol, right, delta_liq,
        axis=0)

    return ifc_elongated



def evolve_and_elongate_interface(
    ifc: tg.RealField2D, delta_sol: tg.RealField2D, 
    delta_liq: tg.RealField2D, *,

    minimizer_supplier: MinimizerSupplier,
    n_steps: int = 31,
    hooks: Optional[
            EvolverHooks[FieldEvolver[tg.RealField2D]]
           ]=None,

    verbose: bool = False
) -> tg.RealField2D:
    r"""

    Evolve interface -> elongate interface

    :param ifc,delta_sol,delta_liq: 2D real fields of the same height & vertical shape

    :param minimizer_supplier: A function that returns a constant :math:`\mu` minimizer given a field and :math:`\mu`
    
    """

    console = rich.get_console()

    '''Make sure all the fields have the same Y dimensions'''
    try:
        assert delta_sol.ny == delta_liq.ny == ifc.ny
        assert delta_sol.ly == delta_liq.ly == ifc.ly
    except AssertionError:
        raise ValueError('delta_sol, delta_liq, and ifc must have the same Y dimensions')


    '''Evolve interface'''
    if verbose:
        console.log(f'Lx={ifc.lx}, Ly={ifc.ly}')
        console.log(f'evolving interface')

    minim = minimizer_supplier(ifc)
    minim.run(n_steps, hooks=hooks)

    '''Elongate interface'''
    if verbose:
        console.log('elongating interface')

    ifc_elongated = elongate_interface(ifc, delta_sol, delta_liq)

    if verbose:
        console.log(f'Lx={ifc_elongated.lx}, Ly={ifc_elongated.ly}')

    return ifc_elongated

