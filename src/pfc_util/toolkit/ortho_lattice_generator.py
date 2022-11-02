from enum import Enum
from typing import List, Tuple, Optional, Union
import numpy as np

import torusgrid as tg
from . import static
from .. import pfc

import rich

import warnings
warnings.warn('ortho_lattice_generator is deprecated, use rotated instead', DeprecationWarning)

console = rich.get_console()

def generate(na, nb):
    """
    Calculate angle and system size
    """
    v1 = ((na+nb)*2*np.pi, (na-nb)*2*np.pi/np.sqrt(3))
    v2 = (-v1[1]*np.sqrt(3), v1[0]*np.sqrt(3))

    Ly = np.sqrt(v1[0]**2 + v1[1]**2)
    Lx = np.sqrt(v2[0]**2 + v2[1]**2)

    theta = np.arctan2(v2[1], v2[0])

    return theta, Lx, Ly


def generate_minimal(na: int, nb: int) -> Tuple[float, float, float]:
    """
    Calculate angle and system size, with GCDs removed
    """
    # diameter
    D = np.pi * 4 / np.sqrt(3)

    # lattice basis
    a = D*np.array((np.sqrt(3)/2, 1/2))
    b = D*np.array((np.sqrt(3)/2, -1/2))

    Pa = na + 2*nb
    Pb = -2*na - nb

    p0 = np.gcd(Pa, Pb)
    pa = Pa // p0
    pb = Pb // p0
    
    v1 = na*a + nb*b
    v2 = pa*a + pb*b

    theta = np.arctan2(v2[1], v2[0])

    Lx = np.sqrt(v2[0]**2+v2[1]**2)
    Ly = np.sqrt(v1[0]**2+v1[1]**2)

    return theta, Lx, Ly


class AutoDimMode(Enum):
    SCALE = 0
    SNAP = 1


def generate_eps(
        na: int, nb: int, eps_str: str, 
        NxNy: Union[Tuple[int, int], AutoDimMode]=AutoDimMode.SCALE,
        minimize: Optional[Tuple[float, int]]=None
    ) -> Tuple[tg.FloatLike, tg.RealField2D, tg.RealField2D]:
    '''Deprecated'''

    theta, Lx, Ly = generate_minimal(na, nb)

    f = static.get_relaxed_unit_cell_size(eps_str)
    Lx *= f[0]
    Ly *= f[1]

    print(f'[olg] f={f}')

    sol0 = static.get_relaxed_minimized_coexistent_unit_cell(eps_str, liquid=False)
    liq0 = static.get_relaxed_minimized_coexistent_unit_cell(eps_str, liquid=True)

    print(f'[olg] sol0 psibar = {sol0.psi.mean()}')
    print(f'[olg] sol0 size = ({sol0.lx}, {sol0.ly})')
    print(f'[olg] sol0 shape = ({sol0.nx}, {sol0.ny})')

    mu = static.get_coexistent_mu_final(eps_str)
    eps = float(eps_str)

    Lx0 = sol0.lx
    Ly0 = sol0.ly
    Nx0 = sol0.nx
    Ny0 = sol0.ny


    _density = np.sqrt(Nx0*Ny0/Lx0/Ly0)

    try:
        if NxNy is AutoDimMode.SCALE:
            Nx = np.rint(Lx0*_density)
            Ny = np.rint(Ly0*_density)

        elif NxNy is AutoDimMode.SNAP:
            Nx = 2 ** int(np.rint(np.log2(Lx*_density)))
            Ny = 2 ** int(np.rint(np.log2(Ly*_density)))
        
        else:
            assert isinstance(NxNy, tuple)
            assert len(NxNy) == 2
            assert type(NxNy[0]) is type(NxNy[1]) is int
            Nx = NxNy[0]
            Ny = NxNy[1]
    except AssertionError:
        raise ValueError('NxNy must be a tuple of 2 integers (Nx, Ny) or an instance of AutoDimMode')


    sol = fd.RealField2D(Lx, Ly, Nx, Ny)
    liq = fd.RealField2D(Lx, Ly, Nx, Ny)

    
    Xr = np.cos(theta) * sol.x - np.sin(theta) * sol.y
    Yr = np.sin(theta) * sol.x + np.cos(theta) * sol.y

    Ir = (Xr / Lx0 * Nx0).astype(int) % Nx0
    Jr = (Yr / Ly0 * Ny0).astype(int) % Ny0

    
    sol_psi = sol0.psi[Ir,Jr] 
    liq_psi = liq0.psi[Ir,Jr] 
    sol.set_psi(sol_psi)
    liq.set_psi(liq_psi)

    print(f'[olg] sol psibar = {sol.psi.mean()}')
    print(f'[olg] sol size = ({Lx}, {Ly})')
    print(f'[olg] sol shape = ({Nx}, {Ny})')


    dfmt = '[{system}][{label}][t={age:.2f}] f={f:.10f} F={F:.10f} omega={omega:.10f} Omega={Omega:.10f} psibar={psibar:.10f}'
    if minimize is not None:
        print(f'[olg] running mu-minimzation')
        model_sol = pfc.PFC(sol)
        model_sol.evolve(minimizer='mu', dt=minimize[0], eps=eps, mu=mu, N_epochs=minimize[1], display_format=dfmt)
        model_liq = pfc.PFC(liq)
        model_liq.evolve(minimizer='mu', dt=minimize[0], eps=eps, mu=mu, N_epochs=minimize[1], display_format=dfmt)

    print(f'[olg] sol psibar = {sol.psi.mean()}')
    print(f'[olg] sol size = ({Lx}, {Ly})')
    print(f'[olg] sol shape = ({Nx}, {Ny})')

    return theta, sol, liq


def generate_from_field(
        na: int, nb: int, 
        sol0: tg.RealField2D, liq0: tg.RealField2D, *,
        f: List[float]=[1., 1.], 
        NxNy: Union[Tuple[int, int], AutoDimMode]=AutoDimMode.SCALE,
) -> Tuple[tg.FloatLike, tg.RealField2D, tg.RealField2D]:
    """
      
    """

    '''Calculate angle and system size'''
    theta, Lx, Ly = generate_minimal(na, nb)

    Lx *= f[0]
    Ly *= f[1]

    Lx0 = sol0.lx
    Ly0 = sol0.ly
    Nx0 = sol0.nx
    Ny0 = sol0.ny

    console.log('lattice generator')
    console.log(f'solid resolution: ({sol0.nx}, {sol0.ny})')
    console.log(f'solid size: {sol0.lx}, {sol0.ly})')

    '''solid number of points per unit volume'''
    _density = np.sqrt(Nx0*Ny0/Lx0/Ly0)


    try:
        if NxNy is AutoDimMode.SCALE:
            '''calculate new shape by scaling'''
            Nx = np.rint(Lx0*_density)
            Ny = np.rint(Ly0*_density)

        elif NxNy is AutoDimMode.SNAP:
            '''calculate new shape by snapping to the closest powers of 2'''

            Nx = 2 ** int(np.rint(np.log2(Lx*_density)))
            Ny = 2 ** int(np.rint(np.log2(Ly*_density)))
        
        else:
            '''directly specified shape'''
            assert isinstance(NxNy, tuple)
            assert len(NxNy) == 2
            assert type(NxNy[0]) is type(NxNy[1]) is int
            Nx = NxNy[0]
            Ny = NxNy[1]

    except AssertionError:
        raise ValueError('NxNy must be a tuple of 2 integers (Nx, Ny) or an instance of AutoDimMode')

    '''The new fields'''
    sol = tg.RealField2D(Lx, Ly, Nx, Ny)
    liq = tg.RealField2D(Lx, Ly, Nx, Ny)

    '''Rotated coordinates'''
    Xr = np.cos(theta) * sol.x - np.sin(theta) * sol.y
    Yr = np.sin(theta) * sol.x + np.cos(theta) * sol.y

    Ir = np.rint(Xr / Lx0 * Nx0).astype(int) % Nx0
    Jr = np.rint(Yr / Ly0 * Ny0).astype(int) % Ny0
    
    '''Rotated fields'''
    sol.psi[...] = sol0.psi[Ir,Jr]
    liq.psi[...] = liq0.psi[Ir,Jr]

    return theta, sol, liq


def harmonic(theta, x, y):
    k1 = (np.cos(theta), np.sin(theta))
    k2 = (np.cos(theta+2*np.pi/3), np.sin(theta+2*np.pi/3))
    k3 = (np.cos(theta+4*np.pi/3), np.sin(theta+4*np.pi/3))
    return np.cos(k1[0]*x+k1[1]*y) + np.cos(k2[0]*x+k2[1]*y) + np.cos(k3[0]*x+k3[1]*y)


