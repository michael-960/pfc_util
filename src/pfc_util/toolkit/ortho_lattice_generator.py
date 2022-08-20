from enum import Enum
from pprint import pprint
import warnings
from typing import List, Tuple, Optional, Union

import numpy as np
from matplotlib import pyplot as plt

from michael960lib.common import deprecated
from torusgrid import fields as fd
from torusgrid.fields import RealField2D
from . import static
from ..core.evolution import PFCMinimizer
from .. import pfc



def generate(na, nb):
    v1 = ((na+nb)*2*np.pi, (na-nb)*2*np.pi/np.sqrt(3))
    v2 = (-v1[1]*np.sqrt(3), v1[0]*np.sqrt(3))

    Ly = np.sqrt(v1[0]**2 + v1[1]**2)
    Lx = np.sqrt(v2[0]**2 + v2[1]**2)

    theta = np.arctan2(v2[1], v2[0])

    return theta, Lx, Ly

def generate_minimal(na: int, nb: int) -> Tuple[float, float, float]:
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

def generate_eps(na: int, nb: int, eps_str: str, 
        NxNy: Union[Tuple[int, int], AutoDimMode]=AutoDimMode.SCALE,
        minimize: Optional[Tuple[float, int]]=None) -> Tuple[float, RealField2D, RealField2D]:
    theta, Lx, Ly = generate_minimal(na, nb)

    f = static.get_relaxed_unit_cell_size(eps_str)
    Lx *= f[0]
    Ly *= f[1]

    print(f'[olg] f={f}')

    sol0 = static.get_relaxed_minimized_coexistent_unit_cell(eps_str, liquid=False)
    liq0 = static.get_relaxed_minimized_coexistent_unit_cell(eps_str, liquid=True)

    print(f'[olg] sol0 psibar = {sol0.psi.mean()}')
    print(f'[olg] sol0 size = ({sol0.Lx}, {sol0.Ly})')
    print(f'[olg] sol0 shape = ({sol0.Nx}, {sol0.Ny})')

    mu = static.get_coexistent_mu_final(eps_str)
    eps = float(eps_str)

    Lx0 = sol0.Lx
    Ly0 = sol0.Ly
    Nx0 = sol0.Nx
    Ny0 = sol0.Ny


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

    
    Xr = np.cos(theta) * sol.X - np.sin(theta) * sol.Y
    Yr = np.sin(theta) * sol.X + np.cos(theta) * sol.Y

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
        sol0: RealField2D, liq0: RealField2D,
        f: List[float]=[1., 1.], 
        NxNy: Union[Tuple[int, int], AutoDimMode]=AutoDimMode.SCALE,
        minimize: Optional[Tuple[float, int]]=None) -> Tuple[float, RealField2D, RealField2D]:

    theta, Lx, Ly = generate_minimal(na, nb)

    Lx *= f[0]
    Ly *= f[1]

    Lx0 = sol0.Lx
    Ly0 = sol0.Ly
    Nx0 = sol0.Nx
    Ny0 = sol0.Ny

    print(sol0.Nx)
    print(sol0.Ny)

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

    Xr = np.cos(theta) * sol.X - np.sin(theta) * sol.Y
    Yr = np.sin(theta) * sol.X + np.cos(theta) * sol.Y

    Ir = (Xr / Lx0 * Nx0).astype(int) % Nx0
    Jr = (Yr / Ly0 * Ny0).astype(int) % Ny0
    
    sol_psi = sol0.psi[Ir,Jr] 
    liq_psi = liq0.psi[Ir,Jr] 
    sol.set_psi(sol_psi)
    liq.set_psi(liq_psi)

    dfmt = '[{system}][{label}][t={age:.2f}] f={f:.10f} F={F:.10f} omega={omega:.10f} Omega={Omega:.10f} psibar={psibar:.10f}'
    if minimize is not None:
        raise NotImplementedError

    return theta, sol, liq



# generate a minimized solid profile for mu=0.195 eps=0.1
@deprecated('use generate_eps instead')
def generate_195(na, nb, Nx, Ny):
    theta, Lx, Ly = generate(na, nb)
    unit = np.load('saved_profiles/unit_cell_0.1950_512.npz')

    Lx0 = unit['Lx']
    Ly0 = unit['Ly']
    Nx0 = unit['psi'].shape[0]
    Ny0 = unit['psi'].shape[1]

    x0 = np.linspace(0, Lx0, Nx0)
    y0 = np.linspace(0, Ly0, Ny0)
    X0, Y0 = np.meshgrid(x0, y0, indexing='ij')
    PSI0 = unit['psi']

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')


    Xr = np.cos(theta) * X - np.sin(theta) * Y    
    Yr = np.sin(theta) * X + np.cos(theta) * Y    

    Ir = (Xr / Lx0 * Nx0).astype(int) % Nx0
    Jr = (Yr / Ly0 * Ny0).astype(int) % Ny0
    
    
    PSI = PSI0[Ir,Jr] 
    

    return X, Y, PSI, Lx, Ly, theta, unit

def harmonic(theta, x, y):
    k1 = (np.cos(theta), np.sin(theta))
    k2 = (np.cos(theta+2*np.pi/3), np.sin(theta+2*np.pi/3))
    k3 = (np.cos(theta+4*np.pi/3), np.sin(theta+4*np.pi/3))
    return np.cos(k1[0]*x+k1[1]*y) + np.cos(k2[0]*x+k2[1]*y) + np.cos(k3[0]*x+k3[1]*y)


