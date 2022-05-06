import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
from . import static
from torusgrid import fields as fd
import warnings

from michael960lib.common import deprecated

def generate(na, nb):
    v1 = ((na+nb)*2*np.pi, (na-nb)*2*np.pi/np.sqrt(3))
    v2 = (-v1[1]*np.sqrt(3), v1[0]*np.sqrt(3))

    Ly = np.sqrt(v1[0]**2 + v1[1]**2)
    Lx = np.sqrt(v2[0]**2 + v2[1]**2)

    theta = np.arctan2(v2[1], v2[0])

    return theta, Lx, Ly

def generate_minimal(na: int, nb: int):
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


def generate_eps(na: int, nb: int, Nx: int, Ny: int, eps_str: str):
    theta, Lx, Ly = generate_minimal(na, nb)

    unit0 = static.get_relaxed_minimized_coexistent_unit_cell(eps_str)
    Lx0 = unit0.Lx
    Ly0 = unit0.Ly
    Nx0 = unit0.Nx
    Ny0 = unit0.Ny

    unitr = fd.RealField2D(Lx, Ly, Nx, Ny)

    Xr = np.cos(theta) * unitr.X - np.sin(theta) * unitr.Y
    Yr = np.sin(theta) * unitr.X + np.cos(theta) * unitr.Y

    Ir = (Xr / Lx0 * Nx0).astype(int) % Nx0
    Jr = (Yr / Ly0 * Ny0).astype(int) % Ny0
    
    psi = unit0.psi[Ir,Jr] 
    unitr.set_psi(psi)
    return theta, unitr


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


