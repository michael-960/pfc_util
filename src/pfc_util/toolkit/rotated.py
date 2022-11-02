from enum import Enum
from typing import List, Tuple, Union
import numpy as np
import torusgrid as tg

import rich

console = rich.get_console()

class AutoShape(Enum):
    SCALE = 0
    SNAP = 1

SNAP = AutoShape.SNAP
SCALE = AutoShape.SCALE


def calculated_rotated(
    na: int, nb: int, *,
    minimal: bool = True,
    precision: tg.PrecisionLike = 'double'
) -> Tuple[tg.FloatLike,tg.FloatLike,tg.FloatLike]:
    """
    Calculate rotation angle and system size in which multiple unit
    cells can be placed subject to the periodic boundary condition.

    A (unrelaxed) unit cell is assumed to have size (4pi, 4pi/sqrt 3).
    In general, a relaxed unit cell will have size f * (4pi, 4pi/sqrt 3), where
    f is a scaling factor close to 1.

    Parameters:

        (na, nb): A vector in the lattice basis specifying the direction of rotation
        minimal: Whether to reduce the length of the orthogonal direction by
                 the common factor

    Return:
        (theta, gx, gy)

        theta - rotation angle
        gx, gy - size after rotation **DIVIDED BY THE ATOM DIAMETER**.
                 The unrelaxed atom diameter is defined as 4pi/sqrt3.

    """
    # diameter
    dtype = tg.get_real_dtype(precision)
    sqrt3 = np.sqrt(dtype(3))
    D = tg.pi(precision) * 4 / sqrt3

    # lattice basis
    a = D * np.array((sqrt3/2, dtype(1)/2))
    b = D * np.array((sqrt3/2, -dtype(1)/2))

    pa = na + 2*nb
    pb = -2*na - nb

    if minimal:
        p0 = np.gcd(pa, pb)
        pa = pa // p0
        pb = pb // p0
    
    v1 = na*a + nb*b
    v2 = pa*a + pb*b

    theta = np.arctan2(v2[1], v2[0])

    gx = np.sqrt(v2[0]**2+v2[1]**2) / D
    gy = np.sqrt(v1[0]**2+v1[1]**2) / D

    return theta, gx, gy


class UnitCellRotator:
    def __init__(
        self, 
        na: int, nb: int,
        minimal: bool = True):

        self.na = na
        self.nb = nb
        self.minimal = minimal

        self._theta, self._gx, self._gy = calculated_rotated(
                na, nb,
                minimal=minimal,
                precision='longdouble')

    @property
    def theta(self) -> np.longdouble:
        return self._theta # type: ignore

    @property
    def gx(self) -> np.longdouble:
        return self._gx # type: ignore

    @property
    def gy(self) -> np.longdouble:
        return self._gy # type: ignore

    def __call__(
        self, 
        unit_cell: tg.RealField2D, 
        shape: Union[Tuple[int, int], AutoShape]=AutoShape.SCALE, *,
        verbose: bool = True
    ) -> tg.RealField2D:
        """
        Rotate and extend unit cells.

        Important note: the unit cell is assumed 
                        to have size (4pi, 4pi/sqrt3) * f with f being a scaling factor close to 1.

        Return:
            A tuple of (float, field) consisting of :
                - the rotation angle
                - the rotated field
        """

        '''Calculate angle and system size'''

        dtype = tg.get_real_dtype(unit_cell.precision)

        diameter = unit_cell.ly
        theta = dtype(self.theta)
        lx = dtype(self.gx * diameter)
        ly = dtype(self.gy * diameter)

        lx0 = unit_cell.lx
        ly0 = unit_cell.ly
        nx0 = unit_cell.nx
        ny0 = unit_cell.ny

        if verbose:
            console.log(f'unit cell shape: ({nx0}, {ny0})')
            console.log(f'unit cell size: {lx0}, {ly0})')

        '''solid number of points per unit volume'''
        _density = np.sqrt(nx0*ny0/lx0/ly0)

        try:
            if shape is AutoShape.SCALE:
                '''calculate new shape by scaling'''
                nx = int(np.rint(lx*_density))
                ny = int(np.rint(ly*_density))

            elif shape is AutoShape.SNAP:
                '''calculate new shape by snapping to the closest powers of 2'''
                nx = 2 ** int(np.rint(np.log2(lx*_density)))
                ny = 2 ** int(np.rint(np.log2(ly*_density)))
            
            else:
                '''directly specified shape'''
                assert isinstance(shape, tuple)
                assert len(shape) == 2
                assert type(shape[0]) is type(shape[1]) is int
                nx = shape[0]
                ny = shape[1]

        except AssertionError:
            raise ValueError('shape must be a tuple of 2 integers (Nx, Ny) or an instance of AutoDimMode')

        '''The new fields'''
        f = tg.RealField2D(lx, ly, nx, ny, precision=unit_cell.precision)

        '''Rotated coordinates'''
        Xr = np.cos(theta) * f.x - np.sin(theta) * f.y
        Yr = np.sin(theta) * f.x + np.cos(theta) * f.y

        Ir = np.rint(Xr / lx0 * nx0).astype(int) % nx0
        Jr = np.rint(Yr / ly0 * ny0).astype(int) % ny0
        
        '''Rotated fields'''
        f.psi[...] = unit_cell.psi[Ir,Jr]
        return f
        


def rotate_unit_cell(
        na: int, nb: int, 
        uc: tg.RealField2D, *,
        scale_factors: List[tg.FloatLike]=[1., 1.], 
        shape: Union[Tuple[int, int], AutoShape]=AutoShape.SCALE,
        minimal: bool = True,
        verbose: bool = True
) -> Tuple[tg.FloatLike, tg.RealField2D]:
    """
    Rotate and extend unit cells         

    Return:
        A tuple of (float, field) consisting of :
            - the rotation angle
            - the rotated field
    """

    '''Calculate angle and system size'''

    theta, Lx, Ly = calculated_rotated(na, nb,
                                       minimal=minimal,
                                       precision=uc.precision)

    Lx *= scale_factors[0]
    Ly *= scale_factors[1]

    Lx0 = uc.lx
    Ly0 = uc.ly
    Nx0 = uc.nx
    Ny0 = uc.ny

    if verbose:
        console.log(f'unit cell shape: ({uc.nx}, {uc.ny})')
        console.log(f'unit cell size: {uc.lx}, {uc.ly})')

    '''solid number of points per unit volume'''
    _density = np.sqrt(Nx0*Ny0/Lx0/Ly0)

    try:
        if shape is AutoShape.SCALE:
            '''calculate new shape by scaling'''
            Nx = np.rint(Lx0*_density)
            Ny = np.rint(Ly0*_density)

        elif shape is AutoShape.SNAP:
            '''calculate new shape by snapping to the closest powers of 2'''

            Nx = 2 ** int(np.rint(np.log2(Lx*_density)))
            Ny = 2 ** int(np.rint(np.log2(Ly*_density)))
        
        else:
            '''directly specified shape'''
            assert isinstance(shape, tuple)
            assert len(shape) == 2
            assert type(shape[0]) is type(shape[1]) is int
            Nx = shape[0]
            Ny = shape[1]

    except AssertionError:
        raise ValueError('shape must be a tuple of 2 integers (Nx, Ny) or an instance of AutoDimMode')

    '''The new fields'''
    f = tg.RealField2D(Lx, Ly, Nx, Ny)

    '''Rotated coordinates'''
    Xr = np.cos(theta) * f.x - np.sin(theta) * f.y
    Yr = np.sin(theta) * f.x + np.cos(theta) * f.y

    Ir = np.rint(Xr / Lx0 * Nx0).astype(int) % Nx0
    Jr = np.rint(Yr / Ly0 * Ny0).astype(int) % Ny0
    
    '''Rotated fields'''
    f.psi[...] = uc.psi[Ir,Jr]

    return theta, uc

