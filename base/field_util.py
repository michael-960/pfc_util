from .field import ComplexField2D, RealField2D
from matplotlib import pyplot as plt
import numpy as np
from typing import List


def plot(fields: List[ComplexField2D], cmap='jet', show=True, vlim=(-1, 1), colorbar=True, ncols=4, fig_dims=(4, 4)):

    if not type(fields) in [tuple, list]:
        fields = [fields]
    nrows = (len(fields)-1) // ncols + 1

    if len(fields) < ncols:
        ncols = len(fields)

    fig = plt.figure(figsize=(fig_dims[0]*ncols, fig_dims[1]*nrows))
    cms = []
    axs = []
    for i, field in enumerate(fields):
        ax = plt.subplot(nrows, ncols, i+1)
        cm = ax.pcolormesh(field.X, field.Y, np.real(field.psi), cmap=cmap, vmin=vlim[0], vmax=vlim[1], shading='nearest')
        if colorbar:
            plt.colorbar(cm, ax=ax, orientation='horizontal')

        ax.set_aspect('equal', adjustable='box')
        cms.append(cm)
        cms.append(ax)

    if show:
       plt.show() 
    else:
       return fig, axs


def set_size(field: ComplexField2D, Lx: float, Ly: float, in_place=False):

    if in_place:
        field.set_size(Lx, Ly)
        return
    else:
        field1 = field.copy()
        field1.set_size(Lx, Ly)
        return field1


def flip(field: ComplexField2D, axis: str, in_place=False):
    if axis not in ['X', 'Y']:
        raise ValueError(f'{axis} is not a valid axis for flipping') 

    psi1 = field.psi.copy()
    if axis == 'X':
        psi1 = psi1[:,::-1]
    if axis == 'Y':
        psi1 = psi1[::-1,:]

    if in_place:
        field.set_psi(psi1)
        return
    else:
        field1 = field.copy()
        field1.set_psi(psi1)
        return field1


def transpose(field: ComplexField2D, in_place=False):
    psi1 = np.transpose(field.psi)
    if in_place:
        field.set_dimensions(field.Ly, field.Lx, field.Ny, field.Nx)
        field.set_psi(psi1)

    else:
        field1 = field.copy()
        field1.set_dimensions(field.Ly, field.Lx, field.Ny, field.Nx)
        field1.set_psi(psi1)
        return field1


def rotate(field: ComplexField2D, angle: str, in_place=False):
    if not angle in ['90', '180', '270']:
        raise ValueError(f'{angle} is not a valid angle for rotation')

    psi1 = field.psi.copy()
    if angle == '90':
        psi1 = np.transpose(psi1)[::-1,:]
    if angle == '180':
        psi1 = psi1[::-1,::-1]
    if angle == '270':
        psi1 = np.transpose(psi1)[:,::-1]

    if in_place:
        if angle in ['90', '270']:
            field.set_dimensions(field.Ly, field.Lx, field.Ny, field.Nx)
        field.set_psi(psi1)
        return
    else:
        field1 = field.copy()
        if angle in ['90', '270']:
            field1.set_dimensions(field.Ly, field.Lx, field.Ny, field.Nx)
        field1.set_psi(psi1)
        return field1



