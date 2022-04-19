import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, fft, ifft
from pprint import pprint


def modify_and_save(new_file_name, saved, **modified):
    data = dict(saved)
    for m in modified:
        data[m] = modified[m]
    np.savez(new_file_name, **data)

def change_resolution(psi, Nx1, Ny1):
    Nx, Ny = psi.shape[0], psi.shape[1]
    psi1 = np.zeros((Nx1, Ny1))

    for i in range(Nx1):
        for j in range(Ny1):
            psi1[i,j] = psi[int(i*Nx/Nx1),int(j*Ny/Ny1)]
    return psi1

def periodic_extend(psi, Mx, My):
    Nx, Ny = psi.shape[0], psi.shape[1]
    Nx1 = Nx * Mx
    Ny1 = Ny * My

    psi1 = np.zeros((Nx1, Ny1))

    for i in range(Nx1):
        for j in range(Ny1):
            psi1[i,j] = psi[i%Nx,j%Ny]
    return psi1




