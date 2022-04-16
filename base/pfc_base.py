import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, rfft2, irfft2, set_global_backend
from pprint import pprint

import pyfftw
import time
import threading
import sys
from util.math import fourier
from util.common import overrides
from .field import FreeEnergyFunctional2D, FieldMinimizer, Field2D, DensityFunctionalSystem2D, NoiseGenerator2D

import tqdm


class PFCFreeEnergyFunctional(FreeEnergyFunctional2D):
    def __init__(self, eps):
        self.eps = eps

    @overrides(FreeEnergyFunctional2D)
    def free_energy_density(self, field: Field2D):
        kernel = 1 - 2*field.K2 + field.K4
        psi_k = rfft2(field.psi)
        psi_k_o = kernel * psi_k
        f = 1/2 * field.psi * irfft2(psi_k_o) + field.psi**4/4 - self.eps/2 * field.psi**2
        return np.real(f)


    @overrides(FreeEnergyFunctional2D)
    def free_energy_density(self, field: Field2D):
        kernel = 1-2*field.K2+field.K4
        psi_k = rfft2(field.psi)
        psi_k_o = kernel * psi_k
        f = 1/2 * field.psi * irfft2(psi_k_o) + field.psi**4/4 - self.eps/2 * field.psi**2
        return np.real(f)

    @overrides(FreeEnergyFunctional2D)
    def derivative(self, field: Field2D):
        D2psi = irfft2(-field.K2*rfft2(field.psi))
        D4psi = irfft2(field.K4*rfft2(field.psi))
        local_mu = (1-self.eps) * field.psi + field.psi**3 + 2*D2psi + D4psi
        return local_mu

class PFCSystem(DensityFunctionalSystem2D):
    def __init__(self, field: Field2D, eps: float):
        super().__init__(field, PFCFreeEnergyFunctional(eps))

class PFCMinimizer(FieldMinimizer):
    def __init__(self, field: Field2D, dt, eps):
        super().__init__(field)
        self.dt = dt
        self.eps = eps
        self.age = 0

        self.history = PFCHistoryBlock(self)

        self.description = 'NULL'
        
    def set_age(self, age):
        self.age = age

    @overrides(FieldMinimizer)
    def step(self):
        raise NotImplementedError()

class PFCHistoryBlock:
    def __init__(self, minimizer: PFCMinimizer):
        self.minimizer = minimizer
        self.t = []
        self.omega = []
        self.Omega = []
        self.f = []
        self.F = []
        self.psibar = []

class ConstantChemicalPotentialMinimizer(PFCMinimizer):
    def __init__(self, field: Field2D, dt: float, eps: float, mu: float, noise_generator: NoiseGenerator2D):
        super().__init__(field, dt, eps)
   
        self._kernel = 1-2*self.field.K2+self.field.K4
        self._exp_dt_kernel = np.exp(-dt*self._kernel)
        self._exp_dt_eps = np.exp(dt*self.eps/2)
        self.mu = mu
        self.noise_generator = noise_generator

        self.fef = PFCFreeEnergyFunctional(eps)

        if not field.fft_initialized():
            field.initialize_fft()


    @overrides(PFCMinimizer)
    def step(self):
        self.age += self.dt
        self.field.psi += self.dt * self.noise_generator.generate() 

        self.field.psi *= self._exp_dt_eps
        self.field.psi -= self.dt/2 * (self.field.psi**3 - self.mu)
        
        self.field.fft2()
        self.field.psi_k *= self._exp_dt_kernel
        self.field.ifft2()

        self.field.psi *= self._exp_dt_eps
        self.field.psi -= self.dt/2 * (self.field.psi**3 - self.mu)
    
    @overrides(PFCMinimizer)
    def on_epoch_end(self, progress_bar: tqdm.tqdm):
        psibar = np.mean(self.field.psi)
        omega = self.fef.mean_free_energy_density(self.field) - self.mu*psibar
        progress_bar.set_postfix_str(f'omega={omega:.8f} | psi_bar={psibar:.6f} | t={self.age:.6f}')

        self.history.t.append(self.age)
        self.history.omega.append(omega)
        self.history.psibar.append(psibar)


class NonlocalConservedMinimizer(PFCMinimizer):
    def __init__(self, field: Field2D, dt: float, eps: float, noise_generator: NoiseGenerator2D):
        super().__init__()






