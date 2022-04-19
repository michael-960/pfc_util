import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, rfft2, irfft2, set_global_backend
from pprint import pprint
import tqdm

import pyfftw
import time
import threading
import sys
from typing import List


from util.math import fourier
from util.common import overrides
from .field import FreeEnergyFunctional2D, FieldMinimizer, RealField2D, NoiseGenerator2D, import_field
from .common import ModifyingReadOnlyObjectError, IllegalActionError


class PFCFreeEnergyFunctional(FreeEnergyFunctional2D):
    def __init__(self, eps):
        self.eps = eps

    @overrides(FreeEnergyFunctional2D)
    def free_energy_density(self, field: RealField2D):
        kernel = 1 - 2*field.K2 + field.K4
        psi_k = rfft2(field.psi)
        psi_k_o = kernel * psi_k
        f = 1/2 * field.psi * irfft2(psi_k_o) + field.psi**4/4 - self.eps/2 * field.psi**2
        return np.real(f)


    @overrides(FreeEnergyFunctional2D)
    def free_energy_density(self, field: RealField2D):
        kernel = 1-2*field.K2+field.K4
        psi_k = rfft2(field.psi)
        psi_k_o = kernel * psi_k
        f = 1/2 * field.psi * irfft2(psi_k_o) + field.psi**4/4 - self.eps/2 * field.psi**2
        return np.real(f)

    @overrides(FreeEnergyFunctional2D)
    def derivative(self, field: RealField2D):
        D2psi = irfft2(-field.K2*rfft2(field.psi))
        D4psi = irfft2(field.K4*rfft2(field.psi))
        local_mu = (1-self.eps) * field.psi + field.psi**3 + 2*D2psi + D4psi
        return local_mu

class PFCStateFunction:
    def __init__(self, Lx, Ly, f, F, psibar, omega=None, Omega=None):
        self.Lx = Lx
        self.Ly = Ly

        self.f = f
        self.F = F
        self.psibar = psibar

        self.omega = omega
        self.Omega = Omega

        self.item_dict = {
            'Lx': self.Lx,
            'Ly': self.Ly,
            'f': self.f,
            'F': self.F,
            'omega': self.omega,
            'Omega': self.Omega,
            'psibar': self.psibar
        } 

    def is_grand_canonical(self):
        return not (self.omega is None)

    def get_item(self, item_name):
        if not item_name in self.item_dict.keys():
            raise ValueError(f'{item_name} is not a valid PFC state function')
        return self.item_dict[item_name]

    def to_string(self, float_fmt='.7f', pad=1, delim='|'):
        delim_padded = ' '*pad + delim + ' '*pad
        items = ['Lx', 'Ly', 'f', 'F', 'omega', 'Omega', 'psibar']
        state_func_list = []

        for item_name in items:
            item = self.item_dict[item_name]
            if not item is None: 
                state_func_list.append(f'{item_name}={item:{float_fmt}}')
        return delim_padded.join(state_func_list)

    def export(self) -> str:
        return self.item_dict


def import_state_function(state: dict) -> PFCStateFunction:
    sf = PFCStateFunction(state['Lx'], state['Ly'], state['f'], state['F'], state['psibar'], state['omega'], state['Omega'])
    return sf

def get_latex(item_name):
    if not item_name in _item_latex_dict.keys():
        raise ValueError(f'{item_name} is not a valid state function function')
    return _item_latex_dict[item_name]


class PFCMinimizer(FieldMinimizer):
    def __init__(self, field: RealField2D, dt, eps):
        super().__init__(field)
        self.dt = dt
        self.eps = eps
        self.age = 0

        self.label = 'NULL'
        self.history = PFCMinimizerHistory()

    @overrides(FieldMinimizer)
    def on_create_progress_bar(self, progress_bar: tqdm.tqdm):
        progress_bar.set_description_str(f'[{self.label}]')
        
    def set_age(self, age):
        self.age = age

    @overrides(FieldMinimizer)
    def step(self):
        raise NotImplementedError()

    @overrides(FieldMinimizer)
    def start(self):
        super().start()
        state_function = self.get_state_function()
        self.history.append_state_function(self.age, state_function)

    @overrides(FieldMinimizer)
    def on_epoch_end(self, progress_bar: tqdm.tqdm):
        state_function = self.get_state_function()
        progress_bar.set_description_str(f'[{self.label}] {state_function.to_string()}')
        self.history.append_state_function(self.age, state_function)

    @overrides(FieldMinimizer)
    def on_nonstop_epoch_end(self):
        state_function = self.get_state_function()
        sys.stdout.write(f'\r[{self.label}] {state_function.to_string()}')
        self.history.append_state_function(state_function)

    @overrides(FieldMinimizer)
    def end(self):
        super().end()
        self.history.commit(self.label, self.field)

    def get_state_function(self) -> PFCStateFunction:
        raise NotImplementedError()


class ConstantChemicalPotentialMinimizer(PFCMinimizer):
    def __init__(self, field: RealField2D, dt: float, eps: float, mu: float, noise_generator:NoiseGenerator2D=None):
        super().__init__(field, dt, eps)
        self.label = f'const-mu eps={eps} mu={mu} dt={dt}'
   
        self._kernel = 1-2*self.field.K2+self.field.K4
        self._exp_dt_kernel = np.exp(-dt*self._kernel)
        self._exp_dt_eps_half = np.exp(dt*self.eps/2)
        self.mu = mu
        self.noise_generator = noise_generator

        self.fef = PFCFreeEnergyFunctional(eps)
        self.is_noisy = not (noise_generator is None)

        if not field.fft_initialized():
            field.initialize_fft()

    @overrides(PFCMinimizer)
    def step(self):
        self.age += self.dt
        if self.is_noisy:
            self.field.psi += self.dt * self.noise_generator.generate() 

        self.field.psi *= self._exp_dt_eps_half
        self.field.psi -= self.dt/2 * (self.field.psi**3 - self.mu)
        
        self.field.fft2()
        self.field.psi_k *= self._exp_dt_kernel
        self.field.ifft2()

        self.field.psi *= self._exp_dt_eps_half
        self.field.psi -= self.dt/2 * (self.field.psi**3 - self.mu)

    @overrides(PFCMinimizer)
    def get_state_function(self):
        psibar = np.mean(self.field.psi)
        psiN = psibar * self.field.Volume
        f = self.fef.mean_free_energy_density(self.field)
        F = self.fef.free_energy(self.field)
        omega = f - self.mu * psibar
        Omega = F - self.mu * psiN
        return PFCStateFunction(self.field.Lx, self.field.Ly, f, F, psibar, omega, Omega)


class NonlocalConservedMinimizer(PFCMinimizer):
    def __init__(self, field: RealField2D, dt: float, eps: float, noise_generator:NoiseGenerator2D=None):
        super().__init__(field, dt, eps)
        self.label = f'nonlocal eps={eps} dt={dt}'

        self._kernel = 1-2*self.field.K2+self.field.K4
        self._exp_dt_kernel = np.exp(-dt*self._kernel)
        self._exp_dt_eps_full = np.exp(dt*self.eps)
        self.noise_generator = noise_generator
        self.fef = PFCFreeEnergyFunctional(eps)
        self.is_noisy = not (noise_generator is None)

        if not field.fft_initialized():
            field.initialize_fft()


    @overrides(PFCMinimizer)
    def step(self):
        self.age += self.dt
        psi3k = rfft2(self.field.psi**3)

        if self.is_noisy:
            self.field.psi += self.dt * self.noise_generator.generate() 

        self.field.fft2()
        psik00 = self.field.psi_k[0,0]
        self.field.psi_k -= self.dt * psi3k
        self.field.psi_k *= self._exp_dt_eps_full
        self.field.psi_k *= self._exp_dt_kernel
        self.field.psi_k[0,0] = psik00
        self.field.ifft2()
        
    @overrides(PFCMinimizer)
    def get_state_function(self):
        psibar = np.mean(self.field.psi)
        f = self.fef.mean_free_energy_density(self.field)
        F = self.fef.free_energy(self.field)
        return PFCStateFunction(self.field.Lx, self.field.Ly, f, F, psibar)


class StressRelaxer(PFCMinimizer):
    def __init__(self, field: RealField2D, dt: float):
        raise NotImplementedError


class PFCMinimizerHistory:
    def __init__(self):
        self.t = []
        self.state_functions = []
        self.age = 0
        self.label = 'NULL'
        self.final_field_state = None
        self.committed = False 

    def append_state_function(self, t, sf: PFCStateFunction):
        if self.committed:
            raise ModifyingReadOnlyObjectError(
            f'history object (labe=\'{self.label}\') is already committed and hence not editable', self)

        if t < self.age:
            raise IllegalActionError(f'time={t} is smaller than the current recorded time')

        self.t.append(t)
        self.state_functions.append(sf)
        self.age = t

    def commit(self, label, field: RealField2D):
        if self.committed:
            raise IllegalActionError(f'history object (label=\'{self.label}\') is already committed')

        self.label = label
        self.committed = True 
        self.final_field_state = field.export_state()

    def is_committed(self):
        return self.committed

    def get_t(self):
        return self.t

    def get_state_functions(self) -> List[PFCStateFunction]:
        return self.state_functions

    def get_final_field_state(self):
        if not self.committed:
            raise IllegalActionError('cannot get final state from uncommitted PFC minimizer history')
        return self.final_field_state

    def get_label(self):
        return self.label

    def export(self) -> dict:
        if self.committed:
            state = dict()
            state['age'] = self.age
            state['label'] = self.label
            state['final_field_state'] = self.final_field_state
            state['state_functions'] = [sf.export() for sf in self.state_functions]
            state['t'] = self.t
            return state
        else:
            raise IllegalActionError(
            'history object (label=\'{self.label}\') has not been committed and is therefore not ready to be exported')


def import_minimizer_history(state: dict) -> PFCMinimizerHistory:
    mh = PFCMinimizerHistory()
    mh.label = state['label']
    mh.state_functions = [import_state_function(sf_state) for sf_state in state['state_functions']]
    mh.age = state['age']
    mh.final_field_state = state['final_field_state']
    mh.t = state['t']
    mh.committed = True
    return mh


_item_latex_dict = {
    'Lx': r'$Lx$',
    'Ly': r'$Ly$',
    'f': r'$f$',
    'F': r'$F$',
    'omega': r'$\omega$',
    'Omega': r'$\Omega$',
    'psibar': r'$\bar\psi$'
} 



