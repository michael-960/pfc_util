import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, rfft2, irfft2, set_global_backend
from pprint import pprint
import tqdm

import pyfftw
import time
import threading
import sys
from util.math import fourier
from util.common import overrides
from .field import FreeEnergyFunctional2D, FieldMinimizer, RealField2D, NoiseGenerator2D
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
        state_funcs = self.get_state_funcs()
        self.history.append_state_funcs(state_funcs)

    @overrides(FieldMinimizer)
    def on_epoch_end(self, progress_bar: tqdm.tqdm):
        state_funcs = self.get_state_funcs()
        #progress_bar.set_postfix_str(self.get_state_description_str(state_funcs))

        progress_bar.set_description_str(f'[{self.label}] {self.get_state_description_str(state_funcs)}')
        self.history.append_state_funcs(state_funcs)


    @overrides(FieldMinimizer)
    def on_nonstop_epoch_end(self):
        state_funcs = self.get_state_funcs()
        sys.stdout.write(f'\r[{self.label}] {self.get_state_description_str(state_funcs)}')
        self.history.append_state_funcs(state_funcs)

    @overrides(FieldMinimizer)
    def end(self):
        super().end()
        self.history.commit(self.label)


    def get_state_description_str(self, state, float_fmt='.7f', pad=1, delim='|'):
        delim_padded = ' '*pad + delim + ' '*pad

        state_func_list = []
        for st in state:
            if not st[1] is None:
                state_func_list.append(f'{st[0]}={st[1]:{float_fmt}}')

        return delim_padded.join(state_func_list)

    def get_state_funcs(self):
        return [('t', None), ('f', None), ('F', None), ('psibar', None), ('omega', None), ('Omega', None)]


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
    def get_state_funcs(self):
        psibar = np.mean(self.field.psi)
        psiN = psibar * self.field.Volume
        f = self.fef.mean_free_energy_density(self.field)
        F = self.fef.free_energy(self.field)
        omega = f - self.mu * psibar
        Omega = F - self.mu * psiN
        return [('t', self.age), ('f', f), ('F', F), ('psibar', psibar), ('omega', omega), ('Omega', Omega)]


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
    def get_state_funcs(self):
        psibar = np.mean(self.field.psi)
        f = self.fef.mean_free_energy_density(self.field)
        F = self.fef.free_energy(self.field)
        return [('t', self.age), ('f', f), ('F', F), ('psibar', psibar), ('omega', None), ('Omega', None)]


class StressRelaxer(PFCMinimizer):
    def __init__(self, field: RealField2D, dt: float):
        raise NotImplementedError


class PFCMinimizerHistory:
    def __init__(self):
        self.t = []
        self.f = []
        self.F = []
        self.omega = []
        self.Omega = []
        self.psibar = []
        self.age = 0
        self.label = 'NULL'

        self.item_dict = {
            't': (r'$t$', self.t),
            'f': (r'$f$', self.f),
            'F': (r'$F$', self.F),
            'omega': (r'$\omega$', self.omega),
            'Omega': (r'$\Omega$', self.Omega),
            'psibar': (r'$\bar\psi$', self.psibar) } 

        self.committed = False 

    def append(self, t, f, F, psibar, omega=None, Omega=None):
        if not self.committed:
            self.t.append(t)
            self.f.append(f)
            self.F.append(F)
            self.omega.append(omega)
            self.Omega.append(Omega)
            self.psibar.append(psibar)
            self.age = t
        else:
            raise ModifyingReadOnlyObjectError(
            f'history object (labe=\'{self.label}\') is already committed and hence not editable', self)

    def append_state_funcs(self, sf):
        self.append(sf[0][1], sf[1][1], sf[2][1], sf[3][1], sf[4][1], sf[5][1])

    def commit(self, label):
        if not self.committed:
            self.label = label
            self.committed = True 
        else:
            raise IllegalActionError(f'history object (label=\'{self.label}\') is already comitted')

    def get_item(self, item_name):
        if not item_name in self.item_dict.keys():
            raise ValueError(f'{item_name} is not a valid PFC minimizer history item')
        return self.item_dict[item_name][1]

    def get_item_latex(self, item_name):
        if not item_name in self.item_dict.keys():
            raise ValueError(f'{item_name} is not a valid history item')
        return self.item_dict[item_name][0]

    def export_state(self):
        if self.commited:
            state = self.item_dict
            state['age'] = self.age
            state['label'] = self.label
            return state
        else:
            raise IllegalActionError(
            'history object (label=\'{self.label}\') has not been commited and is therefore not ready to be exported')


def import_state(state):
    h = PFCMinimizerHistory()
    for i in range(state['t']):
        h.append(state['t'][i], state['f'][i], state['F'][i], state['psibar'][i], state['omega'][i], state['Omega'][i])

    h.commit(state['label'])
    return h




