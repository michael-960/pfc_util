
raise ImportError('Module is outdated and deprecated')

import tqdm
from typing import Optional

import numpy as np

from michael960lib.common import overrides, experimental
from michael960lib.common import ModifyingReadOnlyObjectError, IllegalActionError

from torusgrid.dynamics._base import FancyEvolver, NoiseGenerator2D, EvolverHistory
from torusgrid.fields import RealField2D

from .base import StateFunction, FreeEnergyFunctional, import_state_function


# deprecated module

PFCStateFunction = StateFunction
PFCFreeEnergyFunctional = FreeEnergyFunctional


class PFCMinimizer(FancyEvolver):
    def __init__(self, field: RealField2D, dt: float, eps: float):
        super().__init__(field)
        self.field: RealField2D

        self.dt = dt
        self.eps = eps
        self.age = 0
        self.history = PFCMinimizerHistory()

        self.info['system'] = 'pfc'
        self.info['minimizer'] = 'NULL'

        self.info['dt'] = dt
        self.info['eps'] = eps
        self.info['label'] = self.label = f'NULL eps={eps} dt={dt}'

        self.display_format = '[{label}] f={f:.5f} F={F:.5f} psibar={psibar:.5f}'

    
    @overrides(FancyEvolver)
    def get_evolver_state(self):
        return {'age': self.age}

    @overrides(FancyEvolver)
    def on_create_progress_bar(self, progress_bar: tqdm.tqdm):
        progress_bar.set_description_str(f'[{self.label}]')
        
    def set_age(self, age):
        self.age = age


class ConstantChemicalPotentialMinimizer(PFCMinimizer):
    def __init__(self, 
            field: RealField2D, 
            dt: float, eps: float, mu: float, 
            noise_generator:Optional[NoiseGenerator2D]=None):

        super().__init__(field, dt, eps)
        self.info['minimizer'] = 'mu'
        self.info['mu'] = self.mu = mu
        self.info['label'] = self.label = f'mu eps={eps:.5f} mu={mu:.5f} dt={dt:.5f}'
   
        self.fef = PFCFreeEnergyFunctional(eps)
        self.noise_generator = noise_generator


        if not field.fft_initialized():
            field.initialize_fft() 
        self._kernel = 1-2*self.field.K2+self.field.K4 - self.eps
        self._exp_dt_kernel = np.exp(-dt*self._kernel)
        self._mu_dt_half = self.dt * self.mu / 2

    @overrides(PFCMinimizer)
    def step(self):
        self.age += self.dt

        if self.noise_generator is not None:
            self.field.psi += self.dt * self.noise_generator.generate() 

        self.field.psi[:] += self._mu_dt_half
        self.field.psi /=np.sqrt(1+self.field.psi**2*self.dt)
        
        self.field.fft()
        self.field.psi_k *= self._exp_dt_kernel
        self.field.ifft()

        self.field.psi /= np.sqrt(1+self.field.psi**2*self.dt)
        self.field.psi[:] += self._mu_dt_half

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
    def __init__(self,
            field: RealField2D, dt: float, eps: float, 
            noise_generator: Optional[NoiseGenerator2D] = None):
        super().__init__(field, dt, eps)

        self.info['label'] = self.label = f'nonlocal eps={eps:.5f} dt={dt:.5f}'
        self.info['minimizer'] = 'nonlocal'

        self._kernel = 1-2*self.field.K2+self.field.K4 - self.eps
        self._exp_dt_kernel = np.exp(-dt*self._kernel)
        self.noise_generator = noise_generator
        self.fef = PFCFreeEnergyFunctional(eps)

        if not field.fft_initialized():
            field.initialize_fft()

        self.psibar = np.mean(self.field.psi)

    @overrides(PFCMinimizer)
    def step(self):
        self.age += self.dt

        if self.noise_generator is not None:
            self.field.psi += self.dt * self.noise_generator.generate() 
        
        self.field.psi /= np.sqrt(1+self.field.psi**2*self.dt)
        self.field.psi += - np.mean(self.field.psi) + self.psibar

        self.field.fft()
        self.field.psi_k *= self._exp_dt_kernel
        self.field.ifft()

        self.field.psi += - np.mean(self.field.psi) + self.psibar

        self.field.psi /= np.sqrt(1+self.field.psi**2*self.dt)
        self.field.psi += - np.mean(self.field.psi) + self.psibar

    @overrides(PFCMinimizer)
    def get_state_function(self):
        psibar = np.mean(self.field.psi)
        f = self.fef.mean_free_energy_density(self.field)
        F = self.fef.free_energy(self.field)
        return PFCStateFunction(self.field.Lx, self.field.Ly, f, F, psibar)


class NonlocalConservedRK4(NonlocalConservedMinimizer):
    @experimental('Nonlocal RK4 is experimental and not optimized')
    def __init__(self, field: RealField2D, dt: float, eps: float,
            noise_generator: Optional[NoiseGenerator2D]=None,
            k_regularizer=0.1, inertia=100):
        super().__init__(field, dt, eps, noise_generator)
        self.R = k_regularizer
        self.info['minimizer'] = 'nonlocal-rk4'
        self.info['M'] = inertia
        self.info['R'] = k_regularizer
        self.info['label'] = self.label = f'nonlocal-rk4 eps={eps} dt={dt} R={k_regularizer} M={inertia}'

        self.field_tmp = self.field.copy()
        self.field_tmp.initialize_fft()

        self._deriv = RealField2D(field.Lx, field.Ly, field.Nx, field.Ny)
        self._deriv.initialize_fft()

        self.dfield = self.field.copy()
        self.dfield.set_psi(0)
        self.dfield.initialize_fft()

        self.dfield_tmp = self.dfield.copy()
        self.dfield_tmp.initialize_fft()

        self.inertia = inertia

    def _psi_dot(self):
        self._deriv.psi[:,:] = -self.fef.derivative(self.field_tmp)
        self._deriv.fft()
        self._deriv.psi_k *= np.exp(-self.R*self._deriv.K2)
        self._deriv.ifft()
        F = self._deriv.psi - np.mean(self._deriv.psi)
        return self.dfield_tmp.psi*self.inertia, -(self.dfield_tmp.psi - F)

    # RK4
    @overrides(NonlocalConservedMinimizer)
    def step(self):
        self.age += self.dt
        self.field_tmp.psi[:] = self.field.psi
        self.dfield_tmp.psi[:] = self.dfield.psi

        k1, l1 = self._psi_dot()
        self.field_tmp.psi += k1*self.dt/2
        self.dfield_tmp.psi += l1*self.dt/2

        k2, l2 = self._psi_dot()
        self.field_tmp.psi += k2*self.dt/2
        self.dfield_tmp.psi += l2*self.dt/2

        k3, l3 = self._psi_dot()
        self.field_tmp.psi += k3*self.dt
        self.dfield_tmp.psi += l3*self.dt

        k4, l4 = self._psi_dot()

        self.field.psi += self.dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        self.dfield.psi += self.dt / 6 * (l1 + 2*l2 + 2*l3 + l4)
 

class NonlocalConservedRK4Plain(NonlocalConservedRK4):
    def __init__(self, field, dt, eps,
            noise_generator: Optional[NoiseGenerator2D]=None,
            k_regularizer=0.1):
        super().__init__(field, dt, eps, noise_generator, k_regularizer, 0)

    def _psi_dot(self):
        self._deriv.psi[:,:] = -self.fef.derivative(self.field_tmp)
        self._deriv.fft()
        self._deriv.psi_k *= np.exp(-self.R*self._deriv.K2)
        self._deriv.ifft()
        F = self._deriv.psi - np.mean(self._deriv.psi)
        return F

    @overrides(NonlocalConservedMinimizer)
    def step(self):
        self.age += self.dt
        self.field_tmp.psi[:] = self.field.psi

        k1 = self._psi_dot()
        self.field_tmp.psi += k1*self.dt/2

        k2 = self._psi_dot()
        self.field_tmp.psi += k2*self.dt/2

        k3 = self._psi_dot()
        self.field_tmp.psi += k3*self.dt

        k4 = self._psi_dot()

        self.field.psi += self.dt / 6 * (k1 + 2*k2 + 2*k3 + k4)


class NonlocalDescent(NonlocalConservedMinimizer):
    def __init__(self, field: RealField2D, dt: float, eps: float,
            noise_generator: Optional[NoiseGenerator2D]=None,
            k_regularizer=0.1):
        super().__init__(field, dt, eps, noise_generator)
        self.R = k_regularizer
        self.info['minimizer'] = 'nonlocal-desc'
        self.info['R'] = k_regularizer
        self.info['label'] = self.label = f'nonlocal-desc eps={eps} dt={dt} R={k_regularizer}'

        self.field_tmp = self.field.copy()
        self.field_tmp.initialize_fft()

        self._deriv = RealField2D(field.Lx, field.Ly, field.Nx, field.Ny)
        self._deriv.initialize_fft()

    def _psi_dot(self):
        self._deriv.psi[:,:] = -self.fef.derivative(self.field_tmp)
        self._deriv.fft()
        self._deriv.psi_k *= np.exp(-self.R*self._deriv.K2)
        self._deriv.ifft()
        F = self._deriv.psi - np.mean(self._deriv.psi)
        return F

    @overrides(NonlocalConservedMinimizer)
    def step(self):
        self.age += self.dt
        self.field_tmp.psi[:,:] = self.field.psi[:,:]
        F = self._psi_dot()
        cutoff = 0.01 * self.dt

        dT = self.dt / (np.sum(F**2)*self.field.dV + cutoff)
        self.field.psi += F * dT


class NonlocalTeleporter(NonlocalConservedRK4):
    def __init__(self, dt, eps):
        super().__init__(field, dt, eps, inertia=0)

    @overrides(NonlocalConservedMinimizer)
    def step(self):
        raise NotImplementedError
 


class ConservedMinimizer(PFCMinimizer):
    @experimental('Local conserved minimizer is not implemented yet')
    def __init__(self, field: RealField2D, dt: float, eps: float, noise_generator:NoiseGenerator2D=None):
        super().__init__(field, dt, eps)
        self.label = f'nonlocal eps={eps} dt={dt}'

        self._kernel = 1-2*self.field.K2+self.field.K4 - self.eps
        self._exp_dt_kernel = np.exp(-dt*self._kernel)
        self.noise_generator = noise_generator
        self.fef = PFCFreeEnergyFunctional(eps)
        self.is_noisy = not (noise_generator is None)

        if not field.fft_initialized():
            field.initialize_fft()

        self.psibar = np.mean(self.field.psi)

        self.field_tmp = self.field.copy()

    #@overrides(PFCMinimizer)
    def step(self):
        raise NotImplementedError
        self.age += self.dt

        if self.is_noisy:
            self.field.psi += self.dt * self.noise_generator.generate() 
        
        self.field.psi /= np.sqrt(1+self.field.psi**2*self.dt)
        self.field.psi += - np.mean(self.field.psi) + self.psibar

        self.field.fft()
        self.field.psi_k *= self._exp_dt_kernel
        self.field.ifft()
        self.field.psi += - np.mean(self.field.psi) + self.psibar

        self.field.psi /= np.sqrt(1+self.field.psi**2*self.dt)
        self.field.psi += - np.mean(self.field.psi) + self.psibar



    @overrides(PFCMinimizer)
    def get_state_function(self):
        psibar = np.mean(self.field.psi)
        f = self.fef.mean_free_energy_density(self.field)
        F = self.fef.free_energy(self.field)
        return PFCStateFunction(self.field.Lx, self.field.Ly, f, F, psibar)


class StressRelaxer(PFCMinimizer):
    def __init__(self, field: RealField2D, 
            dt: float, eps: float, mu: float,
            noise_generator: Optional[NoiseGenerator2D] = None,
            expansion_rate: float=1):
        super().__init__(field, dt, eps)
        self.f = 1

        self.info['mu'] = mu
        self.info['label'] = self.label = f'stress-relax eps={eps:.5f} mu={mu:.5f} dt={dt:.5f}'
        self.info['minimizer'] = 'stress-relax'
        self.info['expansion_rate'] = expansion_rate

        self.dt = dt
        self.eps = eps
        self.mu = mu
        self.noise_generator = noise_generator
        self.is_noisy = not (noise_generator is None)
        self.fef = PFCFreeEnergyFunctional(eps)
        self.dT = self.dt * expansion_rate / 2
        self.NN = field.Nx * field.Ny

        if not field.fft_initialized():
            field.initialize_fft()

        self._prepare_minimization()


    def _prepare_minimization(self):
        f = self.field
        #self._exp_dt_eps_half = np.exp(self.dt*self.eps/2)
        self._mu_dt_half = self.dt * self.mu / 2
        #self._conv_helper = self.field.K2 * 0 + 2/self.NN
        self._2_NN = 2 / self.NN

        self._domega_kernels = np.array([
            2/f.Lx*f.Kx2*(1-f.K2), 
            2/f.Ly*f.Ky2*(1-f.K2), 
            -2/f.Lx**2*f.Kx2*(3-5*f.Kx2-3*f.Ky2),
            -2/f.Ly**2*f.Ky2*(3-5*f.Ky2-3*f.Kx2),
            2/f.Lx/f.Ly * 2*f.Kx2*f.Ky2
        ])

        self.Lx0 = f.Lx
        self.Ly0 = f.Ly

        self.set_size_scale(1., 1.)

    @overrides(PFCMinimizer)
    def step(self):
        f = self.field
        self.age += self.dt
        if self.is_noisy:
            f.psi += self.dt * self.noise_generator.generate() 

        f.psi += self._mu_dt_half
        f.psi /=np.sqrt(1+f.psi**2*self.dt)
        
        f.fft()

        _kernel = 1-2*self.K2+self.K4 - self.eps
        _exp_dt_kernel = np.exp(-self.dt*_kernel/2)
        f.psi_k *= _exp_dt_kernel

        self.relax_stress_full()

        _kernel = 1-2*self.K2+self.K4 - self.eps
        _exp_dt_kernel = np.exp(-self.dt*_kernel/2)
        f.psi_k *= _exp_dt_kernel

        f.ifft()

        f.psi /= np.sqrt(1+self.field.psi**2*self.dt)
        f.psi += self._mu_dt_half

    def relax_stress_full(self):
        f = self.field
        #f = self
        dT = self.dt

        self._domega_kernels[:,:,:] = [
                2/self.Lx*self.Kx2*(1-self.K2), 
                2/self.Ly*self.Ky2*(1-self.K2), 
                -2/self.Lx**2*self.Kx2*(3-5*self.Kx2-3*self.Ky2),
                -2/self.Ly**2*self.Ky2*(3-5*self.Ky2-3*self.Kx2),
                2/self.Lx/self.Ly * 2*self.Kx2*self.Ky2]

        omega_list = self.real_convolution_2d(np.abs(f.psi_k**2), self._domega_kernels)

        dLx = -omega_list[0]*dT + (omega_list[0] * omega_list[2] + omega_list[1] * omega_list[4]) * dT**2/2
        dLy = -omega_list[1]*dT + (omega_list[1] * omega_list[3] + omega_list[0] * omega_list[4]) * dT**2/2

        # Lx = Lx0 * fx
        # dLx = Lx0 * dfx
        dfx = dLx / self.Lx0
        dfy = dLy / self.Ly0

        #f.set_size(f.Lx+dLx, f.Ly+dLy)
        self.set_size_scale(self.fx + dfx, self.fy + dfy)

    # size scale factor compared to the original dimensions
    def set_size_scale(self, fx, fy):
        self.fx = fx
        self.fy = fy
        self.Kx2 = self.field.Kx2 / fx**2
        self.Ky2 = self.field.Ky2 / fy**2
        self.K2 = self.Kx2 + self.Ky2
        self.K4 = self.K2**2
        self.Lx = self.field.Lx * self.fx
        self.Ly = self.field.Ly * self.fy


    @overrides(PFCMinimizer)
    def get_state_function(self):
        self.field.set_size(self.Lx, self.Ly)
        self.set_size_scale(1., 1.)
        psibar = np.mean(self.field.psi)
        psiN = psibar * self.field.Volume

        f = self.fef.mean_free_energy_density(self.field)
        F = self.fef.free_energy(self.field)

        omega = f - self.mu * psibar
        Omega = F - self.mu * psiN

        return PFCStateFunction(self.field.Lx, self.field.Ly, f, F, psibar, omega, Omega)

    def real_convolution_2d(self, psi_k_sq, kernel):
        r = kernel * psi_k_sq
        r[:,:,[0,-1]] *= 0.5
        return r.sum(axis=(1,2)) * self._2_NN


class RandomStepMinimizer(PFCMinimizer):
    def __init__(self, field: RealField2D, dt: float, eps: float, mu: float, seed: int):
        super().__init__(field, dt, eps)
        self.target = 'Omega'

        self.info['minimizer'] = 'random'
        self.info['label'] = self.label = 'random eps={eps:.5f} mu={mu:.5f} dt={dt:.5f}'

        if mu is None:
            self.target = 'F'
            self.label = 'random eps={eps} dt={dt}'

        self.mu = mu
        self.fef = PFCFreeEnergyFunctional(eps)

        self.generator = np.random.default_rng(seed)
        self.candidate_field = self.field.copy()

        self.random_scale = self.dt**2 * np.sqrt(np.mean(self.fef.derivative(self.field)**2))
        print(f'scale = {self.random_scale}')

    @overrides(PFCMinimizer) 
    def step(self):
        self.age += self.dt
        self.candidate_field.psi = self.field.psi + self.generator.normal(0, self.random_scale, size=(self.field.Nx,self.field.Ny))
        if self.target == 'F':
            F0 = self.fef.free_energy(self.field) 
            F1 = self.fef.free_energy(self.candidate_field)
            if F1 < F0:
                self.field.psi[:,:] = self.candidate_field.psi
        if self.target == 'Omega':
            O0 = self.fef.grand_potential(self.field, self.mu) 
            O1 = self.fef.grand_potential(self.candidate_field, self.mu)
            if O1 < O0:
                self.field.psi[:,:] = self.candidate_field.psi


    @overrides(PFCMinimizer)
    def get_state_function(self):
        Lx = self.field.Lx
        Ly = self.field.Ly
        psibar = np.mean(self.field.psi)
        f = self.fef.mean_free_energy_density(self.field)
        F = self.fef.free_energy(self.field)

        omega = None
        Omega = None
        if self.target == 'Omega':
            omega = self.fef.mean_grand_potential_density(self.field, self.mu)
            Omega = self.fef.grand_potential(self.field, self.mu)

        return PFCStateFunction(Lx, Ly, f, F, psibar, omega, Omega)



class PFCMinimizerHistory(EvolverHistory):
    def __init__(self):
        super().__init__()
        self.t = []
        self.age = 0

    @overrides(EvolverHistory)
    def append_state_function(self, evolver_state: dict, sf: PFCStateFunction):
        if self.committed:
            raise ModifyingReadOnlyObjectError(
            f'history object (label=\'{self.label}\') is already committed and hence not editable', self)

        t = evolver_state['age']
        if t < self.age:
            raise IllegalActionError(f'time={t} is smaller than the current recorded time')

        self.t.append(t)
        self.state_functions.append(sf)
        self.age = t

    def get_t(self):
        return self.t

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



