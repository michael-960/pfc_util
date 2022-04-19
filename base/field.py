import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, rfft2, irfft2, set_global_backend
from pprint import pprint
import pyfftw
from util.math import fourier
from util.common import overrides
import threading
import tqdm
import time

from .common import IllegalActionError, ModifyingReadOnlyObjectError

# A Field2D is a complex 2D field with definite dimensions, it also includes:
# psi, psi_k(fourier transform), forward plan, backward plan

class ComplexField2D:
    def __init__(self, Lx, Ly, Nx, Ny):
        self.set_dimensions(Lx, Ly, Nx, Ny)
        self.fft2 = None
        self.ifft2 = None
        
    def set_dimensions(self, Lx, Ly, Nx, Ny, verbose=False):
        self.Nx = Nx
        self.Ny = Ny
        self.set_size(Lx, Ly, verbose=verbose)
        self.psi = pyfftw.zeros_aligned((Nx, Ny), dtype='complex128')
        self.psi_k = pyfftw.zeros_aligned((Nx, Ny), dtype='complex128')
        if verbose:
            self.yell(f'new resolution set, Nx={Nx} Ny={Ny}')
            self.yell(f'reset profile, Lx={Lx} Ly={Ly} Nx={Nx} Ny={Ny}')

    def set_size(self, Lx, Ly, verbose=False):
        self.Lx = Lx
        self.Ly = Ly
        self.Volume = Lx*Ly
        x, kx, self.dx, self.dkx, y, ky, self.dy, self.dky = fourier.generate_xk_2d(Lx, Ly, self.Nx, self.Ny, real=False)
        self.dV = self.dx * self.dy
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        self.Kx, self.Ky = np.meshgrid(kx, ky, indexing='ij')

        self.K2 = self.Kx**2 + self.Ky**2
        self.K4 = self.K2**2
        self.K6 = self.K2 * self.K4

        if verbose:
            self.yell(f'new dimensions set, Lx={Lx} Ly={Ly}') 
            self.yell(f'k-space dimensions {self.K4.shape}')
    
    def initialize_fft(self, **fftwargs):
        psi_tmp = self.psi.copy()
        self.fft2 = pyfftw.FFTW(self.psi, self.psi_k, direction='FFTW_FORWARD', axes=(0,1), **fftwargs)
        self.ifft2 = pyfftw.FFTW(self.psi_k, self.psi, direction='FFTW_BACKWARD', axes=(0,1), **fftwargs)
        self.set_psi(psi_tmp)

    def fft_initialized(self):
        return not(self.fft2 is None or self.ifft2 is None)

    def set_psi(self, psi1, verbose=False):
        if not np.isscalar(psi1):
            if (psi1.shape[0] != self.Nx or psi1.shape[1] != self.Ny): 
                raise ValueError(f'array has incompatible shape {psi1.shape} with ({self.Nx, self.Ny})')

        self.psi[:,:] = psi1
        if verbose:
            self.yell('new psi set')

    def save(self, target_npz, verbose=False):
        if verbose:
            self.yell(f'dumping profile data to {target_npz}')
        np.savez(target_npz, **self.export_state()) 

    def export_state(self):
        state = {'psi': self.psi.copy(), 'Lx': self.Lx, 'Ly': self.Ly}
        return state
    
    def copy(self):
        field1 = self.__class__(self.Lx, self.Ly, self.Nx, self.Ny)
        field1.set_psi(self.psi)
        return field1

    def plot(self, lazy_factor=1, cmap='jet', vmin=-1, vmax=1):
        plt.figure(dpi=200)
        LF = lazy_factor
        ax = plt.gca()
        cm1 = ax.pcolormesh(self.X[::LF], self.Y[::LF], np.real(self.psi)[::LF], cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
        ax.set_aspect('equal', adjustable='box')

        plt.colorbar(cm1, ax=ax, orientation='horizontal', location='top', shrink=0.2)

        plt.margins(x=0, y=0, tight=True)
        plt.show()

    def yell(self, s):
        print(f'[field] {s}')


class RealField2D(ComplexField2D):
    def __init__(self, Lx, Ly, Nx, Ny):
        super().__init__(Lx, Ly, Nx, Ny)
        
    def set_dimensions(self, Lx, Ly, Nx, Ny, verbose=False):
        self.Nx = Nx
        self.Ny = Ny
        self.set_size(Lx, Ly, verbose=verbose)
        self.psi = pyfftw.zeros_aligned((Nx, Ny), dtype='float64')
        self.psi_k = pyfftw.zeros_aligned((Nx, Ny//2+1), dtype='complex128')
        if verbose:
            self.yell(f'new resolution set, Nx={Nx} Ny={Ny}')
            self.yell(f'reset profile, Lx={Lx} Ly={Ly} Nx={Nx} Ny={Ny}')

    def set_size(self, Lx, Ly, verbose=False):
        self.Lx = Lx
        self.Ly = Ly
        self.Volume = Lx*Ly
        x, kx, self.dx, self.dkx, y, ky, self.dy, self.dky = fourier.generate_xk_2d(Lx, Ly, self.Nx, self.Ny, real=True)
        self.dV = self.dx * self.dy
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        self.Kx, self.Ky = np.meshgrid(kx, ky, indexing='ij')

        self.K2 = self.Kx**2 + self.Ky**2
        self.K4 = self.K2**2
        self.K6 = self.K2 * self.K4

        if verbose:
            self.yell(f'new dimensions set, Lx={Lx} Ly={Ly}') 
            self.yell(f'k-space dimensions {self.K4.shape}')

    def set_psi(self, psi1, verbose=False):
        if not np.isscalar(psi1):
            if (psi1.shape[0] != self.Nx or psi1.shape[1] != self.Ny): 
                raise ValueError(f'array has incompatible shape {psi1.shape} with ({self.Nx, self.Ny})')
        
        if not np.all(np.isreal(psi1)):
            raise ValueError(f'array is complex') 

        self.psi[:,:] = psi1
        if verbose:
            self.yell('new psi set')


def load_field(filepath, is_complex=False):
    state = np.load(filepath)
    return import_field(state, is_complex=False)

def import_field(state, is_complex=False):
    psi = state['psi']
    Lx = state['Lx']
    Ly = state['Ly']
    Nx = psi.shape[0]
    Ny = psi.shape[1]
    if is_complex: 
        field = ComplexField2D(Lx, Ly, Nx, Ny)
        field.set_psi(psi, verbose=False)
        return field
    else:
        field = RealField2D(Lx, Ly, Nx, Ny)
        field.set_psi(psi, verbose=False)
        return field


class FieldMinimizer:
    def __init__(self, field: ComplexField2D):
        self.field = field
        self.started = False
        self.ended = False
        self.name = 'null'
        self.history = dict()

    # update field
    def step(self):
        pass

    def run_steps(self, N_steps):
        for i in range(N_steps):
            self.step()
            i += 1

    def run_multisteps(self, N_steps, N_epochs):
        self.start()

        progress_bar = tqdm.tqdm(range(N_epochs), bar_format='{l_bar}{bar}|{postfix}')
        self.on_create_progress_bar(progress_bar)
        for i in progress_bar:
            self.run_steps(N_steps)
            self.on_epoch_end(progress_bar)

        self.end()

    def run_nonstop(self, N_steps, custom_keyboard_interrupt_handler=None):
        self.start()
        lock = threading.Lock()
        stopped = False 
        def run():
            while True:
                with lock:
                    if stopped:
                        break
                    self.run_steps(N_steps)
                self.on_nonstop_epoch_end()

        thread = threading.Thread(target=run)
        thread.start()

        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt as e:
                with lock:
                    if custom_keyboard_interrupt_handler is None:
                        stopped = True 
                        break

                    else:
                        time.sleep(0.1)
                        stopped = custom_keyboard_interrupt_handler(self)
                        if stopped:
                            break

        thread.join()
        self.end()

    def on_create_progress_bar(self, progress_bar: tqdm.tqdm):
        pass

    def on_epoch_end(self, progress_bar: tqdm.tqdm):
        pass

    def on_nonstop_epoch_end(self):
        time.sleep(0.001)

    def start(self):
        if self.ended:
            raise MinimizerError(self, 'minimization has already ended')
        self.started = True

    def end(self):
        if self.started:
            self.ended = True
        else:
            raise MinimizerError(self, 'minimization has not been started')


class MinimizerError(IllegalActionError):
    def __init__(self, minimizer: FieldMinimizer, msg=None):
        if minimizer is None:
            self.message = 'no minimizer provided'
        else:
            self.message = msg
        super().__init__(self.message)
        self.minimier = minimizer


class NoiseGenerator2D:
    def __init__(self, seed, amplitude, Nx, Ny, noise_type='gaussian'):
        self.seed = seed
        self.amplitude = amplitude
        self.Nx = Nx
        self.Ny = Ny

        if noise_type == 'gaussian':
            self.generate = lambda: np.random.normal(0, amplitude, size=(Nx, Ny))
        else:
            raise ValueError(f'{noise_error} is not a recognized noise type')

    def generate(self):
        raise NotImplementedError()


# an abstract class
class FreeEnergyFunctional2D:
    def __init__(self):
        raise NotImplementedError()

    def free_energy_density(self, field: ComplexField2D):
        raise NotImplementedError()

    def free_energy(self, field: ComplexField2D):
        return np.sum(self.free_energy_density(field)) * field.dV

    def derivative(self, field: ComplexField2D):
        raise NotImplementedError()

    def mean_free_energy_density(self, field: ComplexField2D):
        return np.mean(self.free_energy_density(field))


class FieldOperationError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

