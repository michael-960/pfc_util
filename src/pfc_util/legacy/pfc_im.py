import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, rfft2, irfft2, set_global_backend
from pprint import pprint
import pyfftw
import time
import threading
import sys

from michael960lib.math import fourier
from michael960lib.common import overrides


# pyfftw implementation
class PhaseFieldCrystal2D:
    def __init__(self, Lx, Ly, Nx, Ny, verbose=True):

        self.set_dimensions(Lx, Ly, Nx, Ny, verbose)
        self.set_psi(np.zeros((Nx, Ny)), verbose=verbose)
        self.set_params(0, 0, verbose=verbose)
        self.set_noise(0, 0)
        self.lock = None
        self.new_lock()
        self.new_history()

        #self.minimizer = 'mu'
        #self.running = False

        set_global_backend(pyfftw.interfaces.scipy_fft)

    def set_noise(self, seed, width):
        self.seed = seed
        np.random.seed(self.seed)
        self.noise_width = width

    def set_params(self, mu, eps, verbose=True):
        self.mu = mu
        self.eps = eps
        if verbose:
            self.yell(f'new params set, mu={mu} eps={eps}')

    def set_dimensions(self, Lx, Ly, Nx, Ny, verbose=True):

        self.Nx = Nx
        self.Ny = Ny
        
        self.resize(Lx, Ly, verbose=verbose)

        self.psi = pyfftw.zeros_aligned((Nx, Ny), dtype='float64')
        self.psi_k = pyfftw.zeros_aligned((Nx, Ny//2+1), dtype='complex128')

        if verbose:
            self.yell(f'new resolution set, Nx={Nx} Ny={Ny}')
            self.yell(f'reset profile, Lx={Lx} Ly={Ly} Nx={Nx} Ny={Ny}')

    def resize(self, Lx, Ly, verbose=True):
        self.Lx = Lx
        self.Ly = Ly
        self.Volume = Lx*Ly
        x, kx, self.dx, self.dkx, y, ky, self.dy, self.dky = fourier.generate_xk_2d(Lx, Ly, self.Nx, self.Ny, real=True)

        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        self.Kx, self.Ky = np.meshgrid(kx, ky, indexing='ij')
        self.K2 = self.Kx**2 + self.Ky**2
        self.K4 = self.K2**2

        self.Kernel = 1-2*self.K2+self.K4

        if verbose:
            self.yell(f'new dimensions set, Lx={Lx} Ly={Ly}') 
            self.yell(f'k-space dimensions {self.K4.shape}')

    def initialize_fftw(self, **fftwargs):
        psi_tmp = self.psi.copy()
        self.for_psi = pyfftw.FFTW(self.psi, self.psi_k, direction='FFTW_FORWARD', axes=(0,1), **fftwargs)
        self.bac_psi = pyfftw.FFTW(self.psi_k, self.psi, direction='FFTW_BACKWARD', axes=(0,1), **fftwargs)
        self.set_psi(psi_tmp)

    def set_psi(self, psi1, verbose=True):
        if psi1.shape[0] != self.Nx or psi1.shape[1] != self.Ny:
            raise ValueError(f'array has incompatible shape {psi1.shape} with ({self.Nx, self.Ny})')

        self.psi[:,:] = psi1
        if verbose:
            self.yell('new psi set')

    def new_history(self):
        #self.history = {'t': [], 'Omega': [], 'psi0': [], 'max-min': [], 'omega': []}
        self.history = []

    def load_profile_from_file(self, saved_npz, verbose=True):
        file_list = saved_npz.files
        if verbose:
            self.yell(f'loading npz file, found the following data:')
            self.yell(f'{file_list}')

        psi = saved_npz['psi']

        self.set_dimensions(saved_npz['Lx'], saved_npz['Ly'], *psi.shape, verbose=verbose)

        self.set_params(saved_npz['mu'], saved_npz['eps'], verbose=verbose)
        self.set_psi(saved_npz['psi'], verbose=verbose)
        self.dt = saved_npz['dt']

        try:
            if saved_npz['history'] is None:
                raise Exception
            self.history = saved_npz['history'].reshape((1,))[0]
        except:
            self.new_history()
            if verbose:
                self.yell('no history file found, initialized empty history')

        if saved_npz['minimizer'] is None:
            self.minimizer = 'mu'
        else:
            self.minimizer = saved_npz['minimizer']

    def dump_profile_to_file(self, target_npz, verbose=True, **kwargs):
        if verbose:
            self.yell(f'dumping profile data to {target_npz}')

        np.savez(target_npz, psi=self.psi, Lx=self.Lx, Ly=self.Ly, 
                 mu=self.mu, eps=self.eps, history=self.history, 
                 dt=self.dt, minimizer=self.minimizer, **kwargs)

    def _evolve_noise(self):
        self.psi += self.dt * np.random.normal(0, self.noise_width, size=(self.Nx, self.Ny))

    def calc_N_tot(self):
        # total particle number = psi integrated over all volume
        return np.sum(np.real(self.psi)) * self.dx * self.dy

    # local chemical potential
    def calc_chemical_potential_density(self):
        D2psi = irfft2(-self.K2*rfft2(self.psi))
        D4psi = irfft2(self.K4*rfft2(self.psi))
        local_mu = (1-self.eps) * self.psi + self.psi**3 + 2*D2psi + D4psi

        return np.real(local_mu)

    # local grand potential density
    def calc_grand_potential_density(self):
        return self.calc_helmholtz_density() - self.mu * self.psi

    def calc_grand_potential(self):
        return np.sum(self.calc_grand_potential_density()) * self.dx * self.dy
    
    # local helmholtz free energy density
    def calc_helmholtz_density(self):
        psi_k = rfft2(self.psi)
        psi_k_o = self.Kernel * psi_k
        f = 1/2 * self.psi * irfft2(psi_k_o) + self.psi**4/4 - self.eps/2 * self.psi**2
        return np.real(f)

    # difference
    def calc_ord_param(self):
        return np.max(self.psi) - np.min(self.psi)
    
    # plot psi, chemical potential density, and grand potential density
    def plot(self, cmap='jet', plot_psi=True, plot_mu=True, plot_omega=True, lazy_factor=1):

        num_plots = plot_psi + plot_mu + plot_omega

        plt.figure(figsize=(12.8, 3.2*num_plots))
        index_plot = 1

        LF = lazy_factor

        if plot_psi:
            ax1 = plt.subplot(num_plots, 1, index_plot, title='$\\psi$')
            cm1 = ax1.pcolormesh(self.X[::LF], self.Y[::LF], np.real(self.psi)[::LF], cmap=cmap, vmin=-1, vmax=1, shading='nearest')
            plt.colorbar(cm1, ax=ax1, orientation='horizontal')
            ax1.set_aspect('equal', adjustable='box')

            index_plot += 1

        if plot_mu:

            ax2 = plt.subplot(num_plots, 1, index_plot, title='$\\mu(\\mathbf{r})-\\mu$')
            cm2 = ax2.pcolormesh(self.X[::LF], self.Y[::LF], self.calc_chemical_potential_density()[::LF] - self.mu, cmap=cmap, vmin=-0.2, vmax=0.2, shading='nearest')
            plt.colorbar(cm2, ax=ax2, orientation='horizontal')
            ax2.set_aspect('equal', adjustable='box')

            index_plot += 1

        if plot_omega:

            ax3 = plt.subplot(num_plots, 1, index_plot, title='$\\omega(\\mathbf{r})$')
            cm3 = ax3.pcolormesh(self.X[::LF], self.Y[::LF], self.calc_grand_potential_density()[::LF], cmap=cmap, vmin=-0.1, vmax=0.1, shading='nearest')
            plt.colorbar(cm3, ax=ax3, orientation='horizontal')
            ax3.set_aspect('equal', adjustable='box')

            index_plot += 1


        plt.tight_layout()
        plt.show(block=True)

    def _prepare_simulation(self, dt):
        self._exp_dt_kernel = np.exp(-dt*self.Kernel)
        self._exp_dt_eps = np.exp(dt*self.eps/2)
        self.dt = dt
        self.initialize_fftw()

        # used for nonlocal convserved
        self._psi_bar = np.mean(self.psi)
        self._exp_dt_kernel_conserved = self._exp_dt_kernel.copy()
        self._exp_dt_kernel_conserved[0,0] = 1.
        
    def _evolve_k(self):
        self.for_psi()
        self.psi_k *= self._exp_dt_kernel
        self.bac_psi()

    def _evolve_x(self):
        self.psi *= self._exp_dt_eps
        self.psi -= self.dt/2 * (self.psi**3 - self.mu)

    def _evolve_conserved_nonlocal(self):
        dt = self.dt

        #psi3 = self.psi**3
        #self.psi -= dt/2 * (psi3 - np.mean(psi3))
        #self.psi = (self.psi - self._psi_bar)*self._exp_dt_eps + self._psi_bar

        #self.for_psi()
        #self.psi_k *= self._exp_dt_kernel_conserved
        #self.bac_psi()

        #psi3 = self.psi**3
        #self.psi -= dt/2 * (psi3 - np.mean(psi3))
        #self.psi = (self.psi - self._psi_bar)*self._exp_dt_eps + self._psi_bar
        
        psi3k = rfft2(self.psi**3)

        self.for_psi()
        psik00 = self.psi_k[0,0]
        self.psi_k -= dt * psi3k
        self.psi_k *= self._exp_dt_eps**2
        self.psi_k *= self._exp_dt_kernel_conserved
        self.psi_k[0,0] = psik00
        self.bac_psi()
        self.psi = np.real(self.psi)

    #
    def summarize(self, display_precision=5):
        dp = display_precision
        self.yell(f'------------------------------------------------------------')
        self.yell(f'Summary:')
        self.yell(f'mu={self.mu}\t\teps={self.eps}')
        self.yell(f'Lx={self.Lx:.{dp}f}\t\tLy={self.Ly:.{dp}f}')
        self.yell(f'Nx={self.Nx}\t\tNy={self.Ny}')
        dpu = self.Nx * self.Ny / self.Volume * np.pi*4 * np.pi*4/np.sqrt(3)
        self.yell(f'DPUC={dpu:.{dp}f}')
        self.yell(f'minimizer={self.minimizer}\tdt={self.dt}')

        if len(self.history['t']) > 0:
            age = self.history['t'][-1]
            self.yell(f'age={age}')
        else:
            self.yell(f'age=not evolved yet')

        Omega = self.calc_grand_potential()
        omega = Omega / self.Volume
        psi0 = self.calc_N_tot() / self.Volume
        self.yell(f'Omega={Omega:.{dp}f}\tomega={omega:.{dp}f}\tpsi0={psi0:.{dp}f}')
        self.yell(f'------------------------------------------------------------')

    def new_lock(self):
        self.lock = threading.Lock()
        return self.lock

    def destroy_lock(self):
        self.lock = None

    def stop_minimization(self):
        self.running = False


    # minimize with constant mu
    def minimize_mu(self, dt, cycle=31, verbose=True, display_precision=5):
        i = 0
        t = 0
        if len(self.history['t']) > 0:
            t = self.history['t'][-1]

        self._prepare_simulation(dt)
        self.running = True

        dp = display_precision

        if verbose:
            self.summarize()
        
        while True:
            with self.lock:
                if not self.running:
                    return
                self._evolve_noise()
                self._evolve_x()
                self._evolve_k()
                self._evolve_x()
            t += dt
            i += 1

                                
            if i >= cycle:
                psi0 = self.calc_N_tot() / self.Volume
                omega = self.calc_grand_potential_density()
                Omega = np.sum(omega) * self.dx * self.dy
                ord_param = self.calc_ord_param()
                mu = np.mean(self.calc_chemical_potential_density())
                with self.lock:
                    if not self.running:
                        return
                    sys.stdout.write(
                        f'\r[pfcim] t={t:.{dp}f} | psi_bar={psi0:.{dp}f} | Omega={Omega:.{dp}f} | ' +\
                        f'omega={Omega/self.Volume:.{dp+2}f} | mu={mu:.{dp+2}f} | diff={ord_param:.{dp}f}       '
                    )

                self.history['Omega'] = np.append(self.history['Omega'], Omega)
                self.history['t'] = np.append(self.history['t'], t)
                self.history['psi0'] = np.append(self.history['psi0'], psi0)
                self.history['max-min'] = np.append(self.history['max-min'], ord_param)
                self.history['omega'] = np.append(self.history['omega'], Omega/self.Volume)
                i = 0


        print()

    # minimize with nonlocal conserved dynamics
    def minimize_nonlocal_conserved(self, dt, cycle=31, verbose=True, display_precision=5):
        i = 0
        t = 0
        if len(self.history['t']) > 0:
            t = self.history['t'][-1]

        self._prepare_simulation(dt)
        self.running = True

        dp = display_precision

        if verbose:
            self.summarize()
        
        while True:
            with self.lock:
                if not self.running:
                    return
                self._evolve_noise()
                self._evolve_conserved_nonlocal()
            t += dt
            i += 1
                                
            if i >= cycle:
                psi0 = self.calc_N_tot() / self.Volume
                omega = self.calc_grand_potential_density()
                Omega = np.sum(omega) * self.dx * self.dy
                helm = self.calc_helmholtz_density()
                Helm = np.sum(helm) * self.dx * self.dy
                ord_param = self.calc_ord_param()
                mu = np.mean(self.calc_chemical_potential_density())
                with self.lock:
                    if not self.running:
                        return
                    sys.stdout.write(
                        f'\r[pfcim] t={t:.{dp}f} | psi_bar={psi0:.{dp}f} | Omega={Omega:.{dp}f} | ' +\
                        f'omega={Omega/self.Volume:.{dp+2}f} | mu={mu:.{dp+2}f} | diff={ord_param:.{dp}f}       '
                    )

                self.history['Omega'] = np.append(self.history['Omega'], Omega)
                self.history['t'] = np.append(self.history['t'], t)
                self.history['psi0'] = np.append(self.history['psi0'], psi0)
                self.history['max-min'] = np.append(self.history['max-min'], ord_param)
                self.history['omega'] = np.append(self.history['omega'], Omega/self.Volume)
                i = 0

        print()

    def run_background(self, func, args, kwargs=dict()):
        if self.lock is None:
            self.new_lock()
        
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=False)
        thread.start()

        return self.lock, thread
    
    def copy(self, verbose=True, copy_history=True):
        other = PhaseFieldCrystal2D(1, 1, 1, 1, verbose=verbose)
        other.set_dimensions(self.Lx, self.Ly, self.Nx, self.Ny, verbose=verbose)
        other.set_params(self.mu, self.eps, verbose=verbose)
        other.set_psi(self.psi, verbose=verbose)

        other.dt = self.dt
        if copy_history:
            other.history = self.history.copy()
        else:
            other.new_history()

        return other

    def yell(self, s):
        print(f'[pfcim] {s}')


