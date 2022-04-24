import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2
from pprint import pprint
import pyfftw
import time
import threading
import sys

from michael960lib.math import fourier

import warnings

warnings.warn('the pfc module is deprecated, use pfc_im instead')

class PhaseFieldCrystal2D:
    def __init__(self, Lx, Ly, Nx, Ny):

        self.set_dimensions(Lx, Ly, Nx, Ny, reset_psi=True)
        self.set_psi(0)
        self.set_params(0, 0)
        self.set_noise(0, 0)

        self.lock = None


    def set_noise(self, seed, width):

        self.seed = seed
        np.random.seed(self.seed)

        self.noise_width = width

    def set_params(self, mu, eps):
        self.mu = mu
        self.eps = eps
        print(f'[pfc] new params set, mu={mu} eps={eps}')

    def set_dimensions(self, Lx, Ly, Nx, Ny, reset_psi=True):

        self.Lx = Lx
        self.Ly = Ly
        self.Volume = Lx*Ly

        self.Nx = Nx
        self.Ny = Ny
        
        x, kx, self.dx, self.dkx = fourier.generate_xk(Lx, Nx)
        y, ky, self.dy, self.dky = fourier.generate_xk(Ly, Ny)
        
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        self.Kx, self.Ky = np.meshgrid(kx, ky, indexing='ij')
        self.K2 = self.Kx**2 + self.Ky**2
        self.K4 = self.K2**2

        self.Kernel = 1-2*self.K2+self.K4
        
        print(f'[pfc] new dimensions set, Lx={Lx} Ly={Ly} Nx={Nx} Ny={Ny}')

        if reset_psi:
            self.psi = np.zeros((Nx, Ny))
            print(f'[pfc] resetted psi')



   
    def set_psi(self, psi1, verbose=True):
        self.psi[:,:] = psi1
        if verbose:
            print('[pfc] new psi set')

    def load_profile_from_file(self, saved_npz):
        file_list = saved_npz.files
        print(f'[pfc] loading npz file, found the following data:')
        print(f'[pfc] {file_list}')

        psi = saved_npz['psi']

        self.set_dimensions(saved_npz['Lx'], saved_npz['Ly'], *psi.shape)
        self.set_params(saved_npz['mu'], saved_npz['eps'])
        self.set_psi(saved_npz['psi'])
        

    def _evolve_k(self, dt):
        psik = fft2(self.psi)
        psik *= np.exp(-dt*self.Kernel)
        self.psi = np.real(ifft2(psik))

    def _evolve_x(self, dt):
        self.psi *= np.exp(-dt * (-self.eps + self.psi**2))
        self.psi += dt * self.mu

    def _evolve_conserved_nonlocal(self, dt):
        psi3k = fft2(self.psi**3)
        psik = fft2(self.psi)

        psik00 = psik[0,0]
        psik *= np.exp(-dt*(self.Kernel-self.eps))
        psik += -dt * psi3k
        psik[0,0] = psik00
        self.psi = np.real(ifft2(psik))


    def _evolve_noise(self, dt):
        self.psi += dt * np.random.normal(0, self.noise_width, size=(self.Nx, self.Ny))

    # total particl number = psi integrated over all volume
    def calc_N_tot(self):
        return np.sum(np.real(self.psi)) * self.dx * self.dy

    # local chemical potential
    def calc_chemical_potential_density(self):
        D2psi = ifft2(-self.K2*fft2(self.psi))
        D4psi = ifft2(self.K4*fft2(self.psi))
        local_mu = (1-self.eps) * self.psi + self.psi**3 + 2*D2psi + D4psi

        return np.real(local_mu)

    # local grand potential density
    def calc_grand_potential_density(self):
        return self.calc_helmholtz_density() - self.mu * self.psi

    def calc_grand_potential(self):
        return np.sum(self.calc_grand_potential_density()) * self.dx * self.dy
    
    # local helmholtz free energy density
    def calc_helmholtz_density(self):
        psi_k = fft2(self.psi)
        psi_k_o = self.Kernel * psi_k
        f = 1/2 * self.psi * ifft2(psi_k_o) + self.psi**4/4 - self.eps/2 * self.psi**2
        return np.real(f)

    #
    def calc_ord_param(self):
        return np.max(self.psi) - np.min(self.psi)

    
    # plot psi, chemical potential density, and grand potential density
    def plot(self, cmap='jet', plot_psi=True, plot_mu=True, plot_omega=True, lazy_factor=1):
        plt.figure(figsize=(12.8, 9.6))
        
        num_plots = plot_psi + plot_mu + plot_omega
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


    def new_lock(self):
        self.lock = threading.Lock()
        return self.lock

    # minimize with constant mu
    def minimize_mu(self, dt, cycle=31, verbose=True, t0=0):
        i = 0
        t = t0
        if verbose:

            print(f'[pfc] ------------------------------------------------------------')
            print(f'[pfc] minimization parameter summary:')
            print(f'[pfc] mu={self.mu} eps={self.eps}')
            print(f'[pfc] Lx={self.Lx} Ly={self.Ly}')
            print(f'[pfc] Nx={self.Nx} Ny={self.Ny}')
            print(f'[pfc] noise_seed={self.seed} noise_amplitude={self.noise_width}')
            print(f'[pfc] dt={dt}')
            print(f'[pfc] ------------------------------------------------------------')
            print(f'[pfc] minimizing grand potential with constant mu={self.mu}')

        while True:
            with self.lock:
                self._evolve_noise(dt)
                self._evolve_x(dt/2)
                self._evolve_k(dt)
                self._evolve_x(dt/2)


            t += dt
            i += 1

            if i >= cycle:

                psi0 = self.calc_N_tot() / self.Volume
                omega = self.calc_grand_potential_density()
                Omega = np.sum(omega) * self.dx * self.dy
                ord_param = self.calc_ord_param()

                i = 0
                sys.stdout.write(f'\r[pfc] t={t:.4f} | psi_bar={psi0:.5f} | Omega={Omega:.5f} | diff={ord_param:.5f}    ')
        print()

    # minimize with nonlocal conserved dynamics
    def minimize_nonlocal_conserved(self, dt, cycle=31, verbose=True, t0=0):
        i = 0
        t = t0
        if verbose:
            psi0 = self.calc_N_tot() / self.Volume
            print(f'[pfc] minimizing grand potential with nonlocal conserved dynamics')
            print(f'[pfc] psi_bar={psi0}')

        while True:
            with self.lock:
                self._evolve_noise(dt)
                self._evolve_conserved_nonlocal(dt)


            t += dt
            i += 1

            if i >= cycle:

                psi0 = self.calc_N_tot() / self.Volume
                omega = self.calc_grand_potential_density()
                Omega = np.sum(omega) * self.dx * self.dy
                ord_param = self.calc_ord_param()

                i = 0
                sys.stdout.write(f'\r[pfc] t={t} | psi_bar={psi0} | Omega={Omega} | diff={ord_param}    ')
        print()




    def run_background(self, func, args):
        if self.lock is None:
            self.new_lock()
        
        thread = threading.Thread(target=func, args=args, daemon=True)
        thread.start()

        return self.lock, thread


class PhaseFieldCrystalExperimental(PhaseFieldCrystal2D):
    def __init__(self, Lx, Ly, Nx, Ny):
        super().__init__(Lx, Ly, Nx, Ny)
        
        self.set_energy_coeff(1, 0)

    def set_energy_coeff(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def _evolve_x(self, dt):
        self.psi *= np.exp(-dt * (-self.eps + self.p1*self.psi**2 + self.p2*self.psi**4))
        self.psi += dt * self.mu


    def calc_chemical_potential_density(self):
        D2psi = ifft2(-self.K2*fft2(self.psi))
        D4psi = ifft2(self.K4*fft2(self.psi))
        local_mu = (1-self.eps) * self.psi + self.p1*self.psi**3 + self.p2*self.psi**5 + 2*D2psi + D4psi

        return np.real(local_mu)


    # local grand potential density
    def calc_grand_potential_density(self):
        return self.calc_helmholtz_density() - self.mu * self.psi
    
    # local helmholtz free energy density
    def calc_helmholtz_density(self):
        psi_k = fft2(self.psi)
        psi_k_o = self.Kernel * psi_k
        f = 1/2 * self.psi * ifft2(psi_k_o) + self.p1*self.psi**4/4 + self.p2*self.psi**6/6 - self.eps/2 * self.psi**2
        return np.real(f)

    # plot psi, chemical potential density, and grand potential density
    def plot(self, cmap='jet'):
        plt.figure(figsize=(12.8, 9.6))
        ax1 = plt.subplot(311, title='$\\psi$')
        ax2 = plt.subplot(312, title='$\\mu(\\mathbf{r})-\\mu$')
        ax3 = plt.subplot(313, title='$\\omega(\\mathbf{r})$')

        cm1 = ax1.pcolormesh(self.X, self.Y, np.real(self.psi), cmap=cmap, shading='nearest')
        ax1.set_aspect('equal', adjustable='box')

        print(np.min(self.psi))
        print(np.max(self.psi))
        cm2 = ax2.pcolormesh(self.X, self.Y, self.calc_chemical_potential_density() - self.mu, cmap=cmap, shading='nearest')
        ax2.set_aspect('equal', adjustable='box')

        cm3 = ax3.pcolormesh(self.X, self.Y, self.calc_grand_potential_density(), cmap=cmap, vmin=-0.1, vmax=0.1, shading='nearest')
        ax3.set_aspect('equal', adjustable='box')

        plt.colorbar(cm1, ax=ax1, orientation='horizontal')
        plt.colorbar(cm2, ax=ax2, orientation='horizontal')
        plt.colorbar(cm3, ax=ax3, orientation='horizontal')
        plt.tight_layout()
        plt.show(block=True)



