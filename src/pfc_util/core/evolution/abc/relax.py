from abc import abstractmethod
from typing import List, Optional, final

import numpy as np
import numpy.typing as npt
from torusgrid.dynamics import SplitStep, Step


from torusgrid.fields import RealField2D


class StressRelaxerBase(SplitStep[RealField2D]):
    '''
    A base class for stress relaxers.
    Subclasses must implement: 
        - get_realspace_steps()
        - step_kernel()
        - on_size_changed()
    '''
    def __init__(self, 
            field: RealField2D, 
            dt: float, *,
            expansion_rate: float=1., resize_cycle: int = 31
            ):
        '''
        Stress relaxer.
        This minimizer follows the following procedure to find the field values
        and size that minimizes the mean free energy **density**:
            
            x scale, y scale = (1, 1)
            Lx, Ly = field.Lx, field.Ly

            {
                [
                    - update field values & x, y scales by dt
                ] (repeat Q times)

                - field.Lx, field.Ly *= scales
                - x, y scales = (1, 1)
            } (repeat)

        '''
        super().__init__(field, dt)

        self.f = 1.

        self.dT = self.dt * expansion_rate # / 2

        self.fx = 1.
        self.fy = 1.

        self.Lx = field.Lx
        '''self.Lx is equivalent to field.Lx * self.fx'''
        self.Ly = field.Lx
        '''self.Ly is equivalent to field.Ly * self.fy'''

        self._domega_kernels = np.zeros(
                (5, *field.psi_k.shape)
                , dtype=np.float_)
        '''
            First & second derivative of mean grand potential density w.r.t
            system size, used to relax stress.

            Shape: (5, Nx, Ny)
            0: domega / dLx
            1: domega / dLy
            2: d2omega / dLx2
            3: d2omega / dLy2
            4: d2omega / dLxdLy
        '''


        self.NN = field.Nx * field.Ny
        '''
        Number of points in the field, used for convolution
        '''
        self._2_NN = 2 / self.NN

        # if not field.fft_initialized():
        #     field.initialize_fft()

        self.resize_cycle = resize_cycle
        self._resize_counter = 0


    def on_start(self, n_steps: int, n_epochs: Optional[int]):
        self.on_size_changed()

    def get_kspace_steps(self) -> List[Step]:
        def relax_stress_(dt: float):
            self.update_domega_kernels()
            self.relax_stress(dt)
            self.Lx, self.Ly = self.grid.Lx*self.fx, self.grid.Ly*self.fy
            self.on_size_changed()

        return [self.step_kernel, relax_stress_]

    def step(self):

        super().step()

        self._resize_counter += 1
        if self._resize_counter >= self.resize_cycle:
            self.resize_field()
            self._resize_counter = 0

    
    @abstractmethod
    def step_kernel(self, dt: float):
        '''
        Minimization for linear terms
        This method is called twice during step():
        Once before relax_stress() and once after. Both are invoked with self.dt/2
        '''
        raise NotImplementedError


    @abstractmethod
    def update_domega_kernels(self):
        '''
        This method is called immediately before relax_stress().
        The purpose is to calculate the derivatives of mean grand potential density
        w.r.t to system size.

        self._domega_kernels should be updated here.
        '''
        raise NotImplementedError

    @abstractmethod
    def on_size_changed(self):
        '''
        This method is called immediately after relax_stress() or resize_field()
        Its purpose is to update variables that depend on the system size such
        as Kx, Ky, K2 etc 
        '''
        raise NotImplementedError


    def relax_stress(self, dt: float): 
        '''
        Stress-relaxing. This method should update self.fx and self.fy (scale factors).
        The default behavior is to use self._domega_kernels to compute dfx and
        dfy
        '''
        f = self.grid

        omega_list = self.real_convolution_2d(np.abs(f.psi_k**2), self._domega_kernels)

        dLx = -omega_list[0]*dt + (omega_list[0] * omega_list[2] + omega_list[1] * omega_list[4]) * dt**2/2
        dLy = -omega_list[1]*dt + (omega_list[1] * omega_list[3] + omega_list[0] * omega_list[4]) * dt**2/2

        dfx = dLx / f.Lx
        dfy = dLy / f.Ly

        self.fx += dfx
        self.fy += dfy

    
    def resize_field(self):
        '''
        This method is called every [Q==resize_cycle] steps. 
        The field is resized to Lx==Lx0*fx, Ly==Ly0*fy, and (fx, fy) are reset
        to (1, 1).
        '''
        Lx0, Ly0 = self.grid.size
        self.grid.set_size(Lx0*self.fx, Ly0*self.fy)
        self.fx = 1.
        self.fy = 1.
        self.on_size_changed()


    @final
    def real_convolution_2d(self, 
            psi_k_sq: npt.NDArray[np.float_],
            kernel: npt.NDArray[np.float_]
        ) -> npt.NDArray[np.float_]:
        '''
        Given |psi_k^2| 
        where psi_k (Nx, Ny/2+1)
        is the real Fourier transform of real field psi (Nx, Ny), calculate
        the inner product <psi K psi> where K = kernel.

        The kernel K is given as an array of shape (Nx, Ny/2+1) in the k space.
        '''
        r = kernel * psi_k_sq
        r[:,:,[0,-1]] *= 0.5
        return r.sum(axis=(1,2)) * self._2_NN

