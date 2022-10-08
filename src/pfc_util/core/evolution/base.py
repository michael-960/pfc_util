import numpy as np

from torusgrid.fields import RealField2D
from torusgrid.dynamics import TemporalEvolver

from ..base import FreeEnergyFunctional, StateFunction


class MinimizerMixin(TemporalEvolver[RealField2D]):
    '''
    A mix-in class for PFC minimizers.
    '''
    def init_pfc_variables(self, eps: float):
        '''
        Setup variables related to PFC
        '''

        self.eps = eps
        self.fef = FreeEnergyFunctional(eps)

        # self.history = PFCMinimizerHistory()

        self.info = dict()
        self.info['system'] = 'pfc'
        self.info['eps'] = eps

        self.display_format = '[{label}] f={f:.5f} F={F:.5f} psibar={psibar:.5f}'

    def start(self) -> None:
        super().start()
        self.data.update(self.info)

    def initialize_fft(self):
        '''
        Initialize field's fft plans if not already initialized
        '''
        if not self.field.fft_initialized():
            self.field.initialize_fft()

    def get_state_function(self):
        psibar = np.mean(self.field.psi)
        f = self.fef.mean_free_energy_density(self.field)
        F = self.fef.free_energy(self.field)
        return StateFunction(self.field.Lx, self.field.Ly, f, F, psibar)

    def get_evolver_state(self):
        return {'age': self.age}

    @property
    def field(self):
        '''
        self.field is just an alias for self.subject
        '''
        return self.subject


class MuMinimizerMixin(MinimizerMixin):
    '''
    A mix-in class for constant chemical potential minimizers.
    '''
    def init_pfc_variables(self, eps: float, mu: float):
        super().init_pfc_variables(eps)
        self.mu = mu
        self.info['mu'] = mu
        self.display_format = '[{label}] f={f:.5f} F={F:.5f} psibar={psibar:.5f} omega={omega:.5f} Omega={Omega:.5f}'

    def get_state_function(self):
        psibar = np.mean(self.field.psi)
        f = self.fef.mean_free_energy_density(self.field)
        F = self.fef.free_energy(self.field)
        omega = self.fef.mean_grand_potential_density(self.field, self.mu)
        Omega = self.fef.grand_potential(self.field, self.mu)

        return StateFunction(self.field.Lx, self.field.Ly, f, F, psibar, omega, Omega)

    def get_evolver_state(self):
        return {'age': self.age}


# class PFCMinimizerHistory:
#     def __init__(self):
#         super().__init__()
#         self.t: List[float] = []
#         self.age = 0.
#
#         self.committed = False
#         self.state_functions: List[PFCStateFunction] = []
#
#     def append_state_function(self, evolver_state: dict, sf: PFCStateFunction):
#         if self.committed:
#             raise ModifyingReadOnlyObjectError(
#             f'history object (label=\'{self.label}\') is already committed and hence not editable', self)
#
#         t = evolver_state['age']
#         if t < self.age:
#             raise IllegalActionError(f'time={t} is smaller than the current recorded time')
#
#         self.t.append(t)
#         self.state_functions.append(sf)
#         self.age = t
#
#     def get_t(self):
#         return self.t
#
#     def export(self) -> dict:
#         if self.committed:
#             state = dict()
#             state['age'] = self.age
#             state['label'] = self.label
#             state['final_field_state'] = self.final_field_state
#             state['state_functions'] = [sf.export() for sf in self.state_functions]
#             state['t'] = self.t
#             return state
#         else:
#             raise IllegalActionError(
#             'history object (label=\'{self.label}\') has not been committed and is therefore not ready to be exported')
#
#
