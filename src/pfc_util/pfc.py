from torusgrid import fields as fd
from .core.evolution import ConstantChemicalPotentialMinimizer, NonlocalConservedMinimizer, StressRelaxer, PFCMinimizer
from michael960lib.common import IllegalActionError, scalarize
from .history import PFCHistory, PFCMinimizerHistoryBlock, PFCEditActionHistoryBlock, import_history

from typing import Optional, Union, Callable
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
import warnings





class PFC:
    def __init__(self, field: fd.RealField2D):
        matplotlib.use('TKAgg')
        matplotlib.style.use('fast')

        self.field = field 
        self.age = 0
        self.history= PFCHistory(self.field)

        self.history_pointer = 0
        self.current_minimizer = None

    def new_minimizer(self, minimizer: PFCMinimizer):
        self.current_minimizer = minimizer

    def new_mu_minimizer(self, dt, eps, mu):
        self.current_minimizer = ConstantChemicalPotentialMinimizer(self.field, dt, eps, mu)

    def new_nonlocal_minimizer(self, dt, eps):
        self.current_minimizer = NonlocalConservedMinimizer(self.field, dt, eps)

    def new_stress_relaxer(self, dt, eps, mu, expansion_rate=1):
        self.current_minimizer = StressRelaxer(self.field, dt, eps, mu)

    def evolve_multisteps(self, N_steps, N_epochs, display_precision: int=7):
        if self.current_minimizer is None:
            raise fd.MinimizerError(self.current_minimizer) 

        self.current_minimizer.set_display_precision(display_precision)
        self.current_minimizer.run_multisteps(N_steps, N_epochs)

        self.history_pointer += 1
        self.age += self.current_minimizer.age
        self.history.cut_and_insert(PFCMinimizerHistoryBlock(self.current_minimizer.history), self.history_pointer)
        self.current_minimizer = None

    def evolve_nonstop(self, N_steps, custom_keyboard_interrupt_handler=None, display_precision: int=7):
        if self.current_minimizer is None:
            raise fd.MinimizerError(self.current_minimizer) 


        self.current_minimizer.set_display_precision(display_precision)
        self.current_minimizer.run_nonstop(N_steps, custom_keyboard_interrupt_handler, display_precision=display_precision)

        self.history_pointer += 1
        self.age += self.current_minimizer.age
        self.history.cut_and_insert(PFCMinimizerHistoryBlock(self.current_minimizer.history), self.history_pointer)
        self.current_minimizer = None

    def evolve(self, minimizer: str, dt: float, eps: float, mu: Optional[float]=None,
               N_steps: int=31, N_epochs:Optional[int]=None,
               custom_keyboard_interrupt_handler: Optional[Callable[[PFCMinimizer], bool]]=None,
               expansion_rate: Optional[float]=None,
               display_precision: int=5):

        if not minimizer in ['mu', 'nonlocal', 'relax']:
            raise ValueError(f'{minimizer} is not a valid minimizer')

        if N_steps <= 0:
            raise ValueError(f'N_steps must be a positive integer')
        

        if minimizer == 'mu':
            if mu is None:
                raise ValueError(f'chemical potential must be specified with constant chemical potential minimizer')

            if not (expansion_rate is None):
                warnings.warn(f'expansion rate will be ignored for constant chemical potential minimizer')

            self.new_mu_minimizer(dt, eps, mu)
        if minimizer == 'nonlocal':
            if not (mu is None):
                warnings.warn(f'chemical potential will be ignored for nonlocal conserved minimizer')
            if not (expansion_rate is None):
                warnings.warn(f'expansion rate will be ignored for nonlocal conserved minimizer')

            self.new_nonlocal_minimizer(dt, eps)

        if minimizer == 'relax':
            if mu is None:
                raise ValueError(f'chemical potential must be specified with constant mu stress relaxer')
            if expansion_rate is None:
                raise ValueError(f'expansion rate must be specified with constant mu stress relaxer')

            self.new_stress_relaxer(dt, eps, mu, expansion_rate=expansion_rate)


        if N_epochs is None:
            self.evolve_nonstop(N_steps, custom_keyboard_interrupt_handler=custom_keyboard_interrupt_handler,
                    display_precision=display_precision)
        else:
            if N_epochs <=0:
                raise ValueError(f'N_epochs must be a positive integer')
            self.evolve_multisteps(N_steps, N_epochs, display_precision=display_precision)

    def field_snapshot(self):
        return self.field.export_state()     

    def plot_history(self, *item_names, show=True):
        if len(item_names) == 0:
            item_names = ['f', 'psibar']
        return self.history.plot(*item_names, start=0, end=self.history_pointer, show=show)
    
    def undo(self):
        if self.history_pointer <= 0:
            self.history_pointer = 0
            raise IllegalActionError('history is already at the oldest state')

        self.history_pointer -= 1
        field_state = self.history.get_block(self.history_pointer).get_final_field_state()
        self.field.set_psi(field_state['psi'])
        self.field.set_size(field_state['Lx'], field_state['Ly'])
    
    def redo(self):
        if self.history_pointer >= len(self.history.blocks)-1:
            self.history_pointer = len(self.history.blocks)-1
            raise IllegalActionError('history is already at the newest state')

        self.history_pointer += 1
        field_state = self.history.get_block(self.history_pointer).get_final_field_state()
        self.field.set_psi(field_state['psi'])
        self.field.set_size(field_state['Lx'], field_state['Ly'])
    
    def export(self) -> dict:
        state = dict()
        state['history'] = self.history.export()
        state['age'] = self.age
        state['history_pointer'] = self.history_pointer
        state['field'] = self.field.export_state()

        return state
    
    def save(self, path):
        state = self.export()
        np.savez(path, state=state)

    def save_hdf5(self, path):
        state = self.export()


def import_pfc_model(state: dict) -> PFC:
    history_state = state['history']
    field_state = state['field']

    pfc_history = import_history(history_state) 
    field = fd.import_field(field_state)

    pfc_model = PFC(field)

    pfc_model.age = state['age']
    pfc_model.history_pointer = state['history_pointer']
    pfc_model.history = pfc_history 

    return pfc_model


def load_pfc_model(path: str) -> PFC:
    data = np.load(path, allow_pickle=True)
    return import_pfc_model(scalarize(data['state']))


