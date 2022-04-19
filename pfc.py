from .base import pfc_base, field as fd
from .base.common import IllegalActionError
from .history import PFCHistory, PFCMinimizerHistoryBlock, PFCEditActionHistoryBlock, import_history

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np

#_default_colors = ['blue', 'red', 'mediumseagreen', 'magenta', 'dodgerblue', 'limegreen', 'darkslategrey', 'orange']
_default_colors = ['steelblue', 'darkseagreen', 'palevioletred']

class PFC:
    def __init__(self, field: fd.RealField2D):
        matplotlib.use('TKAgg')
        matplotlib.style.use('fast')

        self.field = field 
        self.age = 0
        self.history= PFCHistory(self.field)

        self.history_pointer = 0
        self.current_minimizer = None

    def new_mu_minimizer(self, dt, eps, mu):
        self.current_minimizer = pfc_base.ConstantChemicalPotentialMinimizer(self.field, dt, eps, mu)

    def new_nonlocal_minimizer(self, dt, eps):
        self.current_minimizer = pfc_base.NonlocalConservedMinimizer(self.field, dt, eps)

    def evolve(self, N_steps, N_epochs):
        if self.current_minimizer is None:
            raise fd.MinimizerError(self.current_minimizer) 

        self.current_minimizer.run_multisteps(N_steps, N_epochs)

        self.history_pointer += 1
        self.age += self.current_minimizer.age
        self.history.cut_and_insert(PFCMinimizerHistoryBlock(self.current_minimizer.history), self.history_pointer)
        self.current_minimizer = None

    def evolve_nonstop(self, N_steps, custom_keyboard_interrupt_handler=None):
        if self.current_minimizer is None:
            raise fd.MinimizerError(self.current_minimizer) 

        self.current_minimizer.run_nonstop(N_steps, custom_keyboard_interrupt_handler)

        self.history_pointer += 1
        self.age += self.current_minimizer.age
        self.history.cut_and_insert(PFCMinimizerHistoryBlock(self.current_minimizer.history), self.history_pointer)
        self.current_minimizer = None

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
    
    def export(self):
        state = dict()
        state['history'] = self.history.export()
        state['age'] = self.age
        state['history_pointer'] = self.history_pointer
        state['field'] = self.field.export_state()

        return state



def import_pfc_model(state: dict) -> PFC:
    history_state = state['history']
    field_state = state['field']
    pfc_history = import_history(history_state) 
    field = fd.import_field(field_state)
    pfc_model = PFC(field)

    pfc_model.age = state['age']
    pfc_model.history_pointer = state['history_pointer']
    pfc_model.history = import_history(state['history'])

    return pfc_model



def load_pfc_model(path: str) -> PFC:
    saved = np.load(path, allow_pickle=True)
    return import_pfc_model(saved)


   
