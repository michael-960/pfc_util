from .base import pfc_base, field as fd
from .base.common import IllegalActionError
from .history import PFCHistory, PFCMinimizerHistoryBlock, PFCEditActionHistoryBlock
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

        self.fef = None
        self.current_minimizer = None
        self.history_pointer = 0

    def set_eps(self, eps):
        self.fef = pfc_base.PFCFreeEnergyFunctional(eps)

    def new_mu_minimizer(self, dt, eps, mu):
        self.set_eps(eps)
        self.current_minimizer = pfc_base.ConstantChemicalPotentialMinimizer(self.field, dt, eps, mu)
        #self.current_minimizer.set_age(self.age)

    def new_nonlocal_minimizer(self, dt, eps):
        self.set_eps(eps)
        self.current_minimizer = pfc_base.NonlocalConservedMinimizer(self.field, dt, eps)
        #self.current_minimizer.set_age(self.age)

    def evolve(self, N_steps, N_epochs):
        if self.current_minimizer is None:
            raise fd.MinimizerError(self.current_minimizer) 

        self.current_minimizer.run_multisteps(N_steps, N_epochs)

        self.history_pointer += 1
        self.age += self.current_minimizer.age
        self.history.cut_and_insert(PFCMinimizerHistoryBlock(self.current_minimizer.history), self.history_pointer)


    def evolve_nonstop(self, N_steps, custom_keyboard_interrupt_handler=None):
        if self.current_minimizer is None:
            raise fd.MinimizerError(self.current_minimizer) 

        self.current_minimizer.run_nonstop(N_steps, custom_keyboard_interrupt_handler)

        self.history_pointer += 1
        self.age += self.current_minimizer.age
        self.history.cut_and_insert(PFCMinimizerHistoryBlock(self.current_minimizer.history), self.history_pointer)


    def field_snapshot(self):
        return self.field.export_state()     

    def plot_history(self, *item_names, show=True):
        return self.history.plot(*item_names, start=0, end=self.history_pointer, show=show)

    def save(self, path):
        raise NotImplementedError()
    
    def export(self):
        pass
    
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


def load_model(path):
    saved = np.load(path, allow_pickle=True)
    raise NotImplementedError()


   
