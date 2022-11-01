from __future__ import annotations
import warnings
from typing import Optional, Union, Callable, List
from tqdm import tqdm

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import shutil

from michael960lib.common import IllegalActionError, scalarize
from torusgrid import fields as fd

from rich.progress import track

import torusgrid as tg

from .core.evolution import ConstantChemicalPotentialMinimizer, NonlocalConservedMinimizer, StressRelaxer, PFCMinimizer
from .core.base import PFCStateFunction
from .history import PFCHistory, PFCMinimizerHistoryBlock, PFCEditActionHistoryBlock, import_history

warnings.warn('the PFC module is currently unmaintained')

# matplotlib.use('TKAgg')
# matplotlib.style.use('fast')


class PFC:
    def __init__(self, field: fd.RealField2D):

        self.field = field 
        self.age = 0
        self.history= PFCHistory(self.field)

        self.history_pointer = 0
        self.current_minimizer = None

    def new_minimizer(self, minimizer: PFCMinimizer):
        try:
            assert isinstance(minimizer, PFCMinimizer)
            assert not minimizer.started
            assert not minimizer.ended
            assert not minimizer.history.is_committed()
            assert minimizer.age == 0
            assert minimizer.field is self.field
        except AssertionError:
            raise ValueError('invalid minimizer')

        self.current_minimizer = minimizer

    def new_mu_minimizer(self, dt, eps, mu):
        self.current_minimizer = ConstantChemicalPotentialMinimizer(self.field, dt, eps, mu)

    def new_nonlocal_minimizer(self, dt, eps):
        self.current_minimizer = NonlocalConservedMinimizer(self.field, dt, eps)

    def new_stress_relaxer(self, dt, eps, mu, expansion_rate=1):
        self.current_minimizer = StressRelaxer(self.field, dt, eps, mu)

    def evolve_multisteps(self, N_steps, N_epochs,
            display_format: Optional[str]=None,
            callbacks: List[Callable[[PFCMinimizer, PFCStateFunction], None]]=[]):

        if self.current_minimizer is None:
            raise tg.dynamics.MinimizerError(self.current_minimizer) 

        self.current_minimizer.set_display_format(display_format)
        self.current_minimizer.run_multisteps(N_steps, N_epochs, callbacks=callbacks)

        self.history_pointer += 1
        self.age += self.current_minimizer.age
        self.history.cut_and_insert(PFCMinimizerHistoryBlock(self.current_minimizer.history), self.history_pointer)
        self.current_minimizer = None

    def evolve_nonstop(self, N_steps, custom_keyboard_interrupt_handler=None,
            display_format: Optional[str]=None,
            callbacks: List[Callable[[PFCMinimizer, PFCStateFunction], None]]=[]):

        if self.current_minimizer is None:
            raise tg.dynamics.MinimizerError(self.current_minimizer) 

        self.current_minimizer.set_display_format(display_format)
        self.current_minimizer.run_nonstop(N_steps, custom_keyboard_interrupt_handler=custom_keyboard_interrupt_handler, 
                callbacks=callbacks)

        self.history_pointer += 1
        self.age += self.current_minimizer.age
        self.history.cut_and_insert(PFCMinimizerHistoryBlock(self.current_minimizer.history), self.history_pointer)
        self.current_minimizer = None

    def evolve(self,
        minimizer_supplier: Optional[Callable[[PFC], PFCMinimizer]]=None,
        minimizer: str=None, dt: float=None, eps: float=None, mu: Optional[float]=None,
        expansion_rate: Optional[float]=None,
        N_steps: int=31, N_epochs: Optional[int]=None,
        custom_keyboard_interrupt_handler: Optional[Callable[[PFCMinimizer], bool]]=None,
        display_format: Optional[str]=None, 
        callbacks: List[Callable[[PFCMinimizer, PFCStateFunction], None]]=[]):

        if not minimizer_supplier is None:
            if None not in (minimizer, dt, eps, mu, expansion_rate):
                warnings.warn('ignoring other arguments passed to evolve() when using minimizer supplier')

            minim = minimizer_supplier(self)
            self.new_minimizer(minim)

        else:
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
                    display_format=display_format, callbacks=callbacks)
        else:
            if N_epochs <=0:
                raise ValueError(f'N_epochs must be a positive integer')
            self.evolve_multisteps(N_steps, N_epochs, display_format=display_format,
                    callbacks=callbacks)

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
    
    def save(self, path: str):
        state = self.export()

        tmp_name = f'{path}.tmp.file'
        np.savez(tmp_name, state=state)
        shutil.move(f'{tmp_name}.npz', f'{path}.pfc')

    def save_hdf5(self, path):
        raise NotImplementedError
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


class PFCGroup:
    def __init__(self):
        self.models = dict()
    
    def put(self, model: PFC, name: str, attrs=dict()):
        if name in self.get_names():
            warnings.warn(f'key {name} already exists in group and will be overwritten')
        self.models[name] = {'model': model, 'attrs': attrs}

    def save(self, path):
        data = dict() 
        tmp_name = f'{path}.tmp.file'
        for name in self.models:
            model_state = self.models[name]['model'].export()
            attrs = self.models[name]['attrs']
            data[name] = {'model': model_state, 'attrs': attrs}
        np.savez(tmp_name, **data)
        shutil.move(f'{tmp_name}.npz', f'{path}.pfcgroup')

    def get_names(self) -> List[str]:
        return list(self.models.keys())

    def get_model(self, name) -> PFC:
        return self.models[name]['model']

    def get_attrs(self, name) -> dict:
        return self.models[name]['attrs']

    def get_models(self) -> list:
        l = []
        for key in self.models:
            l.append(self.models[key]['model'])
        return l

    def get_sorted_models(self, comparator: str, reverse: bool=False) -> List[PFC]:
        def get_item(model):

            try:
                val = {
                        'Lx': model.field.Lx,
                        'Ly': model.field.Ly,
                        'Nx': model.field.Nx,
                        'Ny': model.field.Ny
                }[comparator]
            except KeyError:
                raise ValueError(f'{comparator} is not a valid attribute')
            return val

        models = self.get_models()
        return sorted(models, key=get_item, reverse=reverse)


def load_pfc_group(path: str, show_progress: bool = True) -> PFCGroup:
    saved = np.load(path, allow_pickle=True)
    g = PFCGroup()
   
    
    if show_progress:
        files = track(saved.files, description='Loading PFC models')
    else:
        saved.files
    
    for name in files:
        try:
            data = scalarize(saved[name])
            model_state = data['model']
            model = import_pfc_model(model_state)
            attrs = data['attrs']
            g.put(model, name, attrs=attrs)
        except Exception:
            print(f'error occured while loading {name}')
    return g



