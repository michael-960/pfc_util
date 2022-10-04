from ..toolkit import profile_editor as pe
from ..toolkit import ortho_lattice_generator as olg
from ..legacy import pfc_im
from matplotlib import pyplot as plt
import numpy as np
import sys
import subprocess
from michael960lib import common as cmn
from michael960lib.common import overrides
import os
import re
import readline
import numexpr
from .prompt import Command, CommandExecutionError, SimplePrompt
import time

_version = '2.6'
_date = '2022/04/13'
_author = 'michael'



class ProfileEditorPrompt(SimplePrompt):
    def __init__(self):
        super().__init__()
        self.prefix = '[PIPE]>> '
        self.add_command(CommandHelp())

        self.add_command(CommandLs())
        self.add_command(CommandListModels())

        self.add_command(CommandLoad())
        self.add_command(CommandSave())

        self.add_command(CommandView())
        self.add_command(CommandInfo())

        self.add_command(CommandExtend())
        self.add_command(CommandCrop())
        self.add_command(CommandChangeRes())
        self.add_command(CommandCopy())
        self.add_command(CommandDelete())
        self.add_command(CommandMove())

        self.add_command(CommandSetParam())
        self.add_command(CommandSetSize())
        self.add_command(CommandSetMinimizer())

        self.add_command(CommandInterface())
        self.add_command(CommandLiquefy())
        self.add_command(CommandInsert())
        self.add_command(CommandFlip())
        self.add_command(CommandRotate())
        self.add_command(CommandConcat())

        self.add_command(CommandExit())

        self.add_command(CommandHistory())
        self.add_command(CommandIPython())
        self.add_command(CommandResetHistory())
        self.add_command(CommandEvolve())

        self.memory_models = dict()

        try:
            readline.read_history_file('.history')
        except FileNotFoundError as e:
            self.output('new command history file initialized in ./.history')

        readline.set_history_length(256)

    @overrides(SimplePrompt)
    def put_banner(self):
        self.output('PFC Interactive Profile Editor (PIPE)')
        self.output(f'version: {_version}')
        self.output(f'last updated on {_date} by {_author}')
        print()
    
    @overrides(SimplePrompt)
    def output(self, s, level=0):
        color = ''
        if level == 0:
            color = cmn.bcolors.OKCYAN
        elif level == 1:
            color = cmn.bcolors.WARNING
        elif level == 2:
            color = cmn.bcolors.FAIL
        print(f'{cmn.bcolors.BOLD}{color}[PIPE]{cmn.bcolors.ENDC} {s}')


    def get(self, varname) -> pfc_im.PhaseFieldCrystal2D:
        try:
            return self.memory_models[varname]
        except KeyError:
            s = f'model {varname} not found'
            raise CommandExecutionError(s)

    def put(self, varname, model):
        self.check_varname(varname)
        self.memory_models[varname] = model

    @overrides(SimplePrompt)
    def _on_exit(self):
        readline.write_history_file('.history')
        return


    def check_varname(self, varname: str):
        blacklist = [
            varname.find('-') != -1,
            varname.find('+') != -1,
            varname.find('*') != -1,
            varname.find('/') != -1,
            varname.find('.') != -1,
            varname.find('!') != -1,
            varname.find('@') != -1,
            varname.find('(') != -1,
            varname.find(')') != -1,
            varname.find('#') != -1,
            varname.find('^') != -1,
            varname.find('&') != -1,
            varname.find('$') != -1,
            varname.find('%') != -1,
            varname[0].isnumeric()
        ]
        if any(blacklist):
            raise CommandExecutionError(f'invalid variable name: {varname}')


class CommandHelp(Command):
    def __init__(self):
        super().__init__('help')

    @overrides(Command)
    def execute(self, prompt: SimplePrompt, args: list):
        prompt.output('PFC profile editor, please enter one of the following commands')
        prompt.output(prompt.commands_str)
        print()

    @overrides(Command)
    def complete(self, prompt: SimplePrompt, args: list):
        return [] 


class CommandLs(Command):
    def __init__(self):
        super().__init__('ls')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        #prompt.output('listing files')
        p = subprocess.run(['ls'] + args)
        print()
        
    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) <= 1:
            return super().complete(prompt, args)
        return [] 


class CommandListModels(Command):
    def __init__(self):
        super().__init__('list-models')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        s = ''
        for md in prompt.memory_models:
            s = s + md + ' '
        prompt.output(f'{s}')
        print()
        
    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        return []


class CommandLoad(Command):
    def __init__(self):
        super().__init__('load')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 2:
            raise CommandExecutionError('syntax: load FILENAME VARNAME')
        filename = args[0]
        varname = args[1]

        try:
            saved = np.load(filename, allow_pickle=True)
        except FileNotFoundError:
            raise CommandExecutionError(f'file \'{filename}\' not found')

        model = pfc_im.PhaseFieldCrystal2D(1, 1, 1, 1, verbose=False)
        model.load_profile_from_file(saved, verbose=False)

        prompt.put(varname, model)
        prompt.output(f'loaded file {filename} to model {varname}')
        print()

        
    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1:
            return super().complete(prompt, args)
        if len(args) == 2:
            return _match_models(prompt, args[-1])
        return [] 


class CommandSave(Command):
    def __init__(self):
        super().__init__('save')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 2:
                raise CommandExecutionError('syntax: save VARNAME FILENAME')

        varname = args[0]
        filename = args[1]
        
        model = prompt.get(varname)
        model.dump_profile_to_file(filename, verbose=False)

        prompt.output(f'saved model {varname} to file {filename}')
        print()
                
    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1:
            return _match_models(prompt, args[-1])
        if len(args) == 2:
            return super().complete(prompt, args)
        return [] 


class CommandView(Command):
    def __init__(self):
        super().__init__('view')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 1:
            raise CommandExecutionError('syntax: view VARNAME')

        varname = args[0]
        extra = args[1:]
        i = 0
        params = {'lazy': 1}
        try:
            while i < len(extra):
                if extra[i] == '--lazy':
                    assert i+1 < len(extra)
                    params['lazy'] = int(extra[i+1])
                    i += 1
                i += 1
        except AssertionError:
            prompt.output('error occured while parsing arguments:', level=1)
            prompt.output(f'{extra}', level=1)

        model = prompt.get(varname)

        prompt.output('invoking pfc model methods')
        model.plot(lazy_factor=params['lazy'], plot_psi=True, plot_mu=False, plot_omega=False)
        print()
                
    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1:
            return _match_models(prompt, args[-1])
        if len(args) == 2:
            return ['--lazy ']

        return [] 


class CommandInfo(Command):
    def __init__(self):
        super().__init__('info')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 1: 
            raise CommandExecutionError('syntax: info VARNAME [options]')
        varname = args[0]
        extra = args[1:]
        params = {'less': False, 'display-precision': 5}
        # parse extra args
        i = 0
        try:
            while i < len(extra):
                if extra[i] == '--less':
                    params['less'] = True
                elif extra[i] == '--display-precision':
                    assert i+1 < len(extra)
                    params['display-precision'] = int(extra[i+1])
                    i += 1
                i += 1

        except AssertionError:
            raise CommandExecutionError('error ocurred while parsing arguments')
        except ValueError as e:
            raise CommandExecutionError(e.args[-1])



        model = prompt.get(varname)
        dp = params['display-precision']
        mu = model.mu
        eps = model.eps
        Lx = model.Lx
        Ly = model.Ly
        Nx = model.Nx
        Ny = model.Ny
        Omega = model.calc_grand_potential()
        omega = Omega / model.Volume

        if params['less']:
            prompt.output(f'mu={mu}\teps={eps}')
            prompt.output(f'Lx={Lx:.{dp}f}\tLy={Ly:.{dp}f}\tNx={Nx}\tNy={Ny}')
            prompt.output(f'Omega={Omega:.{dp}f}\tomega={omega:.{dp}f}')
        else:
            model.summarize(display_precision=dp)
        print()
                
    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1:
            return _match_models(prompt, args[-1])
        if len(args) >= 1:
            if args[-1] == '--display-precision':
                return []
            return ['--display-precision ', '--less ']
        return [] 


class CommandExtend(Command):
    def __init__(self):
        super().__init__('extend')


    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 4:
            raise CommandExecutionError('syntax: extend VAR1 X-RATIO Y-RATIO VAR2')

        var1 = args[0]
        try:
            Mx = int(args[1])
            My = int(args[2])
        except ValueError as e:
            raise CommandExecutionError(e.args[-1])
        var2 = args[3]
        extra = args[4:]

        model1 = prompt.get(var1)
        model2 = model1.copy(verbose=False)

        psi2 = pe.periodic_extend(model1.psi, Mx, My)
        model2.set_dimensions(model1.Lx*Mx, model1.Ly*My, model1.Nx*Mx, model1.Ny*My, verbose=False)
        model2.set_psi(psi2, verbose=False)

        prompt.put(var2, model2)
        print()

               
    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1 or len(args) == 4:
            return _match_models(prompt, args[-1])
        return [] 


class CommandCrop(Command):
    def __init__(self):
        super().__init__('crop')


    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 6:
            raise CommandExecutionError('syntax: crop VAR1 ratio_x1 ratio_y1 ratio_x2 ratio_y2 VAR2')

        var1 = args[0]
        var2 = args[5]
        try:
            ratio_x1 = float(args[1])
            ratio_y1 = float(args[2])
            ratio_x2 = float(args[3])
            ratio_y2 = float(args[4])

            assert 0 <= ratio_x1 < ratio_x2 <= 1
            assert 0 <= ratio_y1 < ratio_y2 <= 1
            
        except ValueError as e:
            raise CommandExecutionError(e.args[-1])
        except AssertionError as e:
            raise CommandExecutionError('the ratios must satisfy the constraints ' +\
                                        '0 <= ratio_x1 < ratio_x2 <= 1 and ' +\
                                        '0 <= ratio_y1 < ratio_y2 <= 1')


        extra = args[6:]

        model1 = prompt.get(var1)
        model2 = model1.copy(verbose=False, copy_history=False)

        Nx2 = int(model1.Nx * (ratio_x2 - ratio_x1))
        Ny2 = int(model1.Ny * (ratio_y2 - ratio_y1))
        first_x = int(model1.Nx * ratio_x1)
        first_y = int(model1.Nx * ratio_y1)

        psi2 = model1.psi[first_x:first_x+Nx2, first_y:first_y+Ny2]

        model2.set_dimensions(model1.Lx*(ratio_x2-ratio_x1), model1.Ly*(ratio_y2-ratio_y1), Nx2, Ny2, verbose=False)
        model2.set_psi(psi2, verbose=False)
        prompt.put(var2, model2)
        print()

               
    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1 or len(args) == 6:
            return _match_models(prompt, args[-1])
        if 1 < len(args) < 6:
            return ['0.125 ', '0.25 ', '0.375 ', '0.5 ', '0.625 ', '0.75 ', '0.875 ', '1']
        return [] 


class CommandChangeRes(Command):
    def __init__(self):
        super().__init__('change-res')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 4:
            raise CommandExecutionError('syntax: change-res VAR1 NX NY VAR2')

        var1 = args[0]
        try:
            Nx = int(args[1])
            Ny = int(args[2])
        except ValueError as e:
            raise CommandExecutionError(e.args[-1])

        var2 = args[3]
        extra = args[4:]

        model1 = prompt.get(var1)
        model2 = model1.copy(verbose=False)


        psi2 = pe.change_resolution(model1.psi, Nx, Ny)

        model2.set_dimensions(model1.Lx, model1.Ly, Nx, Ny, verbose=False)
        model2.set_psi(psi2, verbose=False)

        prompt.put(var2, model2)
        print()


    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1 or len(args) == 4:
            return _match_models(prompt, args[-1])
        return [] 


class CommandCopy(Command):
    def __init__(self):
        super().__init__('copy')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 2:
            raise CommandExecutionError('syntax: copy VAR1 VAR2')

        var1 = args[0]
        var2 = args[1]

        model1 = prompt.get(var1)
        model2 = model1.copy(verbose=False)

        prompt.put(var2, model2)
        print()


    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1 or len(args) == 2:
            return _match_models(prompt, args[-1])
        return [] 


class CommandDelete(Command):
    def __init__(self):
        super().__init__('del')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 1:
            raise CommandExecutionError('syntax: delete VAR')

        var = args[0]

        if not var in prompt.memory_models:
            raise CommandExecutionError(f'{var} does not exist in memory')

        try:
            del prompt.memory_models[var]
        except Exception:
            raise CommandExecutionError(f'could not delete {var} from memory')

        print()


    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1:
            return _match_models(prompt, args[-1])
        return [] 


class CommandMove(Command):
    def __init__(self):
        super().__init__('move')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 2:
            raise CommandExecutionError('syntax: move VAR1 VAR2')

        var1 = args[0]
        var2 = args[1]

        if not var1 in prompt.memory_models:
            raise CommandExecutionError(f'{var1} does not exist in memory')

        prompt.put(var2, prompt.get(var1))
        del prompt.memory_models[var1]

        print()


    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1 or len(args) == 2:
            return _match_models(prompt, args[-1])
        return [] 


class CommandSetParam(Command):
    def __init__(self):
        super().__init__('set-param')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 4:
            raise CommandExecutionError('syntax: set-param VAR1 MU EPS VAR2')

        var1 = args[0]
        try:
            mu2 = float(args[1])
            eps2 = float(args[2])
        except ValueError as e:
            raise CommandExecutionError(e.args[-1])
        var2 = args[3]

        extra = args[4:]

        model1 = prompt.get(var1)
        model2 = model1.copy(verbose=False)

        model2.set_params(mu2, eps2, verbose=False)

        prompt.put(var2, model2)
        print()


    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1 or len(args) == 4:
            return _match_models(prompt, args[-1])
        return [] 


class CommandSetSize(Command):
    def __init__(self):
        super().__init__('set-size')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 4:
            raise CommandExecutionError('syntax: set-size VAR1 LX LY VAR2')

        var1 = args[0]
        try:
            if args[1].endswith('ux'):
                Lx = float(args[1][:-2]) * 4*np.pi
            elif args[1].endswith('uy'):
                Lx = float(args[1][:-2]) * 4*np.pi / np.sqrt(3)
            else:
                Lx = float(args[1])

            if args[2].endswith('ux'):
                Ly = float(args[2][:-2]) * 4*np.pi
            elif args[2].endswith('uy'):
                Ly = float(args[2][:-2]) * 4*np.pi / np.sqrt(3)
            else:
                Ly = float(args[2])

        except ValueError as e:
            raise CommandExecutionError(e.args[-1])

        var2 = args[3]

        model1 = prompt.get(var1)
        model2 = model1.copy(verbose=False)
        model2.resize(Lx, Ly, verbose=False)

        prompt.put(var2, model2)
        print()


    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1 or len(args) == 4:
            return _match_models(prompt, args[-1])
        return []


class CommandSetMinimizer(Command):
    def __init__(self):
        super().__init__('set-minimizer')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 4:
            raise CommandExecutionError('syntax: set-minimizer VAR1 minimizer dt VAR2')

        var1 = args[0]
        try:
            minimizer = args[1]
            dt = float(args[2])
            assert minimizer in ['mu', 'nonlocal']
        except ValueError as e:
            raise CommandExecutionError(e.args[-1])
        except AssertionError as e:
            raise CommandExecutionError('minimizer must be either \'mu\' or \'nonlocal\'')


        var2 = args[3]
        extra = args[4:]

        model1 = prompt.get(var1)
        model2 = model1.copy(verbose=False)
        model2.minimizer = minimizer
        model2.dt = dt

        prompt.put(var2, model2)
        print()


    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1 or len(args) == 4:
            return _match_models(prompt, args[-1])
        if len(args) == 2:
            return ['mu', 'nonlocal']
        return [] 


class CommandInterface(Command):
    def __init__(self):
        super().__init__('interface')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 4:
            raise CommandExecutionError('syntax: interface VAR1 VAR2 WIDTH OUTPUT')

        var_sol = args[0]
        var_liq = args[1]
        var_out = args[3]
        try:
            width = float(args[2])
        except ValueError as e:
            raise CommandExecutionError(e.args[-1])

        extra = args[4:]

        model_sol = prompt.get(var_sol)
        model_liq = prompt.get(var_liq)

        try:
            assert abs(model_sol.Lx - model_liq.Lx) / model_sol.Lx < 1e-6
            assert abs(model_sol.Ly - model_liq.Ly) / model_sol.Ly < 1e-6
            assert model_sol.Nx == model_liq.Nx
            assert model_sol.Ny == model_liq.Ny

        except AssertionError as e:
            raise CommandExecutionError(f'models {var_sol} and {var_liq} do not have the same dimensions')

        model_out = model_sol.copy(verbose=False, copy_history=False)

        X, Y = model_sol.X, model_sol.Y
        xa = model_sol.Lx / 4
        xb = model_sol.Lx / 4 * 3

        bump = (1+np.tanh((X-xa)/width))/2 * (1+np.tanh((-X+xb)/width))/2
        psi_out = model_sol.psi * bump + model_liq.psi * (1-bump)
        model_out.set_psi(psi_out, verbose=False)

        prompt.put(var_out, model_out)
        print()


    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1 or len(args) == 2 or len(args) == 4:
            return _match_models(prompt, args[-1])
        return [] 


class CommandLiquefy(Command):
    def __init__(self):
        super().__init__('liquefy')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 2:
            raise CommandExecutionError('syntax: liquefy VAR1 VAR2 [--density psi0]')

        var = args[0]
        var_out = args[1]
        extra = args[2:]
        
        model = prompt.get(var)
        param = {'density': model.calc_N_tot()/model.Volume}

        i = 0
        while i < len(extra):
            if extra[i] == '--density':
                try:
                    assert len(extra) > i+1
                except AssertionError as e:
                    raise CommandExecutionError('please specify a density after --density')
                param['density'] = float(extra[i+1])
                i += 1
            else:
                raise CommandExecutionError(f'unknown option: {extra[i]}')

            i += 1


        model_out = model.copy(verbose=False)

        model_out.set_psi(param['density'] + 0*model.X, verbose=False)

        prompt.put(var_out, model_out)
        print()


    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1 or len(args) == 2:
            return _match_models(prompt, args[-1])
        if len(args) == 3:
            return ['--density ']
        return [] 


class CommandInsert(Command):
    def __init__(self):
        super().__init__('insert')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 3:
            raise CommandExecutionError(
                'syntax: insert SMALL LARGE VAR [--position left|right|top|bottom|center] [--offset X Y]')

        var_s = args[0]
        var_l = args[1]
        var_out = args[2]
        extra = args[3:]
        
        model_s = prompt.get(var_s)
        model_l = prompt.get(var_l)
        param = {'position': 'center', 'offset': (0,0), 'width': 0}

        if not (model_s.Lx <= model_l.Lx and model_s.Ly <= model_l.Ly):
            raise CommandExecutionError('the first model must have smaller dimensions than the second one')


        # parse extra arguments
        i = 0
        while i < len(extra):
            if extra[i] == '--position':
                if len(extra) <= i+1:
                    raise CommandExecutionError('please specify a position after --position')
                if not extra[i+1] in ['left', 'right', 'top' 'bottom', 'center']:
                    raise CommandExecutionError(f'invalid position: {extra[i+1]}')
                param['position'] = extra[i+1]
                i += 1

            elif extra[i] == '--offset':
                if len(extra) <= i+2:
                    raise CommandExecutionError('please specify x and y offsets after --offset')
                try:
                    param['offset'] = (float(extra[i+1]), float(extra[i+2]))
                except ValueError as e:
                    raise CommandExecutionError(f'invalid offset: {extra[i+1]} {extra[i+2]}')
                i += 2

            else:
                raise CommandExecutionError(f'unkonw option: {extra[i]}')

            i += 1
        
        psi_small = model_s.psi.copy()
        psi_large = model_l.psi.copy()

        box_Nx, box_Ny = int(model_s.Lx / model_l.Lx * model_l.Nx), int(model_s.Ly / model_l.Ly * model_l.Ny)
        
        psi_small_modified = pe.change_resolution(psi_small, box_Nx, box_Ny)
       
        box_X, box_Y = int(model_l.Nx/2 - box_Nx/2), int(model_l.Ny/2 - box_Ny/2)
        if param['position'] == 'left':
            box_X = 0
            
        if param['position'] == 'right':
            box_X = model_l.Nx - box_Nx

        if param['position'] == 'top':
            box_Y = 0
            
        if param['position'] == 'bottom':
            box_Y = model_l.Ny - box_Ny

        psi_large[box_X:box_X+box_Nx, box_Y:box_Y+box_Ny] = psi_small_modified
        

        model_out = model_l.copy(verbose=False, copy_history=False)
        model_out.set_psi(psi_large, verbose=False)

        prompt.put(var_out, model_out)
        print()


    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1 or len(args) == 2 or len(args) == 3:
            return _match_models(prompt, args[-1])
        if len(args) >= 4:
            if args[-2] == '--position':
                return ['top', 'bottom', 'left', 'right', 'center']
            if args[-2] == '--offset':
                return []
            return ['--position', '--offset']
        return [] 

class CommandFlip(Command):
    def __init__(self):
        super().__init__('flip')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 3:
            raise CommandExecutionError(
                'syntax: flip VAR [X|Y] VAROUT')

        var = args[0]
        axis = args[1]
        var_out = args[2]
        extra = args[3:]
        
        model = prompt.get(var)

        model_out = model.copy(verbose=False)

        if axis == 'Y':
            model_out.set_psi(model.psi[::-1,:], verbose=False)
        elif axis == 'X':
            model_out.set_psi(model.psi[:,::-1], verbose=False)
        else:
            raise CommandExecutionError('axis must be either X or Y')

        prompt.put(var_out, model_out)
        print()


    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1 or len(args) == 3:
            return _match_models(prompt, args[-1])
        if len(args) >= 2:
            return ['X', 'Y']
        return []

class CommandRotate(Command):
    def __init__(self):
        super().__init__('rotate')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 3:
            raise CommandExecutionError(
                'syntax: rotate VAR [90|180|270] VAROUT')

        var = args[0]
        angle = args[1]
        var_out = args[2]
        extra = args[3:]
        
        model = prompt.get(var)
        model_out = model.copy(verbose=False)

        if angle == '90':
            model_out.set_dimensions(model.Ly, model.Lx, model.Ny, model.Nx, verbose=False)
            model_out.set_psi(np.transpose(model.psi)[:,::-1], verbose=False)
        elif angle == '180':
            model_out.set_psi(model.psi[::-1,::-1], verbose=False)
        elif angle == '270':
            model_out.set_dimensions(model.Ly, model.Lx, model.Ny, model.Nx, verbose=False)
            model_out.set_psi(np.transpose(model.psi)[::-1,:], verbose=False)
        else:
            raise CommandExecutionError('axis must be 90, 180, or 270')

        prompt.put(var_out, model_out)
        print()

    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1 or len(args) == 3:
            return _match_models(prompt, args[-1])
        if len(args) == 2:
            return ['X', 'Y']
        return []



class CommandConcat(Command):
    def __init__(self):
        super().__init__('concat')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 4:
            raise CommandExecutionError(
                'syntax: concat VAR1 VAR2 [horizontal|vertical] VAROUT')

        var1 = args[0]
        var2 = args[1]
        concat_direction = args[2]
        var_out = args[3]
        extra = args[4:]
        
        model1 = prompt.get(var1)
        model2 = prompt.get(var2)

        model_out = model1.copy(verbose=False, copy_history=False)

        if concat_direction == 'vertical':
            if model1.Nx != model2.Nx or model1.Lx != model2.Lx:
                raise CommandExecutionError(f'models {var1} and {var2} do not have the same width')
            model_out.set_dimensions(model1.Lx, model1.Ly+model2.Ly, model1.Nx, model1.Ny+model2.Ny, verbose=False)
            psi_out = np.zeros(shape=(model1.Nx, model1.Ny+model2.Ny))
            psi_out[:,:model1.Ny] = model1.psi[:,:]
            psi_out[:,model1.Ny:] = model2.psi[:,:]

            model_out.set_psi(psi_out)

        elif concat_direction == 'horizontal':
            if model1.Ny != model2.Ny or model1.Ly != model2.Ly:
                raise CommandExecutionError(f'models {var1} and {var2} do not have the same height')
            model_out.set_dimensions(model1.Lx+model2.Lx, model1.Ly, model1.Nx+model2.Nx, model1.Ny, verbose=False)
            psi_out = np.zeros(shape=(model1.Nx+model2.Nx, model1.Ny)) 
            psi_out[:model1.Nx,:] = model1.psi[:,:]
            psi_out[model1.Nx:,] = model2.psi[:,:]
            model_out.set_psi(psi_out)

        else:
            raise CommandExecutionError(f'conact direction must be either horizontal or vertical')

        prompt.put(var_out, model_out)
        print()

    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1 or len(args) == 2 or len(args) == 4:
            return _match_models(prompt, args[-1])
        if len(args) == 3:
            return ['horizontal', 'vertical']
        return []


class CommandEvolve(Command):
    def __init__(self):
        super().__init__('evolve')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 1:
            raise CommandExecutionError('syntax: evolve VAR [options]')

        var = args[0]
        extra = args[1:]

        params = {'lazy': 1, 'display-precision': 5}
        
        i = 0
        try:
            while i < len(extra):
                if extra[i] == '--lazy-plot':
                    assert i+1 < len(extra)
                    params['lazy'] = int(extra[i+1])
                    i += 1
                if extra[i] == '--display-precision':
                    assert i+1 < len(extra)
                    params['display-precision'] = int(extra[i+1])
                i += 1
        except AssertionError:
            prompt.output('error occured while parsing arguments:', level=1)
            prompt.output(f'{extra}', level=1)

        model = prompt.get(var)


        lock, thread = None, None
        if model.minimizer == 'mu':
            lock, thread = model.run_background(model.minimize_mu, (model.dt,), 
                                                {'display_precision': params['display-precision']})
        if model.minimizer == 'nonlocal':
            lock, thread = model.run_background(model.minimize_nonlocal_conserved, (model.dt,), 
                                                {'display_precision': params['display-precision']})

        while True:
            try:
                time.sleep(1)

            except KeyboardInterrupt:
                with lock:
                    print()
                    prompt.output('------- pfc_im minimizer interrupted --------')
                    prompt.output('0: plot profile')
                    prompt.output('3: continue')
                    prompt.output('8: end minimization')
                    print('---------------------------------------------')

                    resp = input('enter action: ')

                    if resp in ['0']:
                        model.plot(lazy_factor=params['lazy'])

                    if resp in ['8']:
                        model.stop_minimization()
                        print()
                        return

    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1:
            return _match_models(prompt, args[-1])
        if len(args) >= 2:
            if args[-2] == '--lazy-plot' or args[-2] == '--display-precision':
                return [] 
            return ['--lazy-plot ', '--display-precision ']
        return [] 


class CommandHistory(Command):
    def __init__(self):
        super().__init__('history')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 1:
            raise CommandExecutionError('syntax: history VAR')

        var = args[0]
        extra = args[1:]

        model = prompt.get(var)
        ax1 = plt.subplot(211, title='history - grand potential density', xlabel='t', ylabel=r'$\omega$')
        ax2 = plt.subplot(212, title='history - mean density', xlabel='t', ylabel=r'$\bar\psi$')

        try:
            ax1.plot(model.history['t'], model.history['omega'], color='red')
            ax2.plot(model.history['t'], model.history['psi0'], color='black')
        except KeyError as e:
            raise CommandExecutionError(f'history file of {var} is corrupted, consider reseting history')

        plt.tight_layout()
        plt.show()

        print()


    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1:
            return _match_models(prompt, args[-1])
        return [] 


class CommandResetHistory(Command):
    def __init__(self):
        super().__init__('reset-history')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) < 1:
            raise CommandExecutionError('syntax: reset-history VAR')

        var = args[0]
        extra = args[1:]

        model = prompt.get(var)
        model.new_history()

        print()


    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        if len(args) == 1:
            return _match_models(prompt, args[-1])
        return [] 


class CommandIPython(Command):
    ipython_session = '.pipe_ipython_session'
    ipython_startup = '.pipe_ipython_startup.py'
    def __init__(self):
        super().__init__('ipython3')
        os.remove(self.ipython_session)
        open(self.ipython_session, 'a').close()

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        candidates = [f'ipython3 --no-banner -i {self.ipython_startup}', 'python3', 'ipython', 'python']
        terminal_width = 80
        try:
            terminal_width = os.get_terminal_size().columns
        except:
            pass


        #prompt.output(f'launching {candidates[0]} ...')
        print(cmn.bcolors.OKGREEN + '┌' + '─' * (terminal_width//4*3) + '┐' + cmn.bcolors.ENDC)
        os.system(candidates[0])

        print(cmn.bcolors.OKGREEN + '└' + '─' * (terminal_width//4*3) + '┘' + cmn.bcolors.ENDC)
        #prompt.output(f'terminating {candidates[0]} ...')
        print()
        return


         


    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        return [] 


class CommandExit(Command):
    def __init__(self):
        super().__init__('exit')

    @overrides(Command)
    def execute(self, prompt: ProfileEditorPrompt, args: list):
        prompt.running = False
                
    @overrides(Command)
    def complete(self, prompt: ProfileEditorPrompt, args: list):
        return [] 



def _match_models(prompt, arg):
    return [f'{mod}' for mod in prompt.memory_models.keys() if mod.startswith(arg)]


