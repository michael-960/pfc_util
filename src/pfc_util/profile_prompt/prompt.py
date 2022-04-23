#!/opt/anaconda3/envs/turnbull/bin/python3
from ..legacy import pfc_im
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
import subprocess
import readline
from util import common as cmn
import os
import re
import readline


_version = '2.2'
RE_SPACE = re.compile('.*\s+$', re.M)



class SimplePrompt(object):
    def __init__(self):
        self.commands_obj = dict()
        self.commands_str = ''
        self.running = False
        self.prefix = '>'

        readline.parse_and_bind('set show-all-if-ambiguous on')
        readline.parse_and_bind('set colored-stats on')
        readline.parse_and_bind('tab: menu-complete')


        readline.set_completer_delims(' \t\n;')
        readline.set_completer(self.complete)



    def add_command(self, command):
        self.commands_obj[command.name] = command
        self.commands_str = ', '.join((f'{cmd}' for cmd in self.commands_obj.keys()))

    def consume(self):
        cmd_str = input(f'{self.prefix }')
        self.handle_command(cmd_str)

    def handle_command(self, cmd_str: str):
        cmd_and_args = cmd_str.split()
        
        if len(cmd_and_args) == 0:
            return

        cmd = cmd_and_args[0]
        args = cmd_and_args[1:]

        if cmd not in self.commands_obj:
            self.output(f'command {cmd} not found, possible commands are', level=-1)
            self.output(self.commands_str, level=-1)
            print()
            return
        
        self.execute_command(cmd, args)

    def execute_command(self, cmd, args):
        try:
            cmd = self.commands_obj[cmd]
            cmd.execute(self, args)
            return

        except CommandExecutionError as e:
            self.output(e.message, 2)
            print()

            
    def start(self):
        self.put_banner()
        self.running = True
        while self.running:
            try:
                self.consume()
            except KeyboardInterrupt:
                print()
                self.output('enter exit to leave', level=1)
                print()
                pass
        self.output('terminating prompt')
        self._on_exit()

    def put_banner(self):
        pass

    # 0: ok, 1: warning, else: error
    def output(self, s, level=0):
        if level == 0:
            print(f'{cmn.bcolors.BOLD}{cmn.bcolors.OKCYAN}[]{cmn.bcolors.ENDC} {s}')
        elif level == 1:
            print(f'{cmn.bcolors.BOLD}{cmn.bcolors.WARNING}[]{cmn.bcolors.ENDC} {s}')
        else:
            print(f'{cmn.bcolors.BOLD}{cmn.bcolors.FAIL}[]{cmn.bcolors.ENDC} {s}')

    def complete(self, text, state):
        # Generic readline completion entry point
        buff = readline.get_line_buffer()
        line = readline.get_line_buffer().split()

        # show all commands
        if not line:
            return [c + ' ' for c in self.commands_obj.keys()][state]

        # account for last argument ending in a space
        if RE_SPACE.match(buff):
            line.append('')

        # resolve command to the implementation function
        cmd = line[0].strip()
        if cmd in self.commands_obj.keys():
            args = line[1:]
            if args:
                candidates = self.commands_obj[cmd].complete(self, args)
                result_list = [can for can in candidates if can.startswith(args[-1])] + [None]
                
                return result_list[state]
            return [cmd + ' '][state]


        # incomplete command
        results = [c + ' ' for c in self.commands_obj.keys() if c.startswith(cmd)] + [None]
        return results[state]

    def _on_exit(self):
        return



class CommandExecutionError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)


class Command:
    def __init__(self, name):
        self.name = name
        self._execute = lambda args: None
        self._completer = None
        self._prompt = None

    def execute(self, prompt, args):
        self._execute(prompt, args)

    def complete(self, prompt, args):
        return _complete_path(args)



def _listdir(root):
    "List directory 'root' appending the path separator to subdirs."
    res = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            name += os.sep
        res.append(name)
    return res

def _complete_path(args):
    path = '.'    
    if not args:
        path = '.'
    else:
        path = args[-1]

    "Perform completion of filesystem path."
    if not path:
        return _listdir('.')

    dirname, rest = os.path.split(path)
    tmp = dirname if dirname else '.'
    res = [os.path.join(dirname, p)
            for p in _listdir(tmp) if p.startswith(rest)]

    # more than one match, or single match which does not exist (typo)
    if len(res) > 1 or not os.path.exists(path):
        return res
    # resolved to a single directory, so return list of files below it
    if os.path.isdir(path):
        return [os.path.join(path, p) for p in _listdir(path)]
    # exact file match terminates this completion
    return [path + ' ']


