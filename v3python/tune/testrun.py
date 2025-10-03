#!/usr/bin/env python

import sys
import importlib
import readline
from pathlib import Path
import traceback
import json
from dataclasses import asdict

class MaxIndentEncoder(json.JSONEncoder):
    MAX_INDENT_LEVEL = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._indent_unit = self.indent if isinstance(self.indent, str) else ' ' * self.indent if self.indent is not None else ''

    def encode(self, obj):
        return self._recursive_encode(obj, 0)

    def _recursive_encode(self, obj, current_indent_level, max_indent_delta=0):
        indent_str = self._indent_unit * current_indent_level

        if current_indent_level >= self.MAX_INDENT_LEVEL + max_indent_delta:
            # If max indent is reached, use compact representation.
            return json.dumps(obj, separators=(',', ':'))

        if isinstance(obj, dict):
            if not obj:
                return '{}'
            items = []
            for k, v in obj.items():
                key_str = self.default(k) if not isinstance(k, str) else json.dumps(k)
                # Hack
                max_indent_delta = 1 if k == 'adiffs' else 0
                value_str = self._recursive_encode(v, current_indent_level + 1, max_indent_delta)
                items.append(f"{indent_str}{self._indent_unit}{key_str}: {value_str}")
            return "{\n" + ",\n".join(items) + f"\n{indent_str}}}"
        elif isinstance(obj, (list, tuple)):
            if not obj:
                return '[]'
            items = []
            for item in obj:
                items.append(f"{indent_str}{self._indent_unit}{self._recursive_encode(item, current_indent_level + 1)}")
            return "[\n" + ",\n".join(items) + f"\n{indent_str}]"
        else:
            # Fall back to the default encoder for other types (primitives).
            return super().encode(obj)

def first(line, sep=" "):
    seps = line.split(sep, maxsplit=1)
    if len(seps) > 1:
        return seps
    return seps[0], None

class CommandProcessor(object):

    def __init__(self):
        self._all_commands = None
        self._module = None

    def _gen_all_commands(self):
        PFX = 'command_'
        for attr in dir(self):
            if attr.startswith(PFX):
                yield attr.removeprefix(PFX)

    @property
    def all_commands(self):
        if self._all_commands is None:
            self._all_commands = list(self._gen_all_commands())
        return self._all_commands

    def command_help(self, line):
        print("Available Commands:")
        for command in self._gen_all_commands():
            print("\t", command)

    def command_module(self, line):
        if line is None:
            return r'''Missing Arguments. Expect: module <package name>'''
        package, _ = first(line)
        try:
            module = importlib.import_module('.' + package, package='v3python.tune')
        except ImportError as e:
            traceback.print_exc()
            print(e, file=sys.stderr)
            return f'Package Error. Import package {package} error'
        self._module = module

    def command_probe(self, line):
        if line is None:
            return r'''Missing Arguments. Expect: probe <entry> <data directory>'''
        if self._module is None:
            return r'''Error: Need to run module <package name> first'''
        tune = self._module.TuneDesc()
        try:
            entry, odir = first(line)
            entry = tune.ENTRY_CLASS.parse_text(entry)
            odir = Path(odir)
        except Exception as e:
            traceback.print_exc()
            print(e, file=sys.stderr)
            return 'Error when parsing argument ' + line
        def gen():
            kernels = tune.list_kernels(entry)
            for k in kernels:
                yield k, tune.probe_backends(entry, k, odir)
        return dict(gen())

    def command_prepare_data(self, line):
        if line is None:
            return r'''Syntax Error: no package name. Expect: prepare_data <entry> <data directory>'''
        if self._module is None:
            return r'''Error: Need to run module <package name> first'''
        tune = self._module.TuneDesc()
        try:
            entry, odir = first(line)
            entry = tune.ENTRY_CLASS.parse_text(entry)
            odir = Path(odir)
            odir.mkdir(exist_ok=True)
        except Exception as e:
            traceback.print_exc()
            print(e, file=sys.stderr)
            return 'Error when parsing argument ' + line
        tune.prepare_data(entry, odir)

    def command_benchmark(self, line):
        if line is None:
            return r'''Syntax Error: no package name. Expect: benchmark <data directory> <kernel selector>'''
        if self._module is None:
            return r'''Error: Need to run module <package name> first'''
        try:
            data_dir, kernel = first(line)
            data_dir = Path(data_dir)
            impl_selector = self._module.ImplSelector.parse_text(kernel)
        except:
            return 'Error when parsing argument ' + tail
        if not (data_dir / 'entry.json').is_file():
            return f'{data_dir} is not valid data director. Missing entry.json file.'
        tune = self._module.TuneDesc()
        entry, impl_desc, adiffs, times, benchmark_input_metadata = tune.benchmark(data_dir, impl_selector)
        return {
            "entry": asdict(entry),
            "impl_selection": asdict(impl_selector),
            "impl_desc": impl_desc,
            "adiffs": adiffs,
            "times": times,
            "bim": asdict(benchmark_input_metadata),
        }

    def command_cleanup(self, line):
        odir = Path(line)
        if not odir.is_dir():
            return f"{line} is not a directory"
        sig = odir / "entry.json"
        if not sig.is_file():
            return f"{line}/entry.config does not exist"

    def process_input(self, line):
        command, tail = first(line)
        if command.lower() == 'exit':
            return 'exit'
        attr = 'command_' + command
        if not hasattr(self, attr):
            return f"Unknown Command {command}"
        return getattr(self, attr)(tail)

def main():
    cp = CommandProcessor()
    if sys.stdin.isatty():
        def gen_line():
            while True:
                try:
                    raw = input("> ")
                    line = raw.strip()
                    if line.lower() == "exit":
                        break
                    yield line
                except EOFError: # Ctrl+D
                    break
        def report_ok(ret):
            if ret is None:
                return
            print(json.dumps(ret, indent=2, cls=MaxIndentEncoder))
        def report_error(error):
            print(error)
    else:
        def gen_line():
            for raw in sys.stdin:
                line = raw.strip()
                if line.lower() == "exit":
                    break
                yield line
        def report_ok(ret):
            if ret is None:
                print('OK', flush=True)
            else:
                # print(f'{ret=}', file=sys.stderr)
                print('OK', json.dumps(ret), flush=True)
        def report_error(error):
            print("Error", flush=True)
            print(error, file=sys.stderr)
    for line in gen_line():
        ret = cp.process_input(line)
        if isinstance(ret, str):
            report_error(ret)
        else:
            report_ok(ret)

if __name__ == '__main__':
    main()
