#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path
import traceback
from dataclasses import asdict
import argparse
from aotriton.pgdef.exaid import CommandProcessor as _CommandProcessor, first, serve
from .defaults import set_default_device

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('module', default=None, nargs='?')
    p.add_argument('--gpu', default=0, type=int)
    return p.parse_args()

def _load_module(module_name):
    from .registry import load_tune_module
    return load_tune_module(module_name)

class CommandProcessor(_CommandProcessor):
    """`testrun` never maintains or enforces tuning-level state (Revision
    note 3): it has no notion of "this process is a kernel process" anywhere.
    The library constraint (op-level tuning needs the testing library,
    kernel-level needs the tuning library) is enforced upstream, at task
    routing (`task_queue.tuning_level` + the SQL fetch filter, F16) -- by the
    time a task reaches this process its library is already correct for
    whatever it will ask for. `get_impl()` (via `TuneDesc`) still surfaces a
    clear error if a name is nonetheless unresolvable in this process's
    library; `command_probe` below reports such errors per-impl rather than
    aborting, since an interactive user must retain full access to inspect
    anything, never be told "you may not ask that"."""

    def __init__(self, module_name):
        super().__init__()
        if module_name is not None:
            self._module = _load_module(module_name)
        else:
            self._module = None

    def command_module(self, line):
        if line is None:
            return r'''Missing Arguments. Expect: module <package name>'''
        package, _ = first(line)
        try:
            module = _load_module(package)
        except ImportError as e:
            traceback.print_exc()
            print(e, file=sys.stderr)
            return f'Package Error. Import package {package} error'
        self._module = module

    def command_prepare_data(self, line):
        if line is None:
            return r'''Syntax Error: Expect: prepare_data <entry> <odir> [im_text ...]'''
        if self._module is None:
            return r'''Error: Need to run module <package name> first'''
        tune = self._module.TuneDesc()
        try:
            entry, tail = first(line)
            odir, tail = first(tail)
            entry = tune.ENTRY_CLASS.parse_text(entry)
            odir = Path(odir)
            odir.mkdir(parents=True, exist_ok=True)
            extra_ims = []
            while tail:
                im_text, tail = first(tail)
                extra_ims.append(tune.INPUT_METADATA.parse_text(im_text))
        except Exception as e:
            traceback.print_exc()
            print(e, file=sys.stderr)
            return 'Error when parsing argument ' + line
        # Clear GPU cache before prepare_data
        import torch
        torch.cuda.empty_cache()
        tune.prepare_data(entry, odir, extra_ims=extra_ims)

    def command_probe(self, line):
        """probe <data directory> [arch] [tuning_level filter]

        With no filter, attempts EVERY DSL name `list_impls()` returns and
        reports a per-impl error (rather than aborting the whole command) for
        any that fail to resolve -- an interactive user at the REPL must get
        the impls this process's library CAN serve back, plus a legible
        reason for the ones it cannot, not a lost result set from one
        ImportError (Revision note 3). `exaid.probe()` passes the task's
        tuning_level as this filter so a worker only probes what its
        container can serve -- filtering happens here, at the call, before
        any wrong-library import is attempted; never by a caller
        post-filtering this method's output.
        """
        if line is None:
            return r'''Missing Arguments. Expect: probe <data directory> [arch] [tuning_level filter]'''
        if self._module is None:
            return r'''Error: Need to run module <package name> first'''
        tune = self._module.TuneDesc()
        try:
            data_dir, tail = first(line)
            arch, tail = first(tail) if tail else (None, None)
            tuning_level_filter, tail = first(tail) if tail else (None, None)
            data_dir = Path(data_dir)
        except Exception as e:
            return f'Error when parsing argument {line!r}: {e}'
        if not (data_dir / 'entry.json').is_file():
            return f'{data_dir} is not valid data director. Missing entry.json file.'
        arch = arch or None
        tuning_level_filter = tuning_level_filter or None
        entry = tune.get_entry(data_dir)
        names = tune.list_impls(entry, arch=arch)
        if tuning_level_filter:
            names = [n for n in names
                     if self._module.ImplSelector.split_dsl_name(n)[0] == tuning_level_filter]
        def gen():
            for name in names:
                try:
                    yield name, tune.probe_all_impls(data_dir, name)
                except ImportError as e:
                    yield name, {'error': str(e)}
        return dict(gen())

    def command_benchmark(self, line):
        if line is None:
            return r'''Syntax Error: no package name. Expect: benchmark <data directory> <kernel selector>'''
        if self._module is None:
            return r'''Error: Need to run module <package name> first'''
        try:
            data_dir, kernel = first(line)
            data_dir = Path(data_dir)
            impl_selector = self._module.ImplSelector.parse_text(kernel)
        except Exception as e:
            return f'Error when parsing argument {line!r}: {e}'
        if not (data_dir / 'entry.json').is_file():
            return f'{data_dir} is not valid data director. Missing entry.json file.'
        # Clear GPU cache before benchmark
        import torch
        torch.cuda.empty_cache()
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

def main():
    args = parse_args()
    set_default_device(args.gpu)
    serve(CommandProcessor(args.module))

if __name__ == '__main__':
    main()
