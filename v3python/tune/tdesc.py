# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from dacite import from_dict
from dataclasses import asdict

'''
A dual-purpose class

On Celery Controller: generate entries for celery tasks
On Celery Worker: actual perform tasks
'''
class TuningDescription(ABC):
    @property
    @abstractmethod
    def ENTRY_CLASS(self):
        pass

    @property
    @abstractmethod
    def INPUT_METADATA(self):
        pass
    '''
    generate_entries:
        Generate an entry object that can uniquely locate a entry in the tuning
        table (sans Arch/GPU selection, which is handled in upper layer)

    Note:
        An entry will be extended into Input Metadata object, which contains
        additional fields like batch sizes and PRNG seeds.
        This step should be handled inside run_test()
    '''
    @abstractmethod
    def generate_entries(self):
        pass

    @abstractmethod
    def list_kernels(self, entry) -> list[str]:
        pass

    def probe_backends(self, entry, which_kernel: str, root: Path) -> list[dict]:
        with open(root / 'entry.json') as f:
            ej = json.load(f)
        entry = self.ENTRY_CLASS.from_dict(ej['entry'])
        test = ej['tests'][0]
        im = self.INPUT_METADATA.from_dict(test["input_metadata"])
        pt = Path(test["pt_file"])
        return self._do_probe_backends(entry, im, which_kernel, pt)

    @abstractmethod
    def _do_probe_backends(self, entry, im, which_kernel: str, root: Path) -> list[dict]:
        pass

    @abstractmethod
    def _gen_ref(self, entry, root: Path):  # Gen [tname: str, input_metadata, pt: Path]
        pass

    def prepare_data(self, entry, root: Path):
        def iterate_test():
            for tname, im, pt in self._gen_ref(entry, root):
                yield {'test_name': tname, 'input_metadata' : asdict(im), 'pt_file': pt.as_posix()}
        with open(root / 'entry.json', 'w') as f:
            json.dump({'entry' : asdict(entry), 'tests': list(iterate_test()) }, f)

    @abstractmethod
    def run_single_test(self,
                        input_metadata,
                        pt: Path,
                        which_kernel) -> list[float]:  # L1 error
        pass

    @abstractmethod
    def run_single_benchmark(self,
                             input_metadata,
                             pt: Path,
                             which_kernel) -> tuple[dict, list[float]]:
        pass

    def benchmark(self, root: Path, which_kernel: 'KernelSelector'):
        with open(root / 'entry.json') as f:
            ej = json.load(f)
        entry = self.ENTRY_CLASS.from_dict(ej['entry'])
        tests = ej['tests']
        def gen():
            for t in tests:
                im = self.INPUT_METADATA.from_dict(t['input_metadata'])
                pt = t['pt_file']
                yield t['test_name'], im, pt
        adiffs = {tname : self.run_single_test(im, pt, which_kernel) for tname, im, pt in gen()}
        for _, bim, pt in gen():
            impl_desc, times = self.run_single_benchmark(bim, pt, which_kernel)
            break
        return entry, impl_desc, adiffs, times, bim
