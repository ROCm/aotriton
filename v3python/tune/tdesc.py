# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Generator

'''
A dual-purpose class

On Celery Controller: generate configs for celery tasks
On Celery Worker: actual perform tasks
'''
class TestingDescription(ABC):
    '''
    gen_entry_config:
        Generate A config object that can uniquely locate a entry in the tuning
        table (sans Arch/GPU selection, which is handled in upper layer)

    Note:
        A Config may contain additional fields like unittest settings, which
        should be handled by the tester and gen_entry_config should only
        generate one option.
    '''
    @abstractmethod
    def gen_entry_config(self) -> Generator['Config']:
        pass

    @abstractmethod
    def list_kernels(self, entry_config: 'Config') -> list[str]:
        pass

    @abstractmethod
    def probe_backends(self,
                       entry_config: 'Config',
                       kernel_name: str) -> list[dict]:
        pass

    @abstractmethod
    def gen_ref(self, entry_config: 'Config', root: Path, device: str = None) -> Generator[tuple['Config', Path]]:
        pass

    @abstractmethod
    def write_ref(self, config: 'Config', pt: Path) -> tuple['Config', Path]:
        pass

    @abstractmethod
    def run_test(self,
                 confg: 'Config',
                 pt: Path,
                 kernel_name: str,
                 backend_index: int,
                 device: str = None):
        pass
