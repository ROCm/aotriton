# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/<prefix>.<kernel_name>.{h,cc}
# For operator and kernel

from abc import ABC, abstractmethod
import io
from ..base import (
    typed_choice as TC,
    Interface,
    Functional,
)
from ..utils import (
    LazyFile,
    get_template,
    RegistryRepository,
)
# from ..utils.is_tuning_enabled import is_tuning_on_for_kernel
from ..database import Factories as DatabaseFactories
from .autotune import AutotuneCodeGenerator
from ..gpu_targets import cluster_gpus

class InterfaceGenerator(ABC):
    HEADER_TEMPLATE = None  # get_template('shim.h')
    SOURCE_TEMPLATE = None  # get_template('shim.cc')
    PFX = None              # 'shim'/'op'

    def __init__(self, args, iface : Interface, parent_repo : RegistryRepository):
        self._args = args
        self._iface = iface
        # self._tuning = is_tuning_on_for_kernel(self._args, self._iface)
        self._target_gpus = args.target_gpus
        self._target_arch = cluster_gpus(self._target_gpus)
        self._target_arch_keys = list(self._target_arch.keys())
        # print(f'{self._target_arch=}')
        self._parent_repo = parent_repo
        self._this_repo = RegistryRepository()
        self._shim_files = []

    @property
    def this_repo(self):
        return self._this_repo

    @property
    def shim_files(self):
        return self._shim_files

    def generate(self):
        # Un "self._" section
        args = self._args
        iface = self._iface

        # registry
        all_functionals = []

        # autotune phase
        fac = DatabaseFactories.create_factory(args.build_dir)
        for functional in iface.gen_functionals(self._target_arch):
            # print(f'{functional=}')
            df = fac.create_view(functional)
            # print(f'KernelShimGenerator.generate {df=}')
            subg = self.create_sub_generator(functional, df)
            subg.generate()
            self._shim_files.append(subg.cc_file)
            all_functionals.append(functional)

        # shim code phase
        # Must be after autotune due to common functions needed by autotune is
        # generated in autotune module
        shim_path = args.build_dir / iface.FAMILY
        shim_path.mkdir(parents=True, exist_ok=True)
        shim_fn = self.PFX + '.' + iface.NAME + '.h'
        fullfn = shim_path / shim_fn
        with LazyFile(fullfn) as fout:
            self.write_shim_header(all_functionals, fout)
        with LazyFile(fullfn.with_suffix('.cc')) as fout:
            self.write_shim_source(all_functionals, fout)
        self._shim_files.append(fullfn)
        self._shim_files.append(fullfn.with_suffix('.cc'))

    @abstractmethod
    def create_sub_generator(self, functional : Functional):
        pass

    @abstractmethod
    def write_shim_header(self, functionals, fout):
        pass

    @abstractmethod
    def write_shim_source(self, functionals, fout):
        pass
