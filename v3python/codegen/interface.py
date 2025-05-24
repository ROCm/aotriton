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
from .template import get_template
from ..utils import (
    LazyFile,
    RegistryRepository,
)
# from ..utils.is_tuning_enabled import is_tuning_on_for_kernel
from ..database import Factories as DatabaseFactories
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
        self._hdr_include_repo = self._this_repo.get_list_registry('headers_needed_in_header_file')
        self._src_include_repo = self._this_repo.get_list_registry('headers_needed_in_source_file')
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
            subg, use_this_functional = self.create_sub_generator(functional, df)
            if subg is not None:
                subg.generate()
                self._shim_files.append(subg.cc_file)
            if use_this_functional:
                all_functionals.append(functional)

        # Skip re-generation of shim files
        if args.build_for_tuning_second_pass:
            return

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

    def _add_header_for_header(self, iface):
        fn = self._translate_iface_to_header(iface)
        self._hdr_include_repo.register(fn)

    def _add_header_for_source(self, iface):
        fn = self._translate_iface_to_header(iface)
        self._src_include_repo.register(fn)

    def _translate_iface_to_header(self, iface):
        return f'{iface.FILE_PFX}.{iface.NAME}.h'

    '''
    Tuning Related functions
    '''
    def codegen_tune_struct_name(self, arch_number, godel_number):
        tune_name = self._iface.TUNE_NAME.capitalize()
        return f'{tune_name}_{self._iface.NAME}__A{arch_number}__F{godel_number}', True

    def codegen_tune_table_entry_declares(self, functionals):
        decls = []
        for arch_number, target_arch in enumerate(self._target_arch_keys):
            godel_numbers = sorted(list(set([f.godel_number for f in functionals if f.arch == target_arch])))
            for godel_number in godel_numbers:
                struct_name, is_extern = self.codegen_tune_struct_name(arch_number, godel_number)
                if is_extern:
                    decls.append(f'void {struct_name}({self._iface.context_class_name}& params, int mod_number);')
        return '\n'.join(decls)

    def codegen_tune_table_entries(self, functionals):
        lets = []
        for arch_number, target_arch in enumerate(self._target_arch_keys):
            lets.append(4 * ' ' + '{')
            godel_numbers = sorted(list(set([f.godel_number for f in functionals if f.arch == target_arch])))
            for godel_number in range(self._iface.godel_number):
                struct_name, is_extern = self.codegen_tune_struct_name(arch_number, godel_number)
                if godel_number in godel_numbers:
                    if is_extern:
                        lets.append(8 * ' ' + f'&{self._iface.TUNE_NAME}::{struct_name},')
                    else:
                        lets.append(8 * ' ' + f'&{struct_name},')
                else:
                    lets.append(8 * ' ' + f'nullptr,')
            lets.append(4 * ' ' + '},')
        return '\n'.join(lets)

    def codegen_list_of_deduplicated_lut_functions(self):
        registry = self._this_repo.get_data('lut_function', return_none=True)
        if registry is None:
            return '// This Interface does not have tuning LUT function'
        stmt = [f'{item.ret} {item.name} {fsrc};' for fsrc, item in registry.items()]
        return '\n'.join(stmt)

    def codegen_declare_list_of_deduplicated_lut_functions(self):
        registry = self._this_repo.get_data('lut_function', return_none=True)
        if registry is None:
            return '// This Interface does not have tuning LUT function'
        stmt = [f'extern {item.ret} {item.name}{item.params};' for fsrc, item in registry.items()]
        return '\n'.join(stmt)

    '''
    Godel Numbers
    '''
    def codegen_godel_number_body(self):
        body = io.StringIO()
        iface = self._iface
        for tp in iface.list_functional_params():
            self.codegen_godel_number_calculation(tp, body)
        return body.getvalue()

    def codegen_godel_number_calculation(self, tp, fout):
        if tp.nchoices <= 1:
            return
        aname = tp.repr_name # meta._ordered_arguments[0][0]
        INDENT = 4 * ' '
        print(INDENT + '{', file=fout)
        print(2 * INDENT + 'int64_t number = 0;', file=fout)
        for number, tc in enumerate(tp.choices):
            assert not isinstance(tc, TC.ConditionalChoice)
            if isinstance(tc, TC.tensor):
                type_enum = tc.type_enum
                print(2 * INDENT + f'if (args.{aname}->dtype() == {type_enum}) number = {number} ;', file=fout)
            else:
                value = str(tc).lower()
                print(2 * INDENT + f'if (args.{aname} == {value}) number = {number} ;', file=fout)
        print(2 * INDENT + f'sum += number * {tp.godel_number};', file=fout)
        print(1 * INDENT + '}', file=fout)

    def codegen_archmod_number_body(self):
        lets = []
        for i, arch in enumerate(self._target_arch_keys):
            for j, gpu in enumerate(self._target_arch[arch]):
                gpu_enum = f'GPU_AMD_ARCH_{gpu}'.upper()
                # CAVEAT: must return j because some GPU mod may not be selected.
                lets.append(f'if (gpu == {gpu_enum}) return {{ {i}, {j} }}')
        ALIGN = ';\n' + ' ' * 4
        return ALIGN.join(lets)
