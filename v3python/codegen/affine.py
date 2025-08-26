# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/affine.<kernel_name>.{h,cc}

import io
from ..base import (
    typed_choice as TC,
    Functional,
    Interface,
)
from .interface import InterfaceGenerator
from ..affine import AffineKernelDescription
from .affine_cap import AffineCapabilityGenerator
from ..utils import (
    LazyFile,
    RegistryRepository,
    log,
)
from .template import get_template
from .common import codegen_struct_cfields, codegen_includes
import hashlib

class AffineGenerator(InterfaceGenerator):
    HEADER_TEMPLATE = get_template('affine.h')
    SOURCE_TEMPLATE = get_template('affine.cc')
    PFX = 'affine'

    def __init__(self, args, iface : Interface, parent_repo : RegistryRepository):
        super().__init__(args, iface, parent_repo)
        akdesc = iface
        # Patch _target_arch since affine kernel may not support all arches.
        self._target_arch = { arch: gpus for arch, gpus in self._target_arch.items() if arch in akdesc.SUPPORTED_ARCH }
        del self._target_gpus  # For safety
        self._target_arch_keys = list(self._target_arch.keys())

    '''
    Unlike Triton kernel. Affine kernel does not need an autotune table.
    Hence even if a sub-generator is returned, this sub-generator will not generate dedicated files.
    All code will be consolidated into the affine.<kernel_name>.cc file
    '''
    def create_sub_generator(self, functional : Functional, df : 'pandas.DataFrame', sql : str):
        akdesc = functional.meta_object
        if akdesc.is_functional_disabled(functional):
            log(lambda : f'Functional {functional.godel_number=} disabled in affine kernel {akdesc.NAME}')
            use_this_functional = False
            return None, use_this_functional
        use_this_functional = True
        log(lambda : f'Translating Functional with godel number {functional.godel_number}')
        df, dkarg = akdesc.translate_empty_dataframe(functional)
        if df.empty:
            use_this_functional = False
            return None, use_this_functional
        capgen = AffineCapabilityGenerator(self._args, akdesc, functional, df, dkarg, self._this_repo)
        return capgen, use_this_functional

    def write_shim_header(self, functionals, fout):
        akdesc = self._iface
        shared_iface = akdesc.SHARED_IFACE is not None
        shared_iface_family = akdesc.SHARED_IFACE.FAMILY if shared_iface else akdesc.FAMILY
        if shared_iface:
            self._add_iface_for_source(akdesc.SHARED_IFACE)
        d = {
            'shared_iface_family'   : shared_iface_family,
            'shared_iface'          : 1 if shared_iface else 0,
            'kernel_family_name'    : akdesc.FAMILY,
            'affine_kernel_name'      : akdesc.NAME,
            'param_class_name'      : akdesc.param_class_name,
            'context_class_name'    : akdesc.context_class_name,
            'func_fields'           : codegen_struct_cfields(akdesc.func_cfields, nalign=4),
            'residual_func_fields'  : codegen_struct_cfields(akdesc.residual_func_cfields, nalign=8),
            'csv_perf_fields'       : self.codegen_csv_perf_fields(),
            'union_of_possible_structs'     : self.codegen_union_of_possible_structs(),
            'pp_func_decls'         : self.codegen_pp_func_decls(),
            'number_of_functionals_with_residuals' : akdesc.godel_number,
        }
        d['includes'] = codegen_includes(self._hdr_include_repo.get_data())
        print(self.HEADER_TEMPLATE.format_map(d), file=fout)

    def write_shim_source(self, functionals, fout):
        akdesc = self._iface
        shared_iface = akdesc.SHARED_IFACE is not None
        shared_iface_family = akdesc.SHARED_IFACE.FAMILY if shared_iface else akdesc.FAMILY
        validator_defs, capability_table_entries = self.codegen_capability_table(functionals)
        meta_cos = self.codegen_compact_kernels()
        packed_string = self._this_repo.get_data('affine_kernel_packed_string', return_none=True)
        packed_string = '""' if packed_string is None else packed_string
        d = {
            'shared_iface'        : 1 if shared_iface else 0,
            'shared_iface_family' : shared_iface_family,
            'kernel_family_name'  : akdesc.FAMILY,
            'affine_kernel_name'  : akdesc.NAME,  # TODO: use signature so AMD_LOG_LEVEL=3 is more meaningful
            'param_class_name'    : akdesc.param_class_name,
            'context_class_name'  : akdesc.context_class_name,
            'godel_number_body'   : self.codegen_godel_number_body(),
            'get_archmod_number_body'               : self.codegen_archmod_number_body(),
            'meta_cos'                              : meta_cos,
            'kernel_co_name_packed_string'          : packed_string,
            'number_of_functionals_with_residuals'  : akdesc.godel_number,
            'validator_defs'                        : validator_defs,
            'capability_table_entries'              : capability_table_entries,
        }
        d['includes'] = codegen_includes(self._src_include_repo.get_data())
        print(self.SOURCE_TEMPLATE.format_map(d), file=fout)

    def codegen_godel_number_body(self):
        body = io.StringIO()
        iface = self._iface
        for tp in iface.list_functional_params():
            self.codegen_godel_number_calculation(tp, body)
        print(' ' * 4, '// Residual Choices start here', file=body)
        for tp in iface.list_residual_functional_params():
            self.codegen_godel_number_calculation(tp, body, anamespace='residual_args.')  # Accessible directly in context object
        return body.getvalue()

    def codegen_csv_perf_fields(self, nalign=8):
        akdesc = self._iface
        if akdesc.CSV_PROPERTIES is None:
            return ''
        stmt = []
        for csvp in akdesc.CSV_PROPERTIES:
            stmt.append(f'{csvp.iface_param} {csvp.column}')
        ALIGN = ';\n' + ' ' * nalign
        return ALIGN.join(stmt) + ';'  # The template doesn't contain ';' because this struct can be empty.

    def codegen_union_of_possible_structs(self):
        akdesc = self._iface
        if akdesc.DIRECT_KERNEL_ARGS is None:
            return ''
        stmt = []
        for dkargs in akdesc.DIRECT_KERNEL_ARGS:
            stmt.append(f'{dkargs.full_name} {dkargs.NAME}')
            self._add_include_to_header(dkargs.INCLUDE)
        ALIGN = ';\n' + ' ' * 8
        return ALIGN.join(stmt) + ';\n'

    def codegen_pp_func_decls(self):
        akdesc = self._iface
        if akdesc.DIRECT_KERNEL_ARGS is None:
            return ''
        stmt = []
        for dkargs in akdesc.DIRECT_KERNEL_ARGS:
            stmt.append(f'std::tuple<dim3, dim3> pp_direct_kernel_args_for_{dkargs.NAME}(DirectKernelArguments&) const')
        ALIGN = ';\n' + ' ' * 4
        return ALIGN.join(stmt) + ';\n'

    def codegen_compact_kernels(self):
        dic = self._this_repo.get_data('affine_kernel_as_triton_kernel', return_none=True)
        if dic is None:
            return ''
        meta_cos = []
        # TODO, Need something like ksignature for affine kernel
        for (arch, stem_co_name), (co_index, (offset_arch, offset_co)) in dic.items():
            raw = arch.encode('utf-8')
            raw += b'/'
            raw += stem_co_name.encode('utf-8')
            b2sum_u64 = hashlib.blake2b(raw, digest_size=8).hexdigest()
            u8raw = raw.decode('utf-8')
            assert len(b2sum_u64) == 16
            b2sum_u64_hi = b2sum_u64[:8]
            b2sum_u64_lo = b2sum_u64[8:]
            meta_cos.append(f'{{ 0x{b2sum_u64_hi}u, 0x{b2sum_u64_lo}u, {offset_arch}, {offset_co} }}, // {b2sum_u64} = b2sum -l 64 <<< {u8raw}')
        ALIGN = '\n' + 4 * ' '
        return ALIGN.join(meta_cos)

    def codegen_capability_table(self, functionals):
        validator_registry = self._this_repo.get_data('validator_function', return_none=True)
        if validator_registry is None:
            return '', ''
        table_entries = {}
        validator_defs = []
        for fsrc, item in validator_registry.items():
            validator_defs.append(f'{item.ret} {item.name}{item.params}\n' + fsrc)
        capability_table_entries = self.codegen_tune_table_entries(functionals)
        ALIGN_V = '\n\n'
        return ALIGN_V.join(validator_defs), capability_table_entries

    def codegen_tune_struct_name(self, arch_number, godel_number):
        validator_registry = self._this_repo.get_data('validator_assignment')
        validator_name = validator_registry.get((arch_number, godel_number), 'nullptr')
        return validator_name, False
