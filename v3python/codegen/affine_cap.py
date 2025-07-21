# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..base import (
    typed_choice as TC,
    Functional,
)
from .template import get_template
from .basetune import BaseTuneCodeGenerator

class AffineCapabilityGenerator(BaseTuneCodeGenerator):
    VALIDATOR_TEMPLATE = get_template('snippet/validator.cc')

    def __init__(self,
                 args,
                 akdesc : 'AffineKernelDescription',
                 f : Functional,
                 df : 'pd.DataFrame',
                 dkarg : 'v3python.affine.DirectKernelArguments',
                 parent_repo):
        super().__init__(args, f, None, parent_repo)
        self._akdesc = akdesc
        self._df = df
        # print(f'{df}')
        # print(df['CoName'])
        # print(df['CoName'].__class__)
        # # import ipdb; ipdb.set_trace();
        # breakpoint()
        # # print(df['CoName'].iat(0))
        self._dkarg = dkarg

    def get_cc_file(self, f):
        return None

    def generate(self):
        akdesc = self._akdesc
        f = self._f
        godel_number, fsrc = self.codegen_validator()
        validator_registry = self._parent_repo.get_function_registry('validator_function')
        val_function_pfx = f'validator_Arch_{f.arch}_lambda'
        val_params = f'({akdesc.context_class_name}& context, int mod_number)'
        valf_name = validator_registry.register(fsrc, 'hipError_t', val_function_pfx, val_params)
        validator_assignment = self._parent_repo.get_dict_registry('validator_assignment')
        validator_assignment.register((f.arch_number, f.godel_number), valf_name)

    def codegen_validator(self):
        akdesc = self._akdesc
        f = self._f
        def remove_dot_co(s):
            if s.endswith('.co'):
                return s[:-len('.co')]
            return s
        package_path = str(f.full_filepack_dir / 'affine_kernels')
        base_kernel_co = self._df['CoName'].iat[0]
        # put co filename to asms, so it can be propogated to Bare.cluster
        kernel_co = akdesc.co_dir(f) / base_kernel_co
        asm_registry = self._parent_repo.get_hsaco_registry('asms')
        asm_registry.register(package_path, str(kernel_co), append=True)
        stem_co_name = remove_dot_co(base_kernel_co)
        # put co filename string to affine_kernel_packed_string
        affine_kernel_packed_string = self._parent_repo.get_string_registry('affine_kernel_packed_string')
        offset_arch = affine_kernel_packed_string.register(f.arch)
        offset_co = affine_kernel_packed_string.register(stem_co_name)
        # register affine_kernel
        kernel_cluster = self._parent_repo.get_signatured_function_registry('affine_kernel_as_triton_kernel')
        kernel_obj_index = kernel_cluster.register((f.arch, stem_co_name), (offset_arch, offset_co))
        entrance_name = f'fmha_{stem_co_name}'
        mangled_name = f"_ZN5aiter{len(entrance_name)}{entrance_name}E"
        d = {
            'arch'                  : f.arch,
            'godel_number'          : f.godel_number,
            'arch_number'           : f.arch_number,
            'context_class_name'    : akdesc.context_class_name,
            'kernel_obj_index'      : kernel_obj_index,
            'direct_kernel_args'    : self._dkarg.NAME,
            'full_name_kernel_args' : self._dkarg.full_name,
            'package_path'          : package_path,
            'mangled_name'          : mangled_name,
            'perf_args_assignment'  : self.codegen_perf_args_assignment(),
        }
        return f.godel_number, self.VALIDATOR_TEMPLATE.format_map(d)

    def codegen_perf_args_assignment(self, nalign=4):
        akdesc = self._akdesc
        if akdesc.CSV_PROPERTIES is None:
            return ''
        f = self._f
        df = self._df
        stmt = []
        for csvp in akdesc.CSV_PROPERTIES:
            value = csvp.translate_csv_property(df, functional=f)
            stmt.append(f'context.perf_args.{csvp.column} = {value}')
        ALIGN = ';\n' + ' ' * nalign
        return ALIGN.join(stmt)
