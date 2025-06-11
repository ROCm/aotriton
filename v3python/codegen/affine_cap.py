# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .template import get_template

class AffineCapabilityGenerator(BaseTuneCodeGenerator):
    VALIDATOR_TEMPLATE = get_template('snippet/cap.cc')

    def __init__(self,
                 args,
                 akdesc : 'AffineKernelDescription',
                 f : Functional,
                 parent_repo):
        super().__init__(args, f, None, parent_repo)
        self._akdesc = akdesc
        self._df, self._dkarg = akdesc.translate_empty_dataframe(f)
        assert not self._df.empty

    def get_cc_file(self, f):
        return None

    def generate(self):
        godel_number, fsrc = self.codegen_validator()
        validator_registry = self._parent_repo.get_dict_registry('validator_function')
        validator_registry.register((self._f.arch, godel_number), fsrc)

    def codegen_validator(self):
        akdesc = self._akdesc
        f = self._f
        akdesc.translate_functional(f)
        def remove_dot_co(s):
            if s.endswith('.co'):
                return s[:-len('.co')]
            return s
        kernel_co = self._df['CoName'].iat(0)
        stem_co_name = remove_dot_co(kernel_co)
        # put co filename to affine_kernel_packed_string
        affine_kernel_packed_string = self._parent_repo.get_string_registry('affine_kernel_packed_string')
        offset_arch = affine_kernel_packed_string.register(f.arch)
        offset_co = affine_kernel_packed_string.register(stem_co_name)
        # register affine_kernel
        kernel_cluster = self._parent_repo.get_signatured_function_registry('affine_kernel_as_triton_kernel')
        kernel_obj_index = kernel_cluster.register((f.arch, stem_co_name), (offset_arch, offset_co))
        entrance_name = 'fmha_{stem_co_name}'
        mangled_name = "_ZN5aiter{len(entrance_name)}{entrance_name}E"
        d = {
            'arch'                  : f.arch,
            'godel_number'          : f.godel_number,
            'arch_number'           : f.arch_number,
            'kernel_obj_index'      : kernel_obj_index,
            'direct_kernel_args'    : self._dkarg.NAME,
            'package_path'          : str(f.full_filepack_path),
            'mangled_name'          : mangled_name,
        }
        return f.godel_number, self.AUTOTUNE_TEMPLATE.format_map(d)
