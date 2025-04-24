# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .kernel_desc import (
    get_template,
)

class MetroKernel(object):
    # KERNEL_FAMILY  # Must define it
    # DO NOT ADD "Metro" prefix
    METRO_KERNEL_NAME = None

    # Union of all kernels' arguments (rename to the same if named differently since ABI do not care the naming)
    METRO_ARGUMENTS = []
    TYPE_CHOICES = None
    FEAT_CHOICES = None
    TENSOR_RANKS = None
    TENSOR_STRIDE_INPUTS = None

    def __init__(self, route_name, list_of_kernels, is_fallback=False):
        self._route_name = route_name
        self._individual_kernels = list_of_kernels
        self._is_fallback = is_fallback

    @property
    def is_fallback(self):
        return self._is_fallback

    @property
    def enum_name(self):
        CamelRouteName = self._route_name.replace('_', ' ').title().replace(' ', '')
        return f'kMetro_{CamelRouteName}'

    @property
    def individual_kernels(self):
        return self._individual_kernels

    @property
    def target_gpus(self):
        return set.intersection(*[set(k.target_gpus) for k in self._individual_kernels])

    def codegen_dep_header_files(self):
        return [ f'shim.{k.SHIM_KERNEL_NAME}.h' for k in self._individual_kernels ]

    def codegen_godel_number_body(self):
        pass

    def codegen_metro_launch_function_name(self):
        return f'launch_metro_{self._route_name}'

    LAUNCH_TEMPLATE = get_template('snippet/metro_launch.cc')

    def codegen_metro_launch_function_def(self):
        fname = self.codegen_metro_launch_function_name()
        stmt = ['hipError_t {fname}(const {self.metro_param_class_name}& params, const {self.metro_context_class_name}& context, hipStream_t stream) {']
        stmt.append('hipError_t err;')
        ALIGN = '\n' + ' ' * 4
        for i, k in enumerate(self._individual_kernels):
            d = {
              'shim_kernel_name'    : k.SHIM_KERNEL_NAME,
              'shim_ns'             : f'AOTRITON_NS::v2::{k.KERNEL_FAMILY}',
              'param_class_name'    : k.param_class_name,
              'context_class_name'  : k.context_class_name,
              'call_index'          : i,  # It is possible to call a kernel multiple times...
              'shim_kernel_enum'    : k.enum_name,
            }
            stmt.append(self.LAUNCH_TEMPLATE.format_map(d).replace('\n', ALIGN));
        stmt.append('return err;')
        return ALIGN.join(stmt) + '\n}'

    def codegen_requirement_function(self, param_name):
        body = []
        for k in self._individual_kernels:
            # affine kernel
            if hasattr(k, 'codegen_requirement_function_body'):
                body.append(k.codegen_requirement_function_body())
        if body:
            fname = f'metro_check_requirement_{self._route_name}'
            stmt = [f'bool {fname}({param_name}& params, Gpu gpu) {{'] + body + ['}']
            return '\n'.join(stmt), fname
