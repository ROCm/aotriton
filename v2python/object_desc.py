#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import numpy as np
import json
import hashlib

SOURCE_PATH = Path(__file__).resolve()

class ObjectFileDescription(object):
    SIGNATURE_TO_C = {
        'fp32'    : 'float',
        '*fp32'   : 'const float*',
        '*fp16'   : 'const __fp16*',
        '*bf16'   : 'const __bf16*',
        'i8'      : 'int8_t',
        'i16'     : 'int16_t',
        'i32'     : 'int32_t',
        'i64'     : 'int64_t',
        'u8'      : 'uint8_t',
        'u16'     : 'uint16_t',
        'u32'     : 'uint32_t',
        'u64'     : 'uint64_t',
        np.int8   : 'int8_t',
        np.int16  : 'int16_t',
        np.int32  : 'int32_t',
        np.int64  : 'int64_t',
        np.uint8  : 'uint8_t',
        np.uint16 : 'uint16_t',
        np.uint32 : 'uint32_t',
        np.uint64 : 'uint64_t',
    }
    C_SIZE = {
        'bool'              : 1,
        'const __bf16*'     : 8,
        'const __fp16*'     : 8,
        'const float*'      : 8,
        'float'             : 4,
        'int8_t'            : 1,
        'int16_t'           : 2,
        'int32_t'           : 4,
        'int64_t'           : 8,
        'uint8_t'           : 1,
        'uint16_t'          : 2,
        'uint32_t'          : 4,
        'uint64_t'          : 8,
    }
    def is_tensor_type(t):
        return t.startswith('*')

    DEFAULT_NUM_WARPS = 4
    DEFAULT_NUM_STAGES = 1
    DEFAULT_WAVES_PER_EU = 1

    def build_kernel_filename(self, sig: 'KernelSignature'):
        # print(f"{gpu=} {fsels=} {psels=} {compiler_options=}")
        # sig = KernelSignature(self, fsels, psels, compiler_options, gpu)
        # fn = file_name_prefix + '-Kernel-' if file_name_prefix else ''
        # kernel_name =  if kernel_name is None else kernel_name
        fn = self.SHIM_KERNEL_NAME
        # print(f'{sig.compact_signature=}')
        fn += '-Sig-' + sig.compact_signature
        fn += '-Gpu-' + sig.target_arch
        fn += '.hsaco'
        return fn

    def __init__(self,
                 triton_kernel_desc : 'KernelDescription',
                 signature: 'KernelSignature',
                 hsaco_outpath : Path,
                 sancheck_fileexists = False):
        self._triton_kernel_desc = triton_kernel_desc
        self.KERNEL_FAMILY = self._triton_kernel_desc.KERNEL_FAMILY
        self.SHIM_KERNEL_NAME = self._triton_kernel_desc.SHIM_KERNEL_NAME
        self._signature = signature
        self._hsaco_kernel_path = hsaco_outpath / self.build_kernel_filename(signature)
        # self._hsaco_metatdata_path = Path() if triton_metadata_path is None else self._triton_file_path.with_suffix('.json')
        self._hsaco_metatdata_path = self._hsaco_kernel_path.with_suffix('.json')
        if self.compiled_files_exist:
            with self._hsaco_metatdata_path.open('r') as f:
                self._metadata = json.load(f)
        else:
            if sancheck_fileexists and not self.is_functional_disabled():
                assert False, f'GPU Kernel {self._hsaco_kernel_path} failed to compile. This is a bug when -DAOTRITON_BUILD_FOR_TUNING=OFF'
            self._metadata = {}

    @property
    def compiled_files_exist(self):
        return self._hsaco_kernel_path.exists() and self._hsaco_metatdata_path.exists()

    @property
    def compile_successful(self):
        if self.compiled_files_exist:
            with open(self._hsaco_metatdata_path) as f:
                j = json.load(f)
            return j['compile_status'] == 'Complete'
        return False

    @property
    def godel_number(self):
        return self._signature.godel_number

    @property
    def compact_signature(self):
        return self._signature.compact_signature

    @property
    def compact_signature_components(self):
        return self._signature.get_compact_signature_components()

    @property
    def human_readable_signature(self):
        return self._signature.human_readable_signature

    def blake2b_hash(self, package_path):
        raw = package_path.encode('utf-8')
        _, psel, copts = self.compact_signature_components
        s = '__P__' + psel + '__CO__' + copts + '-Gpu-' + self._signature.target_arch
        raw += s.encode('utf-8')
        h = hashlib.blake2b(raw, digest_size=8)
        return h.hexdigest(), raw

    @property
    def c_identifier_signature(self):
        return self._signature.compact_signature.replace('^', 'Ptr').replace('@', 'Align')

    @property
    def functional_signature(self):
        return self._signature.functional_signature

    @property
    def designated_perf_initializer_list(self):
        lets = []
        for sel in self._signature._perf_selections:
            v = sel.argument_value
            for a in sel.argument_names:
                lets.append(f'.{a} = {v}')
        return lets

    @property
    def src(self):
        return self._triton_kernel_desc._triton_file_path

    @property
    def entrance(self):
        return self._triton_kernel_desc._triton_kernel_name

    @property
    def binary_entrance(self):
        if self.compile_successful:
            assert self._metadata, f'Did not load the metadata from {self._hsaco_metatdata_path}'
            return self._metadata['name']
        return ''

    @property
    def obj(self):
        return self._hsaco_kernel_path

    @property
    def signature(self):
        # print(f'{self._signature.triton_api_signature_list=}')
        return ', '.join(self._signature.triton_api_signature_list)

    @property
    def shared_memory_size(self):
        if self.compile_successful:
            return self._metadata['shared']
        return -1

    @property
    def num_warps(self):
        num_warps = self._metadata.get('num_warps', self.DEFAULT_NUM_WARPS)
        return self._signature._compiler_options.get('num_warps', num_warps)

    @property
    def num_stages(self):
        num_stages = self._metadata.get('num_stages', self.DEFAULT_NUM_STAGES)
        return self._signature._compiler_options.get('num_stages', num_stages)

    @property
    def waves_per_eu(self):
        return self._signature._compiler_options.get('waves_per_eu', self.DEFAULT_WAVES_PER_EU)

    @property
    def warp_size(self):
        if self.compile_successful:
            return self._metadata['warp_size']
        return 0

    @property
    def target_arch(self):
        return self._signature.target_arch

    def generate_shim_header_member_function(self) -> str:
        TEMPLATE = ' hipError_t operator()(dim3 grid, {shim_arguments}, hipStream_t stream);\n'
        shim_arguments, _ = self.compute_c_argument()
        fmt = {
                'shim_arguments': shim_arguments,
        }
        return TEMPLATE.format_map(fmt)

    def generate_shim_header_closing_struct_define(self) -> str:
        return '};\n\n'

    def generate_shim_header_extern_template(self) -> str:
        TEMPLATE = 'template struct {shim_kernel_name}<{shim_kernel_specialization}>;'
        template_specialization = self.compute_struct_template_specialization(align1=len(self.SHIM_KERNEL_NAME)+17)
        fmt = {
            'shim_kernel_name': self.SHIM_KERNEL_NAME,
            'shim_kernel_specialization': template_specialization,
        }
        return TEMPLATE.format_map(fmt)

    def compute_c_argument(self, align1=23, align2=30):
        arguments = self.get_c_arguments()
        typed_arguments = [f'{self.get_ctype(a)[1]} {a}' for a in arguments]
        casted_arguments = [f'static_cast<void*>(&{a})' for a in arguments]
        ALIGN1 = ',\n' + ' ' * align1
        ALIGN2 = ',\n' + ' ' * align2
        return ALIGN1.join(typed_arguments), ALIGN2.join(casted_arguments)

    def sanitize_type(self, t):
        colon = t.rfind(':')
        colon = None if colon < 0 else colon
        return t[:colon]

    def get_ctype(self, arg_name):
        for k, v in self._argument_choices.items():
            if arg_name in k:
                stemv = self.sanitize_type(str(v))
                not_constant = stemv in self.SIGNATURE_TO_C
                if not_constant:
                    ctype = self.SIGNATURE_TO_C[stemv]
                elif isinstance(v, bool):
                    ctype = 'bool'
                elif isinstance(v, int):
                    ctype = 'int32_t'
                elif isinstance(v, float):
                    ctype = 'float'
                return not_constant, ctype
        assert False, f"cannot find {arg_name}"

    def get_cvalue(self, arg_name):
        for k, v in self._argument_choices.items():
            if arg_name in k:
                if isinstance(v, bool):
                    return 'true' if v else 'false'
                return v
        assert False, f"cannot find {arg_name}"

    def get_c_arguments(self):
        return self._filter_arguments(request_constants=False)

    def get_template_arguments(self):
        return self._filter_arguments(request_constants=True)

    def _filter_arguments(self, request_constants):
        ret = []
        for a in self._triton_kernel_desc.ARGUMENTS:
            not_constant, _ = self.get_ctype(a)
            if request_constants:
                if not not_constant:
                    ret.append(a)
            else:
                if not_constant:
                    ret.append(a)
        return ret

    # Returns things for
    # extern template<int STAGE,
    #                 int BLOCK_M,
    #                 ...
    #                >
    def compute_struct_template_typenames(self, align1=9):
        constants = self._filter_arguments(request_constants=True)
        typed_constants = []
        for a in constants:
            typed_constants.append(f'{self.get_ctype(a)[1]} {a}')
        ALIGN1 = ',\n' + ' ' * align1
        return ALIGN1.join(typed_constants)

    def compute_struct_template_specialization(self, align1=13):
        constants = self._filter_arguments(request_constants=True)
        constant_values_with_hint = []
        for a in constants:
            v = str(self.get_cvalue(a))
            v += f' /* {a} */'
            constant_values_with_hint.append(v)
        ALIGN1 = ',\n' + ' ' * align1
        return ALIGN1.join(constant_values_with_hint)

    def is_functional_disabled(self):
        return self._signature.is_functional_disabled()

    """
    def compute_template_arguments(self, align1=1, align2=10):
        arguments = self.get_template_arguments()
        type_choices = self._argument_choices
        typed_arguments = []
        template_constants = []
        typename_allocation = {}
        # do NOT loop over type_choices directly.
        # We want to maintain the order of typename Type#
        for a in self._triton_kernel_desc.ARGUMENTS:
            for k, v in type_choices.items():
                if a not in k:
                    continue
                if isinstance(v, str):
                    allocated = False
                    for k in typename_allocation.keys():
                        if a in k:
                            allocated = True
                            break
                    if not allocated:
                        index = len(typed_arguments)
                        tname = f'typename Type{index}'
                        typed_arguments.append(tname)
                        typename_allocation[k] = tname
                elif isinstance(v, bool):
                    cv = 'true' if v else 'false'
                    template_constants.append(f'bool {a}  = {cv}')
                else:
                    template_constants.append(f'{self.get_ctype(v)} {a}  = {v}')

        ALIGN1 = ',\n' + ' ' * align1
        ALIGN2 = ';\n' + ' ' * align2
        return ALIGN1.join(typed_arguments), ALIGN2.join(template_constants)
    """

