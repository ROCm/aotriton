# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .gpu_targets import AOTRITON_GPU_ARCH_TUNING_STRING
import json

class KernelSignature(object):
    def __init__(self,
                 kdesc : 'KernelDescription',
                 func_selections : 'tuple[ArgumentSelection]',
                 perf_selections : 'tuple[ArgumentSelection]',
                 compiler_options: dict,
                 gpu : str):
        self._kdesc = kdesc
        self._func_selections = func_selections
        self._perf_selections = perf_selections
        self._selections = list(func_selections) + list(perf_selections)
        self._compiler_options = {} if compiler_options is None else compiler_options
        self._gpu = gpu
        self._arch = AOTRITON_GPU_ARCH_TUNING_STRING[gpu]

    COMPACT_COMPILER_OPTION = {
        'waves_per_eu' : 'wave',
        'num_warps': 'warp',
        'num_stages': 'stg',
    }

    def get_compact_compiler_option_name(self, co):
        return self.COMPACT_COMPILER_OPTION.get(co, co)

    @property
    def target_gpu(self):
        return self._gpu

    @property
    def godel_number(self):
        return sum([s.godel_number for s in self._func_selections])

    @property
    def compact_signature(self):
        lf = [s.compact_signature for s in self._func_selections]
        lp = [s.compact_signature for s in self._perf_selections]
        lc = [f'{self.get_compact_compiler_option_name(k)}{v}' for k, v in self._compiler_options.items() if k != '_debug']
        sf = '_'.join([x for x in lf if x is not None])
        sp = '_'.join([x for x in lp if x is not None])
        co = '_'.join([x for x in lc if x is not None])
        return 'F__' + sf + '__P__' + sp + '__CO__' + co

    @property
    def human_readable_signature(self):
        lf = [s.human_readable_signature for s in self._func_selections]
        lp = [s.human_readable_signature for s in self._perf_selections]
        lc = [f'{k}={v}' for k, v in self._compiler_options.items() if k != '_debug']
        sf = ' '.join([x for x in lf if x is not None])
        sp = ' '.join([x for x in lp if x is not None])
        co = ' '.join([x for x in lc])
        return sf + ' ; ' + sp + ' ; ' + co

    @property
    def functional_signature(self):
        lf = [s.compact_signature for s in self._func_selections]
        sf = ','.join([x for x in lf if x is not None])
        return 'FONLY__' + sf + '__'

    @property
    def arguments(self):
        return self._kdesc.ARGUMENTS

    @property
    def triton_api_signature_list(self) -> 'list[str]':
        sig = {}
        [s.update_triton_api_signature(sig) for s in self._selections]
        l = [None] * len(self.arguments)
        for k, v in sig.items():
            l[k] = v
        return l

    def codegen_perf_object(self) -> str:
        perf_key_value = []
        for ps in self._perf_selections:
            value = ps.argument_value
            if isinstance(value, bool):
                value = 'true' if value else 'false'
            for aname in ps.argument_names:
                perf_key_value.append(f'.{aname} = {value}')
        return ', '.join(perf_key_value)

    def jsongen_psels(self) -> str:
        d = {}
        for ps in self._perf_selections:
            value = ps.argument_value
            for aname in ps.argument_names:
                d[aname] = value
        return json.dumps(d)

    def jsongen_copts(self) -> str:
        return json.dumps(self._compiler_options)
