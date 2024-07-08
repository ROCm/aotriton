# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# Copyright © 2020-2022 OpenAI
# SPDX-License-Identifier: MIT

from .kernel_argument import TunedArgument

class Config:
    '''
    A compatibile class to store triton.Config
    '''
    def __init__(self, kwargs, num_warps=4, num_stages=2, num_ctas=1, maxnreg=None, pre_hook=None):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_ctas = num_ctas
        self.num_stages = num_stages
        self.maxnreg = maxnreg
        self.pre_hook = pre_hook

    def translate_to_psel_and_co(self, perf_metas : 'list[ArgumentMetadata]'):
        psels = []
        for k, v in self.kwargs.items():
            for meta in perf_metas:
                if meta.has_argument(k):
                    psels.append(TunedArgument(meta, v))
                    break
        if 'waves_per_eu' in self.kwargs:
            co = {'waves_per_eu' : self.kwargs['waves_per_eu'] }
        else:
            co = {}
        co['num_warps'] = self.num_warps
        co['num_stages'] = self.num_stages
        # print(f'translate_to_psel_and_co {psels=} {co=}')
        return psels, co
