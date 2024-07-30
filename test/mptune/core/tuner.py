#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..core import MonadAction, MonadMessage, Monad, MonadService
from abc import abstractmethod

class TunerService(MonadService):

    def init(self, init_object):
        gpu, total_shards = init_object
        self._gpu = gpu
        self._gpu_device = f'cuda:{gpu}'
        self._cached_ctx = None
        self._cached_params = None
        self._cached_tup = None

    def process(self, request):
        import torch
        if request.action == MonadAction.Exit:
            yield request
            return
        print(f'Worker receive {request}')
        with torch.cuda.device(self._gpu):
            torch.manual_seed(20)
            item = 0
            for kernel_name, perf_number, kig in self.profile(request):
                yield request.forward(self.monad).update_payload(profiled_kernel_name=kernel_name,
                                                                 perf_number=perf_number,
                                                                 kig_dict=kig)
                # print(f'Worker yield {kig=}')
                print(f'Worker yield {item=}')
                item += 1
        print(f'Worker complete {request}')

    def cleanup(self):
        pass

    def hit_cache(self, tup):
        if self._cached_tup == tup:
            return self._cached_ctx, self._cached_params
        del self._cached_ctx
        del self._cached_params  # Must, dropout_p is there
        self.create_ctx_cache(tup)
        return self._cached_ctx, self._cached_params

    @abstractmethod
    def create_ctx_cache(self, tup):
        pass

    @abstractmethod
    def profile(self, request):
        pass
