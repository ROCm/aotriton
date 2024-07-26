#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from tuner_common import MonadAction, MonadMessage, Monad, MonadService
import torch

class TunerWorker(Monad):
    def service_factory():
        return TunerService(self._args, self)

class TunerService(MonadService):

    def init(self, init_object):
        gpu, total_shards = init_object
        self._gpu = gpu
        self._gpu_device = f'cuda:{gpu}'
        self._cached_ctx = None
        self._cached_params = None
        self._cached_tup = None

    def process(self, request):
        if request.action == MonadAction.Exit:
            yield request
            return
        with torch.cuda.device(self._gpu):
            torch.manual_seed(20)
            for kernel_name, perf_number, kig in from self.profile(request):
                yield request.pass(kernel_name=kernel_name,
                                   perf_number=perf_number,
                                   kig_dic=kig)

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
