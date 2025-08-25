#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..core import MonadAction, MonadMessage, Monad, MonadService
from abc import abstractmethod

class ProfilerEarlyExit(Exception):
    def __init__(self, msg):
        super().__init__()
        self.msg = msg

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
        self.print(f'Worker receive {request}')
        with torch.cuda.device(self._gpu):
            torch.manual_seed(20)
            item = 0
            gen = self.profile(request)
            try:
                for kernel_name, perf_number, kig in gen:
                    yield request.set_action(MonadAction.Pass) \
                                 .forward(self.monad) \
                                 .update_payload(profiled_kernel_name=kernel_name,
                                                 perf_number=perf_number,
                                                 kig_dict=kig)
                    # print(f'Worker yield {kig=}')
                    self.print(f'Worker yield {item=}')
                    item += 1
            except RuntimeError as e:
                self.print(f'{self.monad.identifier} RuntimeError {e}')
                raise e
            except StopIteration as e:
                self.print(f'{self.monad.identifier}')
            except ProfilerEarlyExit as e:
                assert e.msg.action != MonadAction.Pass
                yield e.msg
        self.print(f'Worker complete {request}')

    def cleanup(self):
        pass

    def hit_cache(self, tup):
        if self._cached_tup == tup:
            return self._cached_ctx, self._cached_params
        if self._cached_ctx is not None:
            del self._cached_ctx
            self._cached_ctx = None
        if self._cached_params is not None:
            del self._cached_params  # Must, dropout_p is there
            self._cached_params = None
        self.create_ctx_cache(tup)
        return self._cached_ctx, self._cached_params

    @abstractmethod
    def create_ctx_cache(self, tup):
        pass

    @abstractmethod
    def profile(self, request):
        pass
