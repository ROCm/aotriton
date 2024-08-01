#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
# FIXME: Should set PYTORCH_NO_HIP_MEMORY_CACHING=1 as well but need to wait for
# https://github.com/pytorch/pytorch/issues/114534
os.environ['HSA_SVM_GUARD_PAGES'] = '1'
os.environ['HSA_DISABLE_FRAGMENT_ALLOCATOR'] = '1'

import pytest
import json
import sys
import subprocess
from threading import Thread
import multiprocessing
from multiprocessing import Process
import argparse
import time
import math
import itertools
from copy import deepcopy
from pathlib import Path
from rocm_arch import rocm_get_gpuarch, rocm_get_allarch
import mptune
from mptune.core import (
    ArgArchVerbose,
    Monad,
    MonadService,
    MonadMessage,
    MonadAction,
    TunerManager,
    TuningResult,
    KernelIndexProress,
)

'''
|----------------------------------------------------
|FlashTup -> n * FlashTunerWorker -> FlashDbService |
|    |           /                       /          |
|FlashObserver  ~~~~~~~~~~~~~~~~~~~~~~~~            |
|------------------- Watch Dog ----------------------
                 FlashTunerManager
'''

KERNEL_PRECEDENCE = ['attn_fwd', 'bwd_kernel_dk_dv', 'bwd_kernel_dq']

class FlashSourceMonad(Monad):
    def service_factory(self):
        return FlashTunerSource(self._args, self)

    def next_kernel(self, msg : MonadMessage) -> MonadMessage:
        inkig = msg.payload.kig_dict
        outkig = deepcopy(inkig)
        for kn in KERNEL_PRECEDENCE:
            if self.next_index(outkig[kn]):
                break
        return msg.update_payload(kig_dict=outkig)

    def next_index(self, kig: KernelIndexProress) -> bool:
        if kig.kernel_index >= kig.total_number_of_kernels:
            return false
        kig.kernel_index += 1
        return True

class FlashTunerSource(MonadService):
    def gen(self):
        a = self._args
        yield from itertools.product(a.batch, a.n_heads, a.d_head, a.seqlen_q, a.seqlen_k, a.causal, a.sm_scale, a.dropout_p, a.return_encoded_softmax, a.dtype, a.bias_type)

    def init(self, _):
        pass

    def process(self, _):
        a = self._args
        skip_set = set()
        if a.continue_from_json_file and a.json_file is not None and a.json_file.is_file():
            # assert False, 'Implement new skipset algorithm!'
            # TODO: skipset
            with open(a.json_file, 'r') as f:
                for line in f.readlines():
                    j = json.loads(line)
                    skip_set.add(j['_debug_task_id'])

        for i, tup in enumerate(self.gen()):
            self.print(f"[{i:06d}] gen_itup {tup}")
            if a.continue_from is not None and i < a.continue_from:
                continue
            if i in skip_set:
                continue
            if a.stop_at is not None and i > a.stop_at:
                break
            payload = TuningResult(tup=tup, kig_dict=self.create_kig_dict())
            yield MonadMessage(task_id=i, action=MonadAction.Pass, source='source', payload=payload)
        self.print(f"gen_itup Exit")
        # Note: main_loop should handle Exit after forwarding all
        # object yield from MonadService.progress
        for i in range(len(self._args.use_multigpu)):
            yield MonadMessage(task_id=None, action=MonadAction.Exit, source='source')

    def cleanup(self):
        pass

    def create_kig_dict(self):
        return { kn : KernelIndexProress() for kn in KERNEL_PRECEDENCE }

class FlashTunerManager(TunerManager):
    def factory_state_tracker(self):
        return mptune.core.StateTracker(self._args)

    def factory_source(self, side_channel):
        return FlashSourceMonad(self._args,
                                identifier='source',
                                side_channel=side_channel)

    def factory_dbaccessor(self, num_workers, side_channel):
        return mptune.flash.DbMonad(self._args,
                                    identifier='dbaccessor',
                                    side_channel=side_channel,
                                    init_object=num_workers)

    def factory_worker(self, nth_worker : int, gpu_device : int, side_channel):
        return mptune.flash.TunerMonad(self._args,
                                       identifier=f'worker_{nth_worker}_on_gpu_{gpu_device}',
                                       side_channel=side_channel,
                                       init_object=(nth_worker, gpu_device),
                                       )

    def factory_ui(self, state_tracker, src, workers, dbaccessor):
        return mptune.tui.TunerApp(state_tracker.get_ui_update_queue(),
                                   src,
                                   workers,
                                   dbaccessor)

def make_ui(manager : TunerManager):
    info_queue = manager._state_tracker.get_ui_update_queue()
    src = manager._src
    workers = manager._workers
    dbaccessor = manager._dba
    app = mptune.tui.TunerApp(args=manager._args, info_queue=info_queue, src=src, workers=workers, dbaccessor=dbaccessor)
    return app

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch', type=int, nargs=1, default=[1], help='(Not a functional) Batch size.')
    p.add_argument('--n_heads', type=int, nargs=1, default=[12], help='(Not a functional) Number of heads')
    p.add_argument('--sm_scale', type=float, nargs=1, default=[1.2], help='(Not a functional) Softmax Scale')
    p.add_argument('--return_encoded_softmax', type=bool, default=[False],
                   help="(A functional for debugging) kernel that returns softmax(dropout(QK')) to validate the correctness of dropout")
    p.add_argument('--d_head', type=int, nargs='+', default=[16,32,64,128,256], help='Head dimensions.')
    # p.add_argument('--seqlen_q', type=int, nargs='+', default=[4,8,16,32,64,128,256,1024,2048,4096,8192], help='Sequence length of Q.')
    # p.add_argument('--seqlen_k', type=int, nargs='+', default=[4,8,16,32,64,128,256,1024,2048,4096,8192], help='Sequence length of K/V.')
    p.add_argument('--seqlen_q', type=int, nargs='+', default=[4,8,16,32,64,128,256,512,1024], help='Sequence length of Q.')
    p.add_argument('--seqlen_k', type=int, nargs='+', default=[4,8,16,32,64,128,256,512,1024], help='Sequence length of K/V.')
    p.add_argument('--causal', type=int, nargs='+', default=[True,False], choices=[0, 1], help='Causal mask. (Use 0/1 for False/True')
    p.add_argument('--dropout_p', type=float, nargs='+', default=[0.5, 0.0], help='Probablity to dropout (0 to disable).')
    p.add_argument('--dtype', type=str, nargs='+',
                   default=['float16', 'bfloat16', 'float32'],
                   choices=['float16', 'bfloat16', 'float32'],
                   help='Datatype to profile.')
    p.add_argument('--bias_type', type=int, nargs='+', default=[0, 1], choices=[0, 1], help='Bias types to profile, 0: None, 1: Matrix.')
    p.add_argument('--verbose', action='store_true', help='Verbose')
    p.add_argument('--validate',
                   action='store_true', help='Validate the correctness of the output to avoid faulty autotune configs')
    p.add_argument('--dry_run', action='store_true', help="Print parameter combinations without running tests")
    p.add_argument('--continue_from', type=int, default=None, help="Continue from n-th functional set")
    p.add_argument('--stop_at', type=int, default=None, help="Stop at n-th functional set")
    p.add_argument('--db_file', type=str, default=None, help="Sqlite Database file (not recommended)")
    p.add_argument('--json_file',
                   type=Path,
                   required=True,
                   default=None,
                   help="Json file for record. Disables printing json to stdout")
    p.add_argument('--overwrite_json_file', dest='continue_from_json_file', action='store_false', help="Do NOT \"Append to Json file instead of overwrite, and skip already tested entries.\"")
    p.add_argument('--create_table_only', action='store_true', help="Do not insert data, only create tables. Used for schema updates.")
    p.add_argument('--use_multigpu', type=int, nargs='+', default=None, help='Profiling on multiple GPUs. Passing -1 for all GPUs available to pytorch.')
    p.add_argument('--arch', type=str, default=None, help='[NOT RECOMMENDED TO SET MANUALLY] Override GPU architecture string. Will use first GPU from `rocm_agent_enumerator -name` if not provided.')
    p.add_argument('--confirm_to_override_arch', action='store_true', help='A defensive option to avoid setting --arch unintentionally.')
    args = p.parse_args()
    assert args.return_encoded_softmax == [False], ('Do not support tuning return_encoded_softmax=True. '
            'RETURN_ENCODED_SOFTMAX will be removed in the future and debug_fill_dropout_rng will be preferred choice.')
    args.causal = [ bool(c) for c in args.causal ]
    if args.arch is None:
        args.arch = rocm_get_gpuarch()
    else:
        assert args.confirm_to_override_arch
    if args.use_multigpu is None:
        args.use_multigpu = [0]
    elif -1 in args.use_multigpu:
        args.use_multigpu = list([i for i in range(len(rocm_get_allarch()))])
    # assert args.causal == [False], f'{args.causal=} {args.return_encoded_softmax=}'
    return args

def main():
    assert os.getenv('PYTORCH_NO_CUDA_MEMORY_CACHING', default=0) == 0, 'PYTORCH_NO_HIP_MEMORY_CACHING does not play nicely with torch.multiprocessing. See https://github.com/pytorch/pytorch/issues/114534'
    # We tried
    # torch.multiprocessing.set_start_method('spawn', force=True)  # Otherwise torch complains
    # multiprocessing.set_start_method('spawn')  # "context has already been set"
    args = parse()
    tuner = FlashTunerManager(args)
    tuner.build_graph()
    app = make_ui(tuner)
    tuner.launch_graph()
    monitor_thread = Thread(target=tuner.monitor)
    monitor_thread.start()
    app.run()
    monitor_thread.join()

if __name__ == '__main__':
    main()
