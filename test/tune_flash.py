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
from collections import defaultdict
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
            return False
        kig.kernel_index += 1
        return True

class FlashTunerSource(MonadService):
    def gen(self):
        a = self._args
        yield from itertools.product(a.batch, a.n_heads, a.d_head, a.seqlen_q, a.seqlen_k, a.causal, a.sm_scale, a.dropout_p, a.return_encoded_softmax, a.dtype, a.bias_type)

    def init(self, _):
        pass

    def is_valid_time(self, timing):
        return isinstance(timing, list) and len(timing) == 3

    def is_inf_time(self, timing):
        return timing is None or isinstance(timing, float) and math.isinf(timing)

    def update_continue_dict(self, j, cd):
        result = j['result']
        if result == 'skipped':
            return
        kn = j['kernel_name']
        kig = j['_debug_kernel_index']
        kit = j['_debug_total_number_of_kernels']
        task_id = j['_debug_task_id']
        if task_id not in cd:
            cd[task_id] = TuningResult(tup=None, kig_dict=self.create_kig_dict())
        # print(f'{cd=}', flush=True)
        # print(f'{cd[task_id]=}', flush=True)
        kig_dict = cd[task_id].kig_dict
        target = kig_dict[kn]
        target.kernel_index = max(target.kernel_index, kig)
        target.total_number_of_kernels = kit
        timing = j['time']
        if self.is_valid_time(timing):
            target.passed_kernels += 1
        elif self.is_inf_time(timing):
            target.failed_kernels += 1
        else:
            target.uncertain_errors += 1
        # if task_id == 0:
        #     self.print(f'update cd[0] to {cd[0]} {kn=}')
        all_complete = True
        for kn in KERNEL_PRECEDENCE:
            if kig_dict[kn].kernel_index + 1 < kig_dict[kn].total_number_of_kernels:
                all_complete = False
                cd[task_id].profiled_kernel_name = kn
                # if task_id == 0:
                #     self.print(f'Kernel {kn} incomplete: {kig_dict[kn].kernel_index + 1=} not < {kig_dict[kn].total_number_of_kernels=}')
                break
        if all_complete:
            # if task_id == 0:
            #     self.print(f'task 0 complete')
            del cd[task_id]
            return task_id
        return None

    def process(self, _):
        a = self._args
        skip_set = set()
        continue_dict = {}
        if a.continue_from_json_file and a.json_file is not None and a.json_file.is_file():
            # TODO: skipset
            with open(a.json_file, 'r') as f:
                for line in f.readlines():
                    j = json.loads(line)
                    task_id = j['_debug_task_id']
                    if task_id in skip_set:
                        continue
                    skip = self.update_continue_dict(j, continue_dict)
                    if skip is not None:  # Must test is None, otherwise skip=False when task_id=0
                        skip_set.add(skip)
        self.print(f'{skip_set=}')
        self.print(f'{continue_dict=}')

        for i, tup in enumerate(self.gen()):
            self.print(f"[{i:06d}] gen_itup {tup}")
            batch, n_heads, d_head, seqlen_q, seqlen_k = tup[:5]
            if a.selective_set:
                if i in a.selective_set:
                    payload = TuningResult(tup=tup, kig_dict=self.create_kig_dict())
                    progress_in_db = MonadMessage(task_id=i, action=MonadAction.Pass, source='source', payload=payload)
                    yield self.monad.next_kernel(progress_in_db)
                    continue
                continue
            if a.continue_from is not None and i < a.continue_from:
                continue
            if i in skip_set:
                continue
            if seqlen_q > a.max_seqlen_q or seqlen_k > a.max_seqlen_k:
                continue
            if a.stop_at is not None and i > a.stop_at:
                break
            kig_dict = self.create_kig_dict()
            if i in continue_dict:
                known_kig_dict = continue_dict[i].kig_dict
                for kn in KERNEL_PRECEDENCE:
                    # print(f'{kig_dict=}')
                    # print(f'{kig_dict[kn]=}')
                    # print(f'{kig_dict[kn].kernel_index=}')
                    # print(f'{known_kig_dict=}')
                    # print(f'{known_kig_dict[kn]=}')
                    # print(f'{known_kig_dict[kn].kernel_index=}', flush=True)
                    kig_dict[kn] = deepcopy(known_kig_dict[kn])
                    kig_dict[kn].last_success_kernel = known_kig_dict[kn].kernel_index
            payload = TuningResult(tup=tup, kig_dict=kig_dict)
            progress_in_db = MonadMessage(task_id=i, action=MonadAction.Pass, source='source', payload=payload)
            if i in continue_dict:
                if -1 in a.debug_skip_next_kernel or i in a.debug_skip_next_kernel:
                    progress_in_db = self.monad.next_kernel(progress_in_db)
            yield self.monad.next_kernel(progress_in_db)
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
    app = mptune.tui.TunerApp(args=manager._args,
                              watchdog=manager.run_watchdog,
                              info_queue=info_queue,
                              src=src,
                              workers=workers,
                              dbaccessor=dbaccessor)
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
    p.add_argument('--seqlen_q', type=int, nargs='+', default=[4,8,16,32,64,128,256,512,1024,2048,4096,8192], help='Sequence length of Q.')
    p.add_argument('--seqlen_k', type=int, nargs='+', default=[4,8,16,32,64,128,256,512,1024,2048,4096,8192], help='Sequence length of K/V.')
    p.add_argument('--max_seqlen_q', type=int, default=8192, help='A neat way to limit max value of --seqlen_q.')
    p.add_argument('--max_seqlen_k', type=int, default=8192, help='A neat way to limit max value of --seqlen_k.')
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
    p.add_argument('--selective_set', type=int, default=None, nargs='*', help="Only use the given task ids. Will override other options like --continue_from")
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
    p.add_argument('--debug_skip_next_kernel',
                   type=int,
                   default=[],
                   nargs='*',
                   help='''[DEBUG] Skip the next untuned kernel for the given task ids.
                           Passing -1 to indicate all task ids.
                           For severely broken kernels, it is possble the process simply hangs
                           indefinitely without kernel driver intervention or GPU reset.
                           This option allow skipping next kernel relative to json record,
                           which are usually the faulty kernel.'''
                  )
    p.add_argument('--debug_headless', action='store_true', help='Set headless mode for textual.App.')
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
    # monitor_thread = Thread(target=tuner.monitor)
    # monitor_thread.start()
    app.run(mouse=False, inline=True, headless=args.debug_headless)
    # monitor_thread.join()

if __name__ == '__main__':
    main()
