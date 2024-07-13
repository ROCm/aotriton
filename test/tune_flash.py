#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch
import json
import sys
import subprocess
import queue
import multiprocessing
from multiprocessing import Process, Queue
import argparse
import itertools
import os
import time
import math

from rocm_arch import rocm_get_gpuarch
from attn_torch_function import (
    DEFAULT_PHILOX_SEED,
    DEFAULT_PHILOX_OFFSET,
    attention,
    debug_fill_dropout_rng,
    AttentionExtraArgs
)
from _common_test import SdpaContext, SdpaParams

_DEBUG_SKIP_TUNE_BACKWARD = True

class ArgArchVerbose:
    def __init__(self, args):
        self._args = args
        self._arch = rocm_get_gpuarch()

    @property
    def verbose(self):
        return self._args.verbose

class TunerWorker(ArgArchVerbose):
    def __init__(self, args):
        super().__init__(args)
        self._tqdm_position = 0
        self._gpu_device = 'cuda'

    def profile_single_config(self, tup, *, shard_prefix=''):
        BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, return_encoded_softmax, dtype, bias_type = tup
        if seqlen_q > 8192 and seqlen_k > 8192:
            N_HEADS = 1
        if causal and seqlen_q != seqlen_k:
            print('FA does not support accept casual=True when seqlen_q != seqlen_k. Skipping')
            return 'Skip', None, None
        if causal and bias_type != 0:
            print('FA does not support accept casual=True when bias_type != 0. Skipping')
            return 'Skip', None, None
        torch.cuda.empty_cache()
        '''
        Create reference dropout_mask
        '''
        if dropout_p > 0.0:
            rdims = (BATCH, N_HEADS, seqlen_q, seqlen_k)
            r = torch.empty(rdims, device=self._gpu_device, dtype=torch.float32)
            philox_seed = DEFAULT_PHILOX_SEED
            philox_offset = DEFAULT_PHILOX_OFFSET
            debug_fill_dropout_rng(r, philox_seed, philox_offset)
            mask = r > dropout_p
            torch.cuda.synchronize()
            del r
        else:
            mask = None
        torch.cuda.empty_cache()
        a = self._args
        '''
        Create SdpaContext for testing
        '''
        ctx = SdpaContext(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype,
                          bias_type=bias_type, storage_flip=None, device=self._gpu_device)
        ctx.create_ref_inputs(target_gpu_device=self._gpu_device)
        ctx.set_require_grads(skip_db=True)
        q, k, v, b = ctx.dev_tensors
        sdpa_params = SdpaParams(causal=causal, sm_scale=sm_scale, dropout_p=dropout_p, dropout_mask=mask)
        ref_out, _ = ctx.compute_ref_forward(sdpa_params)

        '''
        Now, enable autotune (C++ form), enable output validation
        '''
        def fwd_validator(tri_out):
            is_allclose, adiff, _, _ = ctx.validate_with_reference(tri_out, None, no_backward=True)
            '''
            if not is_allclose:
                import numpy as np
                err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out - tri_out)).cpu().numpy(), ref_out.shape)
                print(f'{err_idx=}')
                print(f'{tri_out[err_idx]=}')
                print(f'{ref_out[err_idx]=}')
            '''
            return is_allclose

        ext = AttentionExtraArgs(return_encoded_softmax=return_encoded_softmax,
                autotune=True,
                return_autotune=True,
                autotune_validator=fwd_validator,
                cpp_autotune_tqdm_position=self._tqdm_position,
                cpp_autotune_tqdm_prefix=f'{shard_prefix}{tup}',
                )
        tri_out, encoded_softmax, best_configs = attention(q, k, v, b, causal, sm_scale, dropout_p, ext)
        if self.verbose:
            print('Returned best configs')
            for kernel_name, best in best_configs:
                # print(f'{kernel_name=} {best.kwargs=} {best.num_warps=} {best.num_stages=}')
                print(f'{kernel_name=}')
        if not _DEBUG_SKIP_TUNE_BACKWARD:
            dout = torch.randn_like(q)
            tri_out.backward(dout)
            if self.verbose:
                print('Returned best configs after backward')
                for kernel_name, best in best_configs:
                    print(f'{kernel_name=}')
        head_dim_rounded = 2 ** (D_HEAD - 1).bit_length()
        head_dim_rounded = max(16, head_dim_rounded)
        inputs = {
            'Q_dtype': str(dtype),
            'N_HEADS': N_HEADS,
            'D_HEAD': D_HEAD,
            'max_seqlen_q': seqlen_q,
            'max_seqlen_k': seqlen_k,
            'CAUSAL': causal,
            'RETURN_ENCODED_SOFTMAX': return_encoded_softmax,
            'BLOCK_DMODEL': head_dim_rounded,
            'ENABLE_DROPOUT' : dropout_p > 0.0,
            'PADDED_HEAD' : head_dim_rounded != D_HEAD,
            'BIAS_TYPE' : bias_type,
        }
        return 'Success', inputs, best_configs

class IPCTunerWorker(TunerWorker):
    END_OF_QUEUE_OBJECT = (-1, None)

    def set_shard(self, shard):
        self._tqdm_position = shard
        self._gpu_device = f'cuda:{shard}'

    def do_profile(self, ipc_read, ipc_write):
        a = self._args
        shard, total_shards = ipc_read.get()
        print(f'{shard=} {total_shards=}')
        shard_prefix= '' if shard is None else f'[Shard {shard:02d}/{total_shards:02d}]'
        self.set_shard(shard)
        with torch.cuda.device(shard):
            while True:
                try:
                    i, tup = ipc_read.get()
                    if i == -1 and tup is None:
                        break
                    prefix = shard_prefix + f'[{i:06d}]'
                    action, inputs, best_configs = self.profile_single_config(tup, shard_prefix=prefix)
                    if action == 'Success':
                        ipc_write.put((i, inputs, best_configs))
                except ValueError:  # mp.Queue closed
                    break
        '''
        with torch.cuda.device(shard):
            for i, tup in enumerate(self.gen()):
                if i % total_shards != shard:
                    continue
                print(f"{shard_prefix}[{i:06d}] Handling {tup}")
                if a.continue_from is not None and i < a.continue_from:
                    continue
                if a.stop_at is not None and i > a.stop_at:
                    break
                if a.dry_run:
                    continue
                action, inputs, best_configs = self.profile_single_config(tup)
                if action == 'Success':
                    ipc_write.put((inputs, best_configs))
        ipc_write.put((None, shard))
        '''

class DbAccessor(ArgArchVerbose):
    KERNEL_FAMILY = 'FLASH'
    END_OF_QUEUE_OBJECT = (-1, None, None)

    def create_dbp(self):
        a = self._args
        dbargs = ['python3', '-m', 'v2python.table_tool']
        if self.verbose:
            dbargs += ['-v']
        dbargs += ['-f', self._args.db_file, '-k', self.KERNEL_FAMILY]
        if a.create_table_only:
            dbargs += ['--action', 'createtableonly']
        self._dbp = subprocess.Popen(dbargs,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True)
        os.set_blocking(self._dbp.stdout.fileno(), False)
        os.set_blocking(self._dbp.stderr.fileno(), False)

    def pipe_from_ipc(self, ipc_read):
        self.create_dbp()
        while True:
            try:
                i, inputs, best_configs = ipc_read.get(timeout=30)
                if i == -1 and inputs is None:
                    print('[DbAccessor] No more tasks. Exiting')
                    break
                self.pipe_configs(inputs, best_configs, prefix=f'[{i:06d}]')
            except ValueError:  # mp.Queue closed
                break
        self.stop()
        return

    def pipe_configs(self, inputs, best_configs, *, prefix=''):
        for kernel_name, best in best_configs:
            j = self.translate_config(inputs, kernel_name, best)
            js = json.dumps(j, separators=(',', ':'))
            print(f'{prefix}Piping to db process {js}')
            print(js, file=self._dbp.stdin, flush=True)
            self.splice_pipes()

    def splice_pipes(self):
        nattempts = 10 if self.verbose else 1
        for i in range(nattempts):
            while True:
                line = self._dbp.stdout.readline()
                if line:
                    print(line, end='')
                else:
                    if self.verbose:
                        time.sleep(0.1)
                    break

        for i in range(nattempts):
            while True:
                line = self._dbp.stderr.readline()
                if line:
                    print(line, end='', file=sys.stderr)
                else:
                    if self.verbose:
                        time.sleep(0.1)
                    break
        sys.stdout.flush()
        sys.stderr.flush()

    def translate_config(self, inputs, kernel_name, best):
        tuning_result = {
            'arch' : self._arch,
            'kernel_name' : kernel_name,
            'inputs' : inputs,
            'tuned_kernel' : best.psels,
            'compiler_options' : best.copts,
        }
        return tuning_result

    def stop(self):
        self._dbp.stdin.close()
        print("Waiting for database process to terminate")
        self._dbp.wait()
        self.splice_pipes()

class TunerManager(ArgArchVerbose):

    def gen(self):
        a = self._args
        yield from itertools.product(a.batch, a.n_heads, a.d_head, a.seqlen_q, a.seqlen_k, a.causal, a.sm_scale, a.dropout_p, a.return_encoded_softmax, a.dtype, a.bias_type)

    def gen_itup(self):
        a = self._args
        for i, tup in enumerate(self.gen()):
            # print(f"[{i:06d}] Handling {tup}")
            if a.continue_from is not None and i < a.continue_from:
                continue
            if a.stop_at is not None and i > a.stop_at:
                break
            if a.dry_run:
                continue
            yield i, tup

    def profile_all(self):
        a = self._args
        dba = DbAccessor(a)
        if a.use_multigpu is None:
            for i, tup in self.gen_itup():
                action, inputs, best_configs = self.profile_single_config(tup)
                if action == 'Success':
                    dba.pipe_configs(inputs, best_configs)
            dba.stop()
            return
        shards = list([i for i in range(torch.cuda.device_count())]) if -1 in a.use_multigpu else a.use_multigpu
        ipc_write = Queue()
        ipc_worker_out = Queue()
        ipc_tuners = [IPCTunerWorker(self._args) for i in shards]
        workers = [Process(target=worker.do_profile, args=(ipc_write, ipc_worker_out)) for worker in ipc_tuners]
        db_accessor = Process(target=dba.pipe_from_ipc, args=(ipc_worker_out,))

        '''
        Start processes
        '''
        nlive_processes = len(workers)
        for i, p in enumerate(workers):
            ipc_write.put((i, nlive_processes))
        for p in workers:
            p.start()
        db_accessor.start()
        '''
        Dispatching tasks to ipc_write
        '''
        for i, tup in self.gen_itup():
            obj = (i, tup)
            any_process_alive = self.write_to_ipc(ipc_write, obj, workers)
            if not any_process_alive:
                break
        nlive_processes = self.scan_live_processes(workers)
        for i in range(nlive_processes):
            self.write_to_ipc(ipc_write, IPCTunerWorker.END_OF_QUEUE_OBJECT, workers)
        ipc_write.close()
        """
        while nlive_processes > 0:
            try:
                inputs, best_configs = ipc_worker_out.get(timeout=30)
                # print(f'{inputs=}')
                # print(f'{best_configs=}')
                if inputs is None:
                    shard = best_configs
                    nlive_processes -= 1
                    print(f'Shard {shard} has completed all tasks. Updated {nlive_processes=}')
                    continue
                self.pipe_configs(inputs, best_configs)
            except queue.Empty:
                print("Timed out. Re-scan live processes")
                # "watchdog"
        """
        for p in workers:
            p.join()
        print('All workers joined')
        ipc_worker_out.put(DbAccessor.END_OF_QUEUE_OBJECT)
        ipc_worker_out.close()

    def write_to_ipc(self, ipc_write, obj, workers):
        while True:
            try:
                ipc_write.put(obj, timeout=60)
                return True
            except queue.Full:
                print("Task Queue Full. Re-scan live processes")
                nlive_processes = self.scan_live_processes(workers)
                print(f"{nlive_processes=}")
                if nlive_processes == 0:
                    print("PANIC: All Processes Died")
                    return False

    def scan_live_processes(self, workers):
        nlive_processes = 0
        for i, p in enumerate(workers):
            nlive_processes += 1 if p.is_alive() else 0
        return nlive_processes

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch', type=int, nargs=1, default=[1], help='(Not a functional) Batch size.')
    p.add_argument('--n_heads', type=int, nargs=1, default=[12], help='(Not a functional) Number of heads')
    p.add_argument('--sm_scale', type=float, nargs=1, default=[1.2], help='(Not a functional) Softmax Scale')
    p.add_argument('--return_encoded_softmax', type=bool, default=[False],
                   help="(A functional for debugging) kernel that returns softmax(dropout(QK')) to validate the correctness of dropout")
    p.add_argument('--d_head', type=int, nargs='+', default=[16,32,64,128,256], help='Head dimensions.')
    p.add_argument('--seqlen_q', type=int, nargs='+', default=[4,8,16,32,64,128,256,1024,2048,4096,8192,16384], help='Sequence length of Q.')
    p.add_argument('--seqlen_k', type=int, nargs='+', default=[4,8,16,32,64,128,256,1024,2048,4096,8192,16384], help='Sequence length of K/V.')
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
    p.add_argument('--db_file', type=str, required=True, help="Sqlite Database file")
    p.add_argument('--create_table_only', action='store_true', help="Do not insert data, only create tables. Used for schema updates.")
    p.add_argument('--use_multigpu', type=int, nargs='+', default=None, help='Profiling on multiple GPUs. Passing -1 for all GPUs available to pytorch.')
    args = p.parse_args()
    args.dtype = [ getattr(torch, t) for t in args.dtype ]
    args.causal = [ bool(c) for c in args.causal ]
    # assert args.causal == [False], f'{args.causal=} {args.return_encoded_softmax=}'
    return args

def main():
    multiprocessing.set_start_method('spawn')  # Otherwise torch complains
    args = parse()
    tuner = TunerManager(args)
    tuner.profile_all()

if __name__ == '__main__':
    main()
