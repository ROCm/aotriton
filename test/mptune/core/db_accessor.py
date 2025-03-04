#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..core import MonadAction, MonadMessage, Monad, MonadService
from abc import abstractmethod
import subprocess
import os
import sys
import json

class DbService(MonadService):
    KERNEL_FAMILY = None

    def init(self, num_workers):
        self._num_workers = num_workers
        self._exit_workers = 0
        a = self._args
        if a.json_file is not None and not a.dry_run:
            if a.db_file:
                assert a.json_file != a.db_file
            self._jsonfile = open(a.json_file, 'a' if a.continue_from_json_file else 'w')
        else:
            self._jsonfile = None
        if a.db_file:
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
        else:
            self._dbp = None

    def process(self, request):
        if request.action == MonadAction.Exit:
            self._exit_workers += 1
            if self._exit_workers >= self._num_workers:
                yield request.forward(self.monad)
            return
        inputs = self.constrcut_inputs(request)
        if request.action == MonadAction.Pass:
            payload = request.payload
            autotune_result = self.analysis_result(request)
            self.pipe_configs(payload.profiled_kernel_name,
                              inputs,
                              autotune_result,
                              _debug_task_id=request.task_id)
            yield request.forward(self.monad)
        if request.action == MonadAction.Skip:
            self.pipe_skipped_configs(inputs, request.task_id)
        if request.action in [MonadAction.Skip, MonadAction.DryRun]:
            yield request.forward(self.monad)

    def cleanup(self):
        if self._jsonfile is not None:
            self._jsonfile.close()
        if self._dbp:
            self._dbp.stdin.close()
            self.print("Waiting for database process to terminate")
            self._dbp.wait()
            self.splice_pipes()

    @abstractmethod
    def constrcut_inputs(self, request):
        # Access request.tup
        pass

    @abstractmethod
    def analysis_result(self, request):
        # Access request.perf_number
        pass

    def pipe_configs(self, kernel_name, inputs, autotune_result, _debug_task_id):
        j = self.translate_config(kernel_name, inputs, autotune_result)
        if _debug_task_id is not None:
            j['_debug_task_id'] = _debug_task_id
        js = json.dumps(j, separators=(',', ':'))
        print(js, file=self._jsonfile, flush=True)
        if self._dbp:
            print(js, file=self._dbp.stdin, flush=True)
            self.splice_pipes()

    def pipe_skipped_configs(self, inputs, _debug_task_id):
        skipped_result = {
            'arch' : self._arch,
            'inputs' : inputs,
            '_debug_task_id' : _debug_task_id,
            'result' : 'skipped',
        }
        js = json.dumps(skipped_result, separators=(',', ':'))
        if self._jsonfile is not None:
            print(js, file=self._jsonfile, flush=True)

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

    def translate_config(self, kernel_name, inputs, atr : 'AutotuneResult'):
        tuning_result = {
            'arch' : self._arch,
            'kernel_name' : kernel_name,
            'inputs' : inputs,
            'result' : 'tuned',
            'tuned_kernel' : atr.psels,
            'compiler_options' : atr.copts,
            'ut_passed' : atr.ut_passed,
            'time' : atr.time,
            'adiffs' : atr.adiffs,
            'target_fudge_factors' : atr.target_fudge_factors,
            'sgpr_spill_count' : atr.sgpr_spill_count,
            'vgpr_spill_count' : atr.vgpr_spill_count,
            'hip_status' : str(atr.hip_status),
            '_debug_kernel_index' : atr.kernel_index,
            '_debug_total_number_of_kernels' : atr.total_number_of_kernels,
        }
        return tuning_result
