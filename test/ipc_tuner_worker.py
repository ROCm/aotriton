#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from tuner_worker import TunerWorker

class IPCTunerWorker(TunerWorker):
    END_OF_QUEUE_OBJECT = (-1, None)

    '''
    Initialize multiprocessing related variables
    '''
    def init_mp(self, shard):
        self._shard = shard
        self._tqdm_bar = tqdm(total=1, unit="configs", position=shard+1, leave=True)
        self._gpu_device = f'cuda:{shard}'
        self._cached_gpukernel_process = {}

    def clean_mp(self, shard):
        self.clean_cached_gpukernel_process()

    def do_profile(self, ipc_read, ipc_write):
        a = self._args
        shard, total_shards = ipc_read.get()
        print(f'{shard=} {total_shards=}')
        shard_prefix= '' if shard is None else f'[Shard {shard:02d}/{total_shards:02d}]'
        self.init_mp(shard)
        with torch.cuda.device(shard):
            while True:
                try:
                    i, tup = ipc_read.get()
                    self.print(f'ipc_read {i} {tup}')
                    if i == -1 and tup is None:
                        break
                    prefix = shard_prefix + f'[{i:06d}]'
                    action, inputs, best_configs = self.profile_single_config(tup,
                                                                              prefix=prefix,
                                                                              shard=self._shard)
                    ipc_write.put((i, action, inputs, best_configs))
                except ValueError:  # mp.Queue closed
                    break
        self.clean_mp(shard)
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

