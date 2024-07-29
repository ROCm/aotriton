#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import itertools
from abc import abstractmethod

class TunerManager(ArgArchVerbose):

    @abstractmethod
    def factory_dbaccessor(self):
        pass

    @abstractmethod
    def factory_worker(self, nth_worker : int, gpu_device : int):
        pass

    def gen_itup(self):
        a = self._args
        skip_set = set()
        if a.continue_from_json_file and a.json_file is not None and a.json_file.is_file():
            with open(a.json_file, 'r') as f:
                for line in f.readlines():
                    j = json.loads(line)
                    skip_set.add(j['_debug_task_id'])
        for i, tup in enumerate(self.gen()):
            # print(f"[{i:06d}] gen_itup {tup}")
            if a.continue_from is not None and i < a.continue_from:
                continue
            if i in skip_set:
                continue
            if a.stop_at is not None and i > a.stop_at:
                break
            yield i, tup

    def profile_all(self):
        a = self._args
        dba = self.factory_dbaccessor()
        if a.use_multigpu is None:
            dba.create_dbp(self.KERNEL_FAMILY)
            worker = self.factory_worker()
            worker.do_profile(dba, self.gen_itup)
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
        for shard, p in zip(shards, workers):
            ipc_write.put((shard, nlive_processes))
        for p in workers:
            p.start()
        db_accessor.start()
        '''
        Dispatching tasks to ipc_write
        '''
        for i, tup in self.gen_itup():
            obj = (i, tup)
            self.print(f"write_to_ipc {obj}")
            any_process_alive = self.write_to_ipc(ipc_write, obj, workers)
            if not any_process_alive:
                self.print(f"{any_process_alive=}, leave the generator loop")
                break
        self.print("Task Generator Complete")
        nlive_processes = self.scan_live_processes(workers)
        for i in range(nlive_processes):
            self.write_to_ipc(ipc_write, IPCTunerWorker.END_OF_QUEUE_OBJECT, workers)
            self.print(f"write_to_ipc {IPCTunerWorker.END_OF_QUEUE_OBJECT}")
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
        ipc_write.close()
        print('All workers joined')
        ipc_worker_out.put(DbAccessor.END_OF_QUEUE_OBJECT)
        ipc_worker_out.close()
        db_accessor.join()
        print('Db accessor joined')
        # Otherwise current process may block if any child died
        ipc_write.cancel_join_thread()
        ipc_worker_out.cancel_join_thread()

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
