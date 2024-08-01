#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import itertools
from abc import abstractmethod
from multiprocessing.connection import wait as wait_sentinels
from .aav import ArgArchVerbose
from .monad import Monad
from .message import MonadMessage, MonadAction

class TunerManager(ArgArchVerbose):

    @abstractmethod
    def factory_state_tracker(self) -> Monad:
        pass

    @abstractmethod
    def factory_source(self, side_channel) -> Monad:
        pass

    @abstractmethod
    def factory_dbaccessor(self, side_channel) -> Monad:
        pass

    @abstractmethod
    def factory_worker(self, nth_worker : int, gpu_device : int, side_channel) -> Monad:
        pass

    @abstractmethod
    def factory_ui(self, state_tracker, src, workers, dbaccessor):
        pass

    '''
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
    '''

    def build_graph(self):
        a = self._args
        self._torch_gpus = a.use_multigpu
        num_workers = len(self._torch_gpus)
        state_tracker = self.factory_state_tracker()
        side_channel = state_tracker.get_side_channel_input()
        src = self.factory_source(side_channel=side_channel)
        dba = self.factory_dbaccessor(num_workers=num_workers, side_channel=side_channel)
        workers = [self.factory_worker(i, gpu_device, side_channel=side_channel) for i, gpu_device in enumerate(self._torch_gpus)]
        src.bind_1toN(workers)
        for worker in workers:
            worker.bind_Nto1(dba)

        self._all_monads = [state_tracker, src, dba] + workers
        self._state_tracker = state_tracker
        self._src = src
        self._dba = dba
        self._workers = workers
        self._state_tracker.set_monads_to_track([src, dba] + workers)

    def launch_graph(self):
        a = self._args
        self._state_tracker.start()
        self._dba.start()
        # torch_gpus = a.use_multigpu
        # ngpus = len(torch_gpus)
        # [worker.start((gpu, ngpus)) for worker, gpu in zip(self._workers, self._torch_gpus)]
        self._src.start()
        # Workers will get their init objects from _src
        [worker.start() for worker in self._workers]
        for m in self._all_monads:
            self.print(f"{m.identifier=} {m.sentinel=}")
        self._sentinel_to_monad = { monad.sentinel : monad for monad in self._all_monads }

    def monitor(self):
        success_monads = set()
        while True:
            alive_sentinels = []
            for monad in self._all_monads:
                if monad in success_monads:
                    continue
                alive_sentinels.append(monad.sentinel)
            self.print(f'{alive_sentinels=}')
            failures = wait_sentinels(alive_sentinels)
            self.print(f'{failures=}')
            monads = [self._sentinel_to_monad[sentinel] for sentinel in failures]
            failed_monad_ids = [monad.identifier for monad in monads]
            self.print(f'Monitor: {failed_monad_ids=}')
            # Exiting of state_tracker indicates all tasks are done
            if self._state_tracker in monads:
                return
            # Otherwise restart all faulty processes
            for monad in monads:
                if monad.exitcode != 0:
                    print(f'{monad.identifier} exit with code {monad.exitcode}', flush=True)
                    prog = self._state_tracker.ask_for_last_message(monad, exitcode)
                    self._state_tracker.update_ui(prog.clone().set_action(MonadAction.OOB_Died).update_payload(exitcode=monad.exitcode))
                    if monad.identifier.startswith('worker_'):
                        nextone = self._src.next_kernel(prog)
                        monad.restart_with_last_progress(nextone)
                else:
                    monad.join()
                    # TODO: who should notify the state tracker? The
                    #       exiting monad or the observing manager.
                    #       (For now the exiting monad is used)
                    # self._state_tracker.confirm_exit(monad)
                    success_monads.add(monad)

    """
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

        '''
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
        '''

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
    """

    '''
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
    '''

    """
    def scan_live_processes(self, workers):
        nlive_processes = 0
        for i, p in enumerate(workers):
            nlive_processes += 1 if p.is_alive() else 0
        return nlive_processes
    """
