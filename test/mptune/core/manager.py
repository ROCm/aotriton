#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import itertools
import time
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
        self._success_monads = set()

    def pulling_sentinels(self, alive_monads):
        ret = []
        restarting = []
        self.print(f'monitor time.sleep returns')
        for monad in alive_monads:
            try:
                if monad.exitcode is not None:
                    ret.append(monad)
            except ValueError:
                restarting.append(monad)
        # exitcodes = [monad.exitcode for monad in alive_monads]
        # self.print(f'{exitcodes=}')
        return ret, restarting

    def run_watchdog(self):
        alive_sentinels = []
        alive_monads = []
        for monad in self._all_monads:
            if monad in self._success_monads:
                continue
            # alive_sentinels.append(monad.sentinel)
            alive_monads.append(monad)
        self.print(f'monitor {alive_sentinels=}')
        try:
            # failures = wait_sentinels(alive_sentinels, timeout=0.1)
            failures, restarting = self.pulling_sentinels(alive_monads)
            self.print(f'monitor wait_sentinels {failures=}')
        except Exception as e:
            self.print(f'monitor wait_sentinels timeout or Exception {e}')
            return
        if not failures:
            return
        # monads = [self._sentinel_to_monad[sentinel] for sentinel in failures]
        monads = failures
        failed_monad_ids = [monad.identifier for monad in monads]
        self.print(f'watchdog: {failed_monad_ids=}')
        # Exiting of state_tracker indicates all tasks are done
        if self._state_tracker in monads:
            self.print(f'Monitor exits')
            return
        # Otherwise restart all faulty processes
        for monad in monads:
            if monad.exitcode != 0:
                print(f'{monad.identifier} exit with code {monad.exitcode}', flush=True)
                prog = self._state_tracker.ask_for_last_message(monad)
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
                self._success_monads.add(monad)

    '''
    def monitor(self):
        while True:
            alive_sentinels = []
            alive_monads = []
            for monad in self._all_monads:
                if monad in self._success_monads:
                    continue
                alive_sentinels.append(monad.sentinel)
                alive_monads.append(monad)
            self.print(f'monitor {alive_sentinels=}')
            try:
                # failures = wait_sentinels(alive_sentinels, timeout=0.1)
                failures = self.pulling_sentinels(alive_monads)
                self.print(f'monitor wait_sentinels {failures=}')
            except Exception as e:
                self.print(f'monitor wait_sentinels timeout or Exception {e}')
                continue
            if not failures:
                continue
            monads = [self._sentinel_to_monad[sentinel] for sentinel in failures]
            failed_monad_ids = [monad.identifier for monad in monads]
            self.print(f'Monitor: {failed_monad_ids=}')
            # Exiting of state_tracker indicates all tasks are done
            if self._state_tracker in monads:
                self.print(f'Monitor exits')
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
    '''
