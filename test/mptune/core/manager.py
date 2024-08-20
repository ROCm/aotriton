#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import itertools
import time
from datetime import datetime
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
            self.print(f"{m.identifier=} {m.sentinel=} {m.pid=}")
        self._sentinel_to_monad = { monad.sentinel : monad for monad in self._all_monads }
        self._success_monads = set()

    def classify_monads(self, alive_monads):
        exited = []
        restarting = []
        self.print(f'monitor time.sleep returns')
        for monad in alive_monads:
            try:
                if monad.exitcode is not None:
                    exited.append(monad)
            except ValueError:
                restarting.append(monad)
        # exitcodes = [monad.exitcode for monad in alive_monads]
        # self.print(f'{exitcodes=}')
        return exited, restarting

    def detect_gpu_freeze(self):
        '''
        Detect GPU worker freeze
        '''
        msg = self._state_tracker.ask_for_alive_status()
        assert msg.action == MonadAction.OOB_QueryAlive
        alive_status = msg.payload
        now = datetime.now()
        for monad in self._workers:
            if monad.identifier not in alive_status:
                continue
            status = alive_status[monad.identifier]
            lapsed = (now - status).total_seconds()
            if lapsed > 5.0:
                monad.kill()
                self.print(f'Kill {monad.identifier} due to inactivity. {status=} {now=} {lapsed=}')

    def run_watchdog(self):
        # Not very useful since such cases usually require GPU reset
        # TODO: re-implement with amdsmi to confirm with GPU utilization
        # self.detect_gpu_freeze()

        alive_sentinels = []
        alive_monads = []
        for monad in self._all_monads:
            if monad in self._success_monads:
                continue
            # alive_sentinels.append(monad.sentinel)
            alive_monads.append(monad)
        self.print(f'monitor {alive_sentinels=}')
        try:
            exited, restarting = self.classify_monads(alive_monads)
            self.print(f'monitor wait_sentinels {exited=}')
        except Exception as e:
            self.print(f'monitor wait_sentinels timeout or Exception {e}')
            return
        if not exited:
            return
        monads = exited
        exited_monad_ids = [monad.identifier for monad in monads]
        self.print(f'watchdog: {exited_monad_ids=}')
        # Exiting of state_tracker indicates all tasks are done
        if self._state_tracker in monads:
            self.print(f'Monitor exits')
            return
        # Otherwise restart all faulty processes
        for monad in monads:
            if monad.exitcode != 0:
                print(f'{monad.identifier} exit with code {monad.exitcode}', flush=True)
                last_known_working = self._state_tracker.ask_for_last_message(monad)
                self._state_tracker.update_ui(last_known_working.clone().set_action(MonadAction.OOB_Died).update_payload(exitcode=monad.exitcode))
                if monad.identifier.startswith('worker_'):
                    nextone = self._src.next_kernel(last_known_working)
                    nexttwo = self._src.next_kernel(nextone)
                    monad.restart_with_last_progress(nexttwo)
            else:
                monad.join()
                # TODO: who should notify the state tracker? The
                #       exiting monad or the observing manager.
                #       (For now the exiting monad is used)
                # self._state_tracker.confirm_exit(monad)
                self._success_monads.add(monad)
