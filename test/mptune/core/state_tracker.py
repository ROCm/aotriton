#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .message import MonadMessage, MonadAction
from .monad import Monad, MonadService
from multiprocessing import Queue
from datetime import datetime

class StateTracker(Monad):
    def __init__(self, args, identifier='StateTracker'):
        super().__init__(args, identifier=identifier)
        self._q_up = Queue()
        self._q_resume = Queue()  # ask_for_last_message
        # Note, do not set _q_down, otherwise downstreaming messages may
        # saturate the queue
        self._q_ui = Queue()

    def set_monads_to_track(self, monads):
        self._init_object = [monad.identifier for monad in monads]

    def get_side_channel_input(self):
        return self._q_up

    def get_ui_update_queue(self):
        return self._q_ui

    def service_factory(self):
        return StateTrackerService(self._args, self)

    def ask_for_last_message(self, monad):
        msg = MonadMessage(task_id=None, action=MonadAction.OOB_RequestStatus)
        self.print(f'ask_for_last_message {monad.identifier=}')
        msg.set_source(monad.identifier)
        self._q_up.put(msg)
        return self._q_resume.get()

    def ask_for_alive_status(self):
        msg = MonadMessage(task_id=None,
                           action=MonadAction.OOB_QueryAlive)
        self._q_up.put(msg)
        return self._q_resume.get()

    def update_ui(self, info):
        if self._q_ui:
            self.print(f'StateTracker put ui {info}')
            self._q_ui.put(info)

class StateTrackerService(MonadService):

    def init(self, init_object):
        self._progress_tracker = {}
        self._alive_watchdog = {}
        self._task_confirmation = {}
        self._alive_monads = set(init_object)
        pass

    def process(self, request):
        if request.action == MonadAction.OOB_QueryAlive:
            alive = MonadMessage(task_id=None,
                                 action=MonadAction.OOB_QueryAlive,
                                 source=self.monad.identifier,
                                 payload=self._alive_watchdog)
            self.print(f'state_tracker reply {self._alive_watchdog=}')
            self.monad._q_resume.put(alive)
            return
        self.monad.update_ui(request)  # CAVEAT, do not change the source
        self.print(f'state_tracker send {request} to ui')
        if request.action == MonadAction.OOB_Init:
            self.print(f'Monad {request.source=} goes alive')
            # self._alive_monads.add(request.source)
            return
        if request.action == MonadAction.OOB_RequestStatus:
            print(f'Requesting Status {request.source=}')
            progress = self._progress_tracker[request.source]
            if progress.task_id in self._task_confirmation:
                self.print(f'Replying progress {self._task_confirmation[progress.task_id].payload=} to {request.source=}')
                self.monad._q_resume.put(self._task_confirmation[progress.task_id] \
                                             .clone() \
                                             .set_source(request.source))
            else:
                self.print(f'Replying progress {progress=} to {request.source=}')
                self.monad._q_resume.put(progress)
            return
        if request.action == MonadAction.OOB_AckRecv:
            self._progress_tracker[request.source] = request
            '''
            Cannot remember the reason to check the request.source
            But the effect is clear, the SIGSEGV task reported by the worker as
            AckRecv will be silently discarded and the progress will not be
            updated
            # Old code:
            # if request.task_id is not None and request.source in ['dbaccessor', 'opdb']:
            '''
            if request.task_id is not None:
                self._task_confirmation[request.task_id] = request
            self.print(f'Update _progress_tracker[{request.source=}] to {request}')
            return
        if request.action == MonadAction.Pass:
            self._alive_watchdog[request.source] = datetime.now()
            return
        if request.action == MonadAction.Exit:
            self.print(f"before {self._alive_monads=} {request.source=}")
            self._alive_monads.remove(request.source)
            self.print(f"after {self._alive_monads=}")
            if not self._alive_monads:
                self.print(f"StateTracker yields Exit")
                yield MonadMessage(task_id=None, action=MonadAction.Exit, source='StateTracker')
            return

    def cleanup(self):
        pass
