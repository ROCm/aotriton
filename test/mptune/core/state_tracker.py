#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .message import MonadMessage, MonadAction
from .monad import Monad, MonadService
from multiprocessing import Queue

class StateTracker(Monad):
    def __init__(self, args, identifier='StateTracker'):
        super().__init__(args, identifier=identifier)
        self._q_up = Queue()
        self._q_down = Queue()

    def set_monads_to_track(self, monads):
        self._init_object = [monad.identifier for monad in monads]

    def get_side_channel_input(self):
        return self._q_up

    def service_factory(self):
        return StateTrackerService(self._args, self)

    def ask_for_last_message(self, monad):
        msg = MonadMessage(task_id=None, action=MonadAction.OOB_RequestStatus)
        msg.set_source(monad.identifier)
        self._q_up.put(msg)
        return self._q_up.get()

class StateTrackerService(MonadService):

    def init(self, init_object):
        self._progress_tracker = {}
        self._alive_monads = set(init_object)
        pass

    def process(self, request):
        if request.action == MonadAction.OOB_Init:
            print(f'Monad {request.source=} goes alive')
            # self._alive_monads.add(request.source)
            return
        if request.action == MonadAction.OOB_RequestStatus:
            # print(f'Requesting Status {request.source=}')
            yield self._progress_tracker[request.source]
            return
        if request.action == MonadAction.OOB_AckRecv:
            self._progress_tracker[request.source] = request
            # print(f'Update {request.source=} to {request}')
            return
        if request.action == MonadAction.Exit:
            print(f"before {self._alive_monads=}")
            self._alive_monads.remove(request.source)
            print(f"after {self._alive_monads=}")
            if not self._alive_monads:
                print(f"StateTracker yields Exit")
                yield MonadMessage(task_id=None, action=MonadAction.Exit, source='StateTracker')
            return

    def cleanup(self):
        pass
