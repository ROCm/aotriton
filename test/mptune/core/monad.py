#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .aav import ArgArchVerbose
from abc import ABC, abstractmethod
from enum import Enum
from multiprocessing import Process, Queue
import queue
from collections import namedtuple
from copy import deepcopy

MonadAction = Enum('MonadAction', ['Pass', 'Skip', 'DryRun', 'Exit', 'Exception'])

'''
Message passed through multiprocessing.Queue

Nothing special but needs an ID
'''
class MonadMessage(ABC):

    def __init__(self, task_id, action : MonadAction):
        self._task_id = task_id
        self._action = action
        self._source = None

    @property
    def task_id(self):
        return self._task_id

    @property
    def action(self):
        return self._action

    @property
    def source(self):
        return self._source

    @property
    def skip_reason(self):
        if hasattr(self, '_skip_reason'):
            return self._skip_reason
        return None

    def set_source(self, source):
        self._source = source

    def make_skip(self, reason=None) -> 'MonadMessage':
        ret = MonadMessage(self.task_id, MonadAction.Skip)
        ret._skip_reason = reason
        return ret

    def make_dryrun(self) -> 'MonadMessage':
        ret = deepcopy(self)
        ret._action = MonadAction.DryRun
        return ret

    def make_pass(self, **kwargs) -> 'MonadMessage':
        ret = deepcopy(self)
        ret._action = MonadAction.Pass
        for k, v in kwargs:
            setattr(ret, k, v)
        return ret

# class QueuePair(object):
# 
#     def __init__(self):
#         self._request_flow = None
#         self._feedback_flow = None
#         self._monads_up = set()
#         self._monads_down = set()
# 
#     @property
#     def request_flow(self):
#         if self._request_flow is None:
#             self._request_flow = Queue()
#         return self._request_flow
# 
#     @property
#     def feedback_flow(self):
#         if self._feedback_flow is None:
#             self._feedback_flow = Queue()
#         return self._feedback_flow
# 
#     def connect(self, up : Monad, down : Monad):
#         self._monads_up.add(up)
#         self._monads_down.add(down)


'''
A Monad will runs in a separate process.
For the main process this is a proxy object.

This class is supposed to be inherited to actual implement a service in cpptuning framework.
'''
class Monad(ArgArchVerbose):

    def __init__(self, args, identifier=None, side_channel=None):
        super().__init__(args)
        self._q_up= None
        self._q_down = None
        self._identifier = identifier  # None identifier implies unique in the system
        self._process = None
        self._side_channel = side_channel
        self._init_obj = None

    '''
    1-1 Protocol: assert to ensure only creating QP once
    '''
    def bind_1to1(self, downstream : 'Monad'):
        assert self._q_down is None
        assert downstream._q_up is None
        self._q_down = downstream._q_up = Queue()
        self._q_down.cancel_join_thread()
        # self._q_down.connect(self, downstream)

    '''
    1-N Protocol: assert to ensure no overwriting
    '''
    def bind_1toN(self, downstreams : 'Monad'):
        assert self._q_down is None
        q = Queue()
        q.cancel_join_thread()
        self._q_down = q
        for ds in downstreams:
            assert ds._q_up is None
            ds._q_up = qp
            # self._q_down.connect(self, ds)

    '''
    N-1 Protocol: reuse downstream QP
    '''
    def bind_Nto1(self, downstream : 'Monad'):
        if downstream._q_up is None:
            downstream._q_up = Queue()
            downstream._q_up.cancel_join_thread()
        assert self._q_down is None
        self._q_down = downstream._q_up
        # self._q_down.connect(self, downstream)

    def start(self, init_obj):
        self._process = Process(target=self.main, args=(self._q_up, self._q_down))
        self._init_obj = deepcopy(init_obj)
        self._q_up.put(init_obj)

    def restart(self):
        self._process = Process(target=self.main, args=(self._q_up, self._q_down))

    def join(self):
        self._process.join()

    def main(self, q_up, q_down):
        service = self.service_factory()
        if q_up:
            init_obj = q_up.get()
            service.init(init_obj)
        while True:
            try:
                if q_up:
                    req = q_up.get()
                    if action == MonadAction.Exit:
                        '''
                        Exit handling: instead of broadcasting in a 1-N queue,
                        putting the object back to the shared queue.
                        '''
                        q_up.put(req)
                else:
                    req = None
                '''
                Note for MonadService.process: propogate Exit to downstream.
                     Unless not needed.
                '''
                for res in service.process(req):
                    res.set_source(self._identifier)
                    '''
                    Report to downstream and/or side channel
                      - downstream for further processing
                      - side channel for status tracking
                    '''
                    if q_down:
                        q_down.put(res)
                    if self._side_channel:
                        self._side_channel.put(res)
                    '''
                    Propogate to downstream
                    '''
                    if res.action == MonadAction.Exit:
                        break
            except ValueError:  # mp.Queue closed
                break
        service.cleanup()

    @abstractmethod
    def service_factory(self):
        pass

    def report_exception(self, e: Exception):
        msg = MonadMessage(None, MonadAction.Exception)
        msg.exception = e
        self._side_channel.put(msg)

    def report_status(self, msg):
        self._side_channel.put(msg)

class MonadService(ArgArchVerbose):

    def __init__(self, args, monad):
        super().__init__(args)
        self._monad = monad  # In case need it

    @abstractmethod
    def init(self, init_object):
        pass

    @abstractmethod
    def process(self, request):
        pass

    @abstractmethod
    def cleanup(self):
        pass

    def report_exception(self, e: Exception):
        self._monad.report_exception(e)

    def report_status(self, msg):
        self._monad.report_status(msg)
