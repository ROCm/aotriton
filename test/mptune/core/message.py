#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from enum import Enum
from copy import deepcopy
from argparse import Namespace

MonadAction = Enum('MonadAction', ['Pass',
    'Skip',
    'DryRun',
    'Exit',
    'Exception',
    'OOB_Init',
    'OOB_Died',
    'OOB_RequestStatus',  # OOB means side channel communication only
    'OOB_AckRecv',
    'OOB_Restart',
    'OOB_QueryAlive',
])

'''
Message passed through multiprocessing.Queue

Nothing special but needs an ID
'''
class MonadMessage(ABC):

    def __init__(self, *, task_id, action : MonadAction, source=None, payload=None):
        self._task_id = task_id
        self._action = action
        self._source = source
        self._payload = payload

    def __format__(self, format_spec):
        return f'MonadMessage(task_id={self.task_id}, action={self.action}, source={self.source}, payload={self.payload})'

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
    def payload(self):
        return self._payload

    @property
    def skip_reason(self):
        if hasattr(self, '_skip_reason'):
            return self._skip_reason
        return None

    def set_source(self, source):
        self._source = source
        return self

    def update_payload(self, **kwargs):
        if self._payload is None:
            self._payload = Namespace()
        for k, v in kwargs.items():
            setattr(self._payload, k, v)
        return self

    def make_skip(self, monad, reason=None) -> 'MonadMessage':
        ret = self.forward(monad)
        ret._action = MonadAction.Skip
        ret._skip_reason = reason
        return ret

    def make_dryrun(self, monad) -> 'MonadMessage':
        ret = deepcopy(self)
        ret.set_source = monad.identifier
        ret._action = MonadAction.DryRun
        return ret

    def clone_ackrecv(self, monad) -> 'MonadMessage':
        ret = deepcopy(self).set_source(monad.identifier)
        ret._action = MonadAction.OOB_AckRecv
        return ret

    def forward(self, monad) -> 'MonadMessage':
        ret = deepcopy(self).set_source(monad.identifier)
        return ret

    def clone(self) -> 'MonadMessage':
        return deepcopy(self)

    def set_action(self, action):
        self._action = action
        return self
