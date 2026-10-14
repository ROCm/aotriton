# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Fan-in: a join over the results of N parallel steps of one task.

How a join works (the mechanism this module plugs into):

The join is a "semaphore message", an ordinary message that carries a
`depends` key listing the message classes it waits for.
- A handler returns it like any other message.
- The broker's `forward()` sees `depends` and, instead of queueing it, parks it
  under each class it depends on.
- Every later message of one of those classes is offered to it: the broker
  calls `resolve_dependency(blocked, incoming)` on the parked message's
  handler class. This runs in the broker, with no database. The incoming
  message is still delivered to its own queue.
- When that returns True, the broker removes `depends` and forwards the
  semaphore message, and its handler runs as usual.
- If the broker shuts down while it is still parked,
  `teardown_with_unmet_dependency()` decides what happens; by default the
  task is given back.

`resolve_dependency` is the extension point: a handler class may override it
to release on any condition. This module is the default implementation
(`MessageHandler.resolve_dependency` calls `resolve()`), which should be
enough for most cases: wait until a result has arrived for every expected
key.

Usage. A producer emits the join message and N workers; the join waits, in
the broker, until a result has arrived for every expected key.

    # the step that fans out
    join = child(msg, 'tune/postprocess', target...)          # no target_queue: filled in later
    fanin.expect(join, keys=[('attn_fwd', 0), ('attn_fwd', 1)], depends='tune/impl_result')

    # each parallel step's result message
    fanin.contribute(result_msg, key=('attn_fwd', 0), value=report)

    # the join's handler, once released
    reports = fanin.values(msg)       # {('attn_fwd', 0): report, ...}

Keys are any JSON values (tuples come back as tuples). With no keys there is
nothing to wait for: `expect()` adds no `depends`, so the join is not blocked
at all.
"""

import json
from typing import Any, Iterable

FANIN = 'fanin'


def _encode(key) -> str:
    return json.dumps(key, sort_keys=True, separators=(',', ':'))


def _decode(text: str):
    def tuplify(x):
        return tuple(tuplify(i) for i in x) if isinstance(x, list) else x
    return tuplify(json.loads(text))


def expect(msg: dict, keys: Iterable, *, depends: str) -> dict:
    """Turn `msg` into a join over `keys`, released by messages of class
    `depends`. Returns `msg`."""
    encoded = [_encode(k) for k in keys]
    if encoded:
        msg['depends'] = [depends]
        msg[FANIN] = {'expected': encoded, 'received': {}}
    return msg


def contribute(msg: dict, key, value: Any = None) -> dict:
    """Mark `msg` as the result for `key` of a join. Returns `msg`."""
    msg['fanin_key'] = key
    msg['fanin_value'] = value
    return msg


def resolve(blocked: dict, incoming: dict) -> bool:
    """Record `incoming` in `blocked` if it is one of its results; True once
    every expected key has arrived."""
    state = blocked.get(FANIN)
    if state is None:
        return False
    if incoming.get('class') not in blocked.get('depends', ()):
        return False
    if blocked.get('task_id') != incoming.get('task_id'):
        return False
    if 'fanin_key' not in incoming:
        return False
    state['received'][_encode(incoming['fanin_key'])] = incoming.get('fanin_value')
    return set(state['expected']) <= set(state['received'])


def values(msg: dict) -> dict:
    """The collected results of a released join, by decoded key."""
    state = msg.get(FANIN) or {'received': {}}
    return {_decode(k): v for k, v in state['received'].items()}
