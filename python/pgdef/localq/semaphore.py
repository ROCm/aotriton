# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Semaphore messages: fan-in (a join) over the results of N parallel steps of
one task.

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
enough for most cases: wait until a signal has arrived for every key.

Usage. A producer emits the semaphore message and N parallel messages; each
parallel step's result signals one key, and the semaphore message is
released once every key has been signalled.

    # the step that fans out: wait(depends, keys, result_msg)
    join = semaphore.wait('tune/impl_result', [('attn_fwd', 0), ('attn_fwd', 1)],
                          child(msg, 'tune/postprocess'))   # no target_queue: filled in later

    # each parallel step's result message signals its key
    semaphore.signal(signal_msg, key=('attn_fwd', 0), value=report)

    # the semaphore message's handler, once released
    reports = semaphore.signals(msg)       # {('attn_fwd', 0): report, ...}

Getting the results. A fan-in usually wants the parallel steps' results,
not just their completion:

- Small results: carry them in the signal. `signal(msg, key, value)` stores
  `value` in the semaphore message's state in the broker, and the released
  handler reads all of them with `signals(msg)` as `{key: value}`. Values
  must be JSON values. Keep them small: every value is held in broker memory
  until release, and the released message carries them all. They are not
  durable; if the node dies, the task is retried or failed, never resumed
  from them.
- Large or durable results: write them to the database (e.g. a
  pgdef.report row) in the step that produces them, signal after the write,
  and read them back in the released handler through the application's pq.

Timing: a signal is recorded when its message is FORWARDED, that is, when a
handler returns it, not when that message's own handler has run. So the
released handler must not assume the signalling messages were handled. If
it needs what their handlers write, the signal must come from the step
after the write, not from the message that triggers the write.

Keys are any JSON values (tuples come back as tuples). With no keys there is
nothing to wait for: `wait()` adds no `depends`, so the message is not
blocked at all.
"""

import json
from typing import Any, Iterable

SEMAPHORE = 'semaphore'   # the message field holding the semaphore's state


def _encode(key) -> str:
    return json.dumps(key, sort_keys=True, separators=(',', ':'))


def _decode(text: str):
    def tuplify(x):
        return tuple(tuplify(i) for i in x) if isinstance(x, list) else x
    return tuplify(json.loads(text))


def wait(depends: str | Iterable[str], keys: Iterable, result_msg: dict) -> dict:
    """Make `result_msg` a semaphore message: the broker holds it until a
    signal has arrived for every key in `keys`, from messages of class
    `depends` (one class name, or several), then releases it.

    Returns `result_msg`, so a handler can emit it directly."""
    encoded = [_encode(k) for k in keys]
    if encoded:
        result_msg['depends'] = [depends] if isinstance(depends, str) else list(depends)
        result_msg[SEMAPHORE] = {'expected': encoded, 'received': {}}
    return result_msg


def signal(msg: dict, key, value: Any = None) -> dict:
    """Mark `msg` as the signal for `key` of a semaphore message, carrying
    `value`. Returns `msg`."""
    msg['signal_key'] = key
    msg['signal_value'] = value
    return msg


def resolve(blocked: dict, incoming: dict) -> bool:
    """Record `incoming` in `blocked` if it is one of its signals; True once a
    signal has arrived for every key. This is the default
    `MessageHandler.resolve_dependency`."""
    state = blocked.get(SEMAPHORE)
    if state is None:
        return False
    if incoming.get('class') not in blocked.get('depends', ()):
        return False
    if blocked.get('task_id') != incoming.get('task_id'):
        return False
    if 'signal_key' not in incoming:
        return False
    state['received'][_encode(incoming['signal_key'])] = incoming.get('signal_value')
    return set(state['expected']) <= set(state['received'])


def signals(msg: dict) -> dict:
    """The values the signals of a released semaphore message carried, by
    decoded key."""
    state = msg.get(SEMAPHORE) or {'received': {}}
    return {_decode(k): v for k, v in state['received'].items()}
