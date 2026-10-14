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
  handler class. This runs in the broker. The incoming
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
    reports = semaphore.get_aggregate_results(msg)   # {('attn_fwd', 0): report, ...}

Getting the results. A fan-in usually wants the parallel steps' results,
not just their completion, and the semaphore message collects them:

- Each parallel step returns a message marked with `signal(msg, key, value)`.
  `value` is that step's result.
- While the semaphore message is held in the broker, `resolve()` (the default
  `resolve_dependency`) accumulates every matching message into it: a
  message whose class is in `depends`, whose task is the same, and which
  carries a key. Its value is stored in the semaphore message's own state,
  `msg['semaphore']['received'][key] = value`, next to the expected keys
  (`msg['semaphore']['expected']`). A repeated key overwrites the earlier
  value.
- When released, the semaphore message carries all of them to its handler,
  which reads them from its own input message: `get_aggregate_results(msg)`
  returns `{key: value}`, with keys decoded (tuples come back as tuples).

Only the value travels, not the rest of the signalling message, so put
everything the join needs into it. Values must be JSON values. They are held
in the broker until release, so keep them small.

Keys are JSON scalars, or lists/tuples of them (they come back as tuples);
anything else, such as a dict, is rejected with TypeError. A signal only
counts for the semaphore message of the same task AND the same attempt, so
a retried task is never released by an earlier attempt's leftovers. With no keys there is
nothing to wait for: `wait()` adds no `depends`, so the message is not
blocked at all.
"""

import json
from typing import Any, Iterable

SEMAPHORE = 'semaphore'   # the message field holding the semaphore's state


def _check_key(key) -> None:
    """A key is a JSON scalar, or a list/tuple of keys. Dicts are rejected:
    get_aggregate_results() returns the keys as dict keys, so they must be
    hashable once decoded (lists come back as tuples)."""
    if key is None or isinstance(key, (str, int, float, bool)):
        return
    if isinstance(key, (list, tuple)):
        for k in key:
            _check_key(k)
        return
    raise TypeError(f'semaphore key must be a JSON scalar or a list/tuple of them, not {key!r}')


def _encode(key) -> str:
    _check_key(key)
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
    _check_key(key)
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
    # A retried task runs a new attempt; signals left over from an older
    # attempt's graph must not satisfy the new attempt's semaphore.
    if blocked.get('attempt') != incoming.get('attempt'):
        return False
    if 'signal_key' not in incoming:
        return False
    state['received'][_encode(incoming['signal_key'])] = incoming.get('signal_value')
    return set(state['expected']) <= set(state['received'])


def get_aggregate_results(msg: dict) -> dict:
    """Read the aggregate results of a released semaphore message: the value
    each signal carried, as {key: value} with keys decoded. Reading only;
    `msg` is not changed."""
    state = msg.get(SEMAPHORE) or {'received': {}}
    return {_decode(k): v for k, v in state['received'].items()}
