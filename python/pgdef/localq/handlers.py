# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The handler graph: implicit, but readable.

A task runs on one node as a small graph of message handlers. A handler's
return value IS the next message; there is no DAG language. What makes the
graph readable is that each handler declares, next to its `handle()`, where it
runs and what it may produce:

    class ProbeHandler(MessageHandler):
        CLASS = 'tune/probe'
        QUEUE = 'gpu'
        EMITS = ('tune/impl', 'tune/postprocess')
        def handle(self, msg, ctx): ...

`GenericWorker` checks every returned message against `EMITS` (raising on a
violation) and fills `target_queue` from the target handler's `QUEUE`, so a
handler never writes a queue name and the declared edges cannot drift from the
code. `python -m aotriton.pgdef.localq.graph <app>` prints the graph.

Engine classes (namespace `pgdef/`) are handled here: `task_failed` and
`graceful_cancel`. `dag_ack` is not a handler: the broker consumes it to
release the PG reader that is holding the task.

Message shape (JSON): `class`, `target_queue` (filled in by the worker),
`task_id`, `pool`, plus whatever the handlers exchange. `task_header()` and
`child()` carry the fields every message of one task shares.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from aotriton.pgdef.pq import tasks as pgtasks

from . import semaphore as _semaphore

logger = logging.getLogger(__name__)

QUEUES = ('gpu', 'cpu')

DAG_ACK = 'pgdef/dag_ack'
TASK_FAILED = 'pgdef/task_failed'
GRACEFUL_CANCEL = 'pgdef/graceful_cancel'

#: Message classes the engine consumes itself: valid in EMITS, no handler, no queue.
BROKER_CLASSES = (DAG_ACK,)

# Fields every message of one task carries (the task's columns, the immutable
# payload, the attempt number and the reader holding the claim), so a handler
# never needs a database read to know what it is working on. `lease_owner` is
# what complete/fail/release are guarded with (see `owner_of`).
HEADER_KEYS = ('task_id', 'pool', 'subpool', 'category', 'attempt', 'lease_owner', 'payload')


class Permanent(Exception):
    """Raised by a handler for a failure that retrying cannot fix (bad input).

    Any other exception fails the attempt only, and the task is retried while
    attempts remain (see pgdef.pq.logistics).
    """


class EmitsViolation(RuntimeError):
    """A handler returned a message its EMITS does not declare."""


def task_header(msg: dict) -> dict:
    """The fields shared by every message of the task `msg` belongs to."""
    return {k: msg[k] for k in HEADER_KEYS if k in msg}


def child(msg: dict, cls: str, **fields) -> dict:
    """A new message of class `cls` for the same task as `msg`."""
    return {'class': cls, **task_header(msg), **fields}


def task_ref(msg: dict) -> pgtasks.TaskRef:
    return (msg['task_id'], msg['pool'])


def owner_of(msg: dict) -> str:
    """The reader whose claim `msg` belongs to (header `lease_owner`, set by
    the PG reader and copied by child()). Pass it as `owner=` to
    pq.tasks.complete/fail/release, which require it, so a graph that outlived
    its claim cannot settle a later claim of the same task."""
    return msg['lease_owner']


def ack(msg: dict, *, negative: bool = False, error: str | None = None) -> dict:
    """The message that releases the PG reader holding the task."""
    out = {'class': DAG_ACK, 'task_id': msg['task_id']}
    if negative:
        out['negative'] = True
    if error is not None:
        out['error'] = error
    return out


@dataclass
class NodeContext:
    """What a handler may use: who is running it, and the node's resources.

    GPU workers have `gpu_id` and runners; CPU workers have a `db_conn`. A
    resource a role does not have is None, and a handler that needs it must be
    registered on the other queue.
    """
    app: str
    role: str
    worker_id: str
    gpu_id: int | None = None
    db_conn: Any = None
    runner_types: dict = field(default_factory=dict)    # name -> class, from run_node_role(runners=)
    start: type | None = None                           # the application's start handler class
    _live: dict = field(default_factory=dict)

    def runner(self, name: str, *key):
        """The (cached) runner of type `name` for `key`, created as
        `runner_types[name](*key, gpu_id)`. One live object per (name, key)."""
        cache_key = (name, key)
        if cache_key not in self._live:
            try:
                cls = self.runner_types[name]
            except KeyError:
                raise KeyError(f"no runner type {name!r}; this node knows {sorted(self.runner_types)}") from None
            self._live[cache_key] = cls(*key, self.gpu_id)
        return self._live[cache_key]

    def close(self) -> None:
        """Ask every live runner to exit (worker shutdown)."""
        for runner in self._live.values():
            try:
                runner.exit()
            except Exception:
                logger.warning('runner did not exit cleanly', exc_info=True)
        self._live.clear()


class MessageHandler:
    """One step of an application's graph.

    Subclass attributes (the declaration):
        CLASS     message class this handler processes ('<app>/<step>')
        QUEUE     'gpu' or 'cpu': which workers run it
        EMITS     message classes it may return
        PRIORITY  broker queue priority of ITS messages (higher = sooner)

    Instances are created once per worker process, with no arguments; resources
    come from the `ctx` that `handle()` receives.
    """

    CLASS: str = ''
    QUEUE: str = ''
    EMITS: tuple[str, ...] = ()
    PRIORITY: int = 0

    def handle(self, msg: dict, ctx: NodeContext) -> dict | list[dict] | None:
        """Process `msg`; return the next message(s) or None."""
        raise NotImplementedError

    # -- broker side: joins -------------------------------------------------
    # Called IN THE BROKER, on the class (no instance, no database, no runner).

    @classmethod
    def resolve_dependency(cls, blocked_msg: dict, incoming_msg: dict) -> bool:
        """`incoming_msg` arrived; has `blocked_msg`'s dependency been met?

        The default is the shared helper (`pgdef.localq.semaphore`): a semaphore
        message built with `semaphore.wait()` is released once a signal has
        arrived for every key.
        """
        return _semaphore.resolve(blocked_msg, incoming_msg)

    @classmethod
    def teardown_with_unmet_dependency(cls, msg: dict) -> dict | None:
        """The broker is shutting down gracefully and `msg` is still blocked:
        return a message to enqueue instead. The default gives the task back
        (graceful_cancel). Override to clean up too, then call super()."""
        return child(msg, GRACEFUL_CANCEL, target_queue='cpu')


def check_declaration(handler_cls: type) -> None:
    """Fail early (at node start) on a handler that is not fully declared."""
    name = handler_cls.__name__
    if not handler_cls.CLASS:
        raise ValueError(f'{name}.CLASS is not set')
    if handler_cls.QUEUE not in QUEUES:
        raise ValueError(f'{name}.QUEUE must be one of {QUEUES}, got {handler_cls.QUEUE!r}')
    if not isinstance(handler_cls.EMITS, tuple):
        raise ValueError(f'{name}.EMITS must be a tuple of message classes')


class HandlerSet:
    """The classes of one application, plus the engine's: lookups for the
    broker (priorities, joins) and the workers (queues, EMITS checking)."""

    def __init__(self, handler_classes):
        classes = list(handler_classes) + [TaskFailedHandler, GracefulCancelHandler]
        self.by_class: dict[str, type] = {}
        for cls in classes:
            check_declaration(cls)
            if cls.CLASS in self.by_class and self.by_class[cls.CLASS] is not cls:
                raise ValueError(f'two handlers claim message class {cls.CLASS!r}')
            self.by_class[cls.CLASS] = cls
        self.validate_emits()

    def validate_emits(self) -> None:
        """Every declared edge must lead somewhere: to a handler or to the broker."""
        for cls in self.by_class.values():
            for target in cls.EMITS:
                if target not in self.by_class and target not in BROKER_CLASSES:
                    raise ValueError(f'{cls.__name__}.EMITS names {target!r}, which no handler '
                                     f'handles (known: {sorted(self.by_class)})')

    def queue_of(self, msg_class: str) -> str | None:
        cls = self.by_class.get(msg_class)
        return cls.QUEUE if cls else None

    def priorities(self) -> dict[str, int]:
        return {c: h.PRIORITY for c, h in self.by_class.items()}

    def for_queue(self, queue: str) -> list[type]:
        return [h for h in self.by_class.values() if h.QUEUE == queue]


# ---------------------------------------------------------------------------
# The engine's own handlers
# ---------------------------------------------------------------------------

class TaskFailedHandler(MessageHandler):
    """The one failure path.

    Any exception raised by any handler becomes a `task_failed` message. This
    handler records the failure (`pq.tasks.fail`: the task goes back to
    pending while attempts remain, to failed otherwise), lets the application
    clean up, and releases the PG reader with a negative ack, so a reader slot
    is never lost. (A message the broker cannot deliver does not come here: the
    broker sends the reader a negative ack with the reason itself, and the
    reader records the failure.)

    Input: task_failed message (the error, whether it is permanent, and the
    failing message as `origin`).
    Output: a negative dag_ack, which releases the PG reader.
    """
    CLASS = TASK_FAILED
    QUEUE = 'cpu'
    EMITS = (DAG_ACK,)
    PRIORITY = 100

    def handle(self, msg, ctx):
        error = msg.get('error', 'unknown error')
        permanent = bool(msg.get('permanent', False))
        status = pgtasks.fail(ctx.db_conn, task_ref(msg), error, permanent=permanent,
                              owner=owner_of(msg))
        logger.error(f"task_id={msg['task_id']} failed ({'permanent' if permanent else 'attempt'}): "
                     f"{error} -> {status}")
        origin = msg.get('origin') or {}
        cleanup = getattr(ctx.start, 'cleanup', None)
        if cleanup is not None:
            try:
                cleanup(origin, ctx)
            except Exception:
                logger.warning('cleanup after failure raised', exc_info=True)
        return ack(msg, negative=True)


class GracefulCancelHandler(MessageHandler):
    """Give a task back during graceful shutdown (its attempt is not consumed),
    and let the application clean up.

    Input: graceful_cancel message.
    Output: none.
    """
    CLASS = GRACEFUL_CANCEL
    QUEUE = 'cpu'
    EMITS = ()
    PRIORITY = 100

    def handle(self, msg, ctx):
        pgtasks.release(ctx.db_conn, task_ref(msg), owner=owner_of(msg))
        logger.info(f"task_id={msg['task_id']} released back to pending")
        cleanup = getattr(ctx.start, 'cleanup', None)
        if cleanup is not None:
            try:
                cleanup(msg, ctx)
            except Exception:
                logger.warning('cleanup after release raised', exc_info=True)
        return None


class StartHandler(MessageHandler):
    """Base class for an application's start handler: the one that handles the
    message the PG reader emits for a freshly claimed task.

    The start message carries the task's columns, `payload`, `appcookie` and
    whatever the application's `enrich()` returned (under `enrich`). Override
    `cleanup()` to remove per-task scratch state when the task fails or is
    given back.

    Input: the PG reader's start message.
    Output: the application's first step (declared in the subclass's EMITS).
    """

    @classmethod
    def cleanup(cls, msg: dict, ctx: NodeContext) -> None:
        pass
