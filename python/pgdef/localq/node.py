# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
run_node_role(): start one process of a node for an application.

pgdef is a library, not a framework: it never discovers or imports an
application. The application owns an entry point, `python -m aotriton.<app>.node
<role> ...`, which is a few lines that call this function with its own handlers:

    # python/tune/node.py
    if __name__ == '__main__':
        run_node_role(app='tune', handlers=HANDLERS, start=TuneStartHandler,
                      runners={'tuner': ExaidTunerWorker}, enrich=enrich,
                      argv=sys.argv[1:])

`.def/container/worker_service.sh --category <app>[,<level1>]` runs `python -m aotriton.<app>.node
<role>` for each process of the node:

    broker       [--queues gpu,cpu]
    gpu_worker   --gpu_id N
    cpu_worker   [--worker_id cpu-N]
    reader       --pool POOL [--subpool SUB] [--category tune,kernel] [--worker_id reader-N]

One application per node. A node may run zero GPU workers or zero CPU workers
(an application that needs only one type): tell the broker with `--queues`, and
a message for a queue with no workers fails its task instead of waiting.
"""

import argparse
import logging
import os
import signal
import socket
import sys

from psycopg.rows import dict_row

from aotriton.pgdef import paths
from aotriton.pgdef.pq.category import argparse_type, format_category
from aotriton.pgdef.pq.conn import ReconnectableConn, params_from_env
from aotriton.pgdef.utils import configure_logging_with_flush

from .broker import LocalBroker
from .generic_worker import GenericWorker
from .handlers import QUEUES, HandlerSet, NodeContext, StartHandler
from .pg_reader import DEFAULT_LEASE_SECONDS, PGReader

logger = logging.getLogger(__name__)

ROLES = ('broker', 'gpu_worker', 'cpu_worker', 'reader')


def default_broker_socket() -> str:
    return str(paths.broker_socket())


def build_parser(app: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f'python -m aotriton.{app}.node',
                                     description=f'One process of a {app} node')
    parser.add_argument('role', choices=ROLES)
    parser.add_argument('--broker_socket', type=str, default=default_broker_socket(),
                        help='Path to the broker Unix socket')
    parser.add_argument('--queues', type=str, default=','.join(QUEUES),
                        help="broker: queues that have workers on this node, e.g. 'cpu' for a zero-GPU node")
    parser.add_argument('--gpu_id', type=int, help='gpu_worker: GPU device ID')
    parser.add_argument('--worker_id', type=str, help='worker/reader identifier')
    parser.add_argument('--pool', type=str, help='reader: pool to claim from (the arch)')
    parser.add_argument('--subpool', type=str, default=None,
                        help='reader: machine group inside the pool; claims generic tasks plus its own')
    parser.add_argument('--category', type=argparse_type, default=(app, None),
                        help='reader: category to serve, e.g. tune,kernel (default: the application only)')
    parser.add_argument('--lease_s', type=float, default=DEFAULT_LEASE_SECONDS,
                        help='reader: task lease in seconds (renewed every third of it)')
    parser.add_argument('--hostname', type=str, default=None,
                        help='reader: host part of the worker id (default: this host)')
    return parser


def _install_worker_signals(worker):
    def handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully")
        worker.shutdown()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    # Interrupt blocking I/O on signals
    signal.set_wakeup_fd(worker.wakeup_write_fd)


def run_node_role(*, app: str, handlers, start: type[StartHandler], runners: dict | None = None,
                  enrich=None, argv: list[str] | None = None) -> None:
    """Run one role of a node for `app` until it is told to stop.

    Args:
        app: application name (also the category's level 0 for a reader)
        handlers: the application's MessageHandler subclasses, including `start`
        start: the start handler class; its CLASS is what the PG reader emits
        runners: runner type name -> class (`ExaidTunerWorker`, ...), for
            `ctx.runner(name, *key)`
        enrich: optional callable(conn, task) -> dict for the reader
        argv: command line after the program name
    """
    configure_logging_with_flush()
    args = build_parser(app).parse_args(argv)

    handler_set = HandlerSet(handlers)
    if start not in handler_set.by_class.values():
        raise ValueError(f'start handler {start.__name__} is not among the application handlers')

    {'broker': _run_broker, 'gpu_worker': _run_gpu_worker,
     'cpu_worker': _run_cpu_worker, 'reader': _run_reader}[args.role](
        app, args, handler_set, start, runners or {}, enrich)


def _run_broker(app, args, handler_set, start, runners, enrich):
    queues = tuple(q for q in args.queues.split(',') if q)
    unknown = [q for q in queues if q not in QUEUES]
    if unknown:
        sys.exit(f'--queues: unknown queue(s) {unknown}; expected a subset of {list(QUEUES)}')

    broker = LocalBroker(socket_path=args.broker_socket, handlers=handler_set.by_class,
                         priorities=handler_set.priorities(), queues=queues)

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully")
        broker.shutdown()

    def sighup_handler(signum, frame):
        logger.info("Received SIGHUP, requesting graceful shutdown with teardown")
        broker.graceful_shutdown()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGHUP, sighup_handler)

    logger.info(f"Starting LocalBroker for {app} with queues {queues} (PID={os.getpid()})")
    try:
        broker.start()
        broker.run()
    except KeyboardInterrupt:
        broker.shutdown()
        broker.run()  # Let it exit cleanly and cleanup
    except Exception as e:
        logger.error(f"Broker failed: {e} (PID={os.getpid()})", exc_info=True)
        sys.exit(1)


def _run_worker(worker, ctx, name):
    _install_worker_signals(worker)
    logger.info(f"Starting {name}")
    try:
        worker.run()
    except KeyboardInterrupt:
        worker.shutdown()
    except Exception as e:
        logger.error(f"{name} failed: {e} (PID={os.getpid()})", exc_info=True)
        worker.shutdown()
        sys.exit(1)
    finally:
        ctx.close()
        if ctx.db_conn is not None:
            ctx.db_conn.close()


def _run_gpu_worker(app, args, handler_set, start, runners, enrich):
    if args.gpu_id is None:
        sys.exit('gpu_worker needs --gpu_id')
    worker_id = args.worker_id or f'gpu-{args.gpu_id}'
    ctx = NodeContext(app=app, role='gpu_worker', worker_id=worker_id, gpu_id=args.gpu_id,
                      runner_types=runners, start=start)
    worker = GenericWorker(worker_id, 'gpu', handler_set, ctx, broker_socket=args.broker_socket)
    _run_worker(worker, ctx, f'GPU worker {worker_id}')


def _run_cpu_worker(app, args, handler_set, start, runners, enrich):
    worker_id = args.worker_id or 'cpu-0'
    # One persistent connection per process; ReconnectableConn transparently
    # reconnects if the server closes it.
    db_conn = ReconnectableConn(params_from_env(), row_factory=dict_row, autocommit=True)
    ctx = NodeContext(app=app, role='cpu_worker', worker_id=worker_id, db_conn=db_conn,
                      runner_types=runners, start=start)
    worker = GenericWorker(worker_id, 'cpu', handler_set, ctx, broker_socket=args.broker_socket)
    _run_worker(worker, ctx, f'CPU worker {worker_id}')


def _run_reader(app, args, handler_set, start, runners, enrich):
    if not args.pool:
        sys.exit('reader needs --pool')
    if args.category[0] != app:
        sys.exit(f"--category {format_category(args.category)}: this node runs application {app!r}")
    host = args.hostname or socket.gethostname()
    worker_id = f"{host}/{args.worker_id or 'reader-0'}"
    reader = PGReader(worker_id=worker_id, host=host, pool=args.pool, subpool=args.subpool,
                      category=args.category, broker_socket=args.broker_socket,
                      conn_params=params_from_env(), start=start, enrich=enrich, lease_s=args.lease_s)
    _install_worker_signals(reader)
    try:
        reader.run()
    except KeyboardInterrupt:
        reader.shutdown()
    except Exception as e:
        logger.error(f"PG Reader {worker_id} failed: {e} (PID={os.getpid()})", exc_info=True)
        reader.shutdown()
        sys.exit(1)
