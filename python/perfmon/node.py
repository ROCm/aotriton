# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The perfmon application's node entry point.

    python -m aotriton.perfmon.node broker
    python -m aotriton.perfmon.node gpu_worker --gpu_id 0
    python -m aotriton.perfmon.node cpu_worker
    python -m aotriton.perfmon.node reader --pool gfx950 [--subpool <variant>]

`.tune/remote/worker_service.sh --category perfmon` starts these. pgdef never
imports this module; this module hands it the handlers, the runner type and
the reader's enrichment.
"""

import argparse
import sys

from aotriton.pgdef.localq.node import run_node_role

from .exaid_perfmon import ExaidPerfmonRunner
from .handlers import HANDLERS, START


def node_subpool(argv: list[str]) -> str | None:
    """The `--subpool` this process was started with (a reader's), or None."""
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument('--subpool', default=None)
    known, _ = peek.parse_known_args(argv)
    return known.subpool


def make_enrich(subpool: str | None):
    """The reader's enrichment: the node's own subpool, so every report
    records which machine group measured it, whatever the task asked for.
    The reader is the only process that knows it, and enrich runs in it."""
    def enrich(conn, task) -> dict:
        return {'node_subpool': subpool}
    return enrich


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    run_node_role(app='perfmon',
                  handlers=HANDLERS,
                  start=START,
                  runners={'perfmon': ExaidPerfmonRunner},
                  enrich=make_enrich(node_subpool(argv)),
                  argv=argv)


if __name__ == '__main__':
    main()
