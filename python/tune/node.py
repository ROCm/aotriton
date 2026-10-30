# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The tune application's node entry point.

    python -m aotriton.tune.node broker
    python -m aotriton.tune.node gpu_worker --gpu_id 0
    python -m aotriton.tune.node cpu_worker
    python -m aotriton.tune.node reader --pool gfx950 --subpool MI355 --category tune,kernel

`.def/container/worker_service.sh --category tune,kernel` starts these. pgdef is a library:
it never imports this module; this module hands it the handlers, the runner
type and the reader's enrichment.
"""

import sys

from aotriton.pgdef.localq.node import run_node_role

from .exaid_tuner import ExaidTunerWorker
from .handlers import HANDLERS, START
from .pq.extra_uts import get_extra_uts


def enrich(conn, task) -> dict:
    """Per-task data that is not in the payload: the extra unit tests that were
    registered for the task when it was requeued after failing pytest checks."""
    extra_im_texts = get_extra_uts(conn, task.id)
    return {'extra_im_texts': extra_im_texts} if extra_im_texts else {}


if __name__ == '__main__':
    run_node_role(app='tune', handlers=HANDLERS, start=START,
                  runners={'tuner': ExaidTunerWorker}, enrich=enrich,
                  argv=sys.argv[1:])
