# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A forced impl_index that selects no kernel must fail the task.

Before, the tuning shim's hipErrorSharedObjectSymbolNotFound left NaN outputs,
compare() turned them into a null adiff, and compute_best_results rejected the
candidate as a broken kernel. On gfx1100 that silently discarded every
attn_fwd candidate past the padded twin's 64th.

The check itself runs in level_kernel's direct_call(), which needs a tuning
build of pyaotriton; it is exercised on GPU, not here.
"""

import pytest

from aotriton.tune.kftdesc import ImplNotAvailable


def test_message_names_the_selection():
    with pytest.raises(ImplNotAvailable, match=r'^attn_fwd=176 selected no kernel'):
        raise ImplNotAvailable.for_index('attn_fwd', 176)


def test_postprocess_fails_the_task(monkeypatch):
    pytest.importorskip('psycopg')  # requirements-tuning.txt
    from aotriton.tune.localq import handlers
    calls = []

    class FakeTaskQueue:
        def __init__(self, conn):
            pass

        def mark_completed(self, task_id, arch):
            calls.append(('completed', task_id))

        def mark_failed(self, task_id, *, arch, error_message):
            calls.append(('failed', task_id, error_message))

    monkeypatch.setattr(handlers, 'TaskQueue', FakeTaskQueue)

    def postprocess(results):
        handlers.PostprocessHandler(db_conn=None).handle({
            'task_id': 7, 'task_config': {'arch': 'gfx1100', 'tmpdir': '/nonexistent'},
            'received_impls': {'attn_fwd': {str(i): {'result': r} for i, r in enumerate(results)}}})
        return calls.pop()

    assert postprocess(['OK', 'NotOK', 'crash']) == ('completed', 7)
    failed = postprocess(['OK', 'ImplNotAvailable'])
    assert failed[:2] == ('failed', 7) and "attn_fwd[1]" in failed[2]
