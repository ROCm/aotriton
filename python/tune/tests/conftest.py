# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PostgreSQL fixtures for tune's `pg`-marked tests: the same three fixtures as
python/test/conftest.py, from the same helpers (aotriton.pgdef.pq.testing)."""

import pytest


@pytest.fixture(scope='session')
def pg_params():
    from aotriton.pgdef.pq import testing
    return testing.params_or_skip(pytest)


@pytest.fixture
def pg_conn(pg_params):
    from aotriton.pgdef.pq import testing
    conn = testing.fresh_schemas(pg_params)
    yield conn
    conn.close()
