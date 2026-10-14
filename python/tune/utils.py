# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
from dataclasses import fields
import json
import math
import logging

logger = logging.getLogger(__name__)


def safeload(s):
    return json.loads(s) if s else None

def asdict_shallow(obj) -> dict:
    return {field.name: getattr(obj, field.name) for field in fields(obj)}

def __getattr__(name):
    """PEP 562 module __getattr__: lazily resolve `dacite_tuple`.

    `dacite` is an optional, tuning-only dependency (see modules/flash/tune's
    module docstrings and python/tune/tests/test_tune_infra.py). Every caller here
    imports it via `from aotriton.tune.utils import ... dacite_tuple ...`,
    which -- if `dacite_tuple` were a plain module-level value computed via
    `dacite.Config(...)` at import time -- would force an eager `import
    dacite` merely by importing this module, even for callers (e.g. the
    family-package loader in registry.py) that never call anything needing
    dacite. Resolving it lazily here means importing `aotriton.tune.utils`
    itself never requires dacite; only actually reading `dacite_tuple` does.
    """
    if name == 'dacite_tuple':
        import dacite
        value = dacite.Config(cast=[tuple])
        globals()['dacite_tuple'] = value  # cache: only pay the import cost once
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def sanitize_float(value):
    """
    Sanitize float values for PostgreSQL JSONB compatibility.
    - NaN -> None (becomes JSON null)
    - +Inf -> max float32
    - -Inf -> min float32
    """
    if isinstance(value, float):
        if math.isnan(value):
            return None
        elif math.isinf(value):
            if value > 0:
                return float(sys.float_info.max)
            else:
                return float(-sys.float_info.max)
    return value

def sanitize_value(obj):
    """Recursively sanitize an object to replace NaN/Inf values."""
    if isinstance(obj, dict):
        return {k: sanitize_value(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        result = [sanitize_value(item) for item in obj]
        return tuple(result) if isinstance(obj, tuple) else result
    elif isinstance(obj, float):
        return sanitize_float(obj)
    else:
        return obj
