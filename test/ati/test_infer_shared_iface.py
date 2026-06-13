# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 5.2: SHARED_IFACE is inferred from operator -> metro -> kernel, not
declared. Uses lightweight stand-ins for the operator/metro/kernel shapes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aotriton.template_instantiation.operator import infer_shared_iface


class _Kernel:
    def __init__(self, name):
        self.NAME = name
        self.SHARED_IFACE = None


class _Cond:
    def __init__(self, if_kernel, else_kernel=None):
        self.if_kernel = if_kernel
        self.else_kernel = else_kernel


class _Metro:
    def __init__(self, steps):
        self._steps = steps

    def list_kernels(self):
        return self._steps


class _Op:
    def __init__(self, name, backends):
        self.NAME = name
        self._backends = backends

    def list_backends(self):
        return self._backends


def test_metro_subkernels_get_operator():
    k1 = _Kernel('attn_fwd')
    debug = _Kernel('debug')
    op = _Op('op_attn_fwd', [_Metro([k1, _Cond(debug)])])
    infer_shared_iface([op])
    assert k1.SHARED_IFACE is op                 # default-backend sub-kernel
    assert debug.SHARED_IFACE is op              # conditional branch too


def test_affine_backend_untouched():
    # A backend that already declares SHARED_IFACE (an affine/aiter kernel) is never
    # overridden — inference only fills in unset (ATI-adapter) kernels.
    k1 = _Kernel('attn_fwd')
    sentinel = object()
    affine = _Kernel('aiter')
    affine.SHARED_IFACE = sentinel
    op = _Op('op_attn_fwd', [_Metro([k1]), affine])
    infer_shared_iface([op])
    assert k1.SHARED_IFACE is op
    assert affine.SHARED_IFACE is sentinel       # declared -> never overridden


def test_does_not_override_declared():
    sentinel = object()
    k = _Kernel('x')
    k.SHARED_IFACE = sentinel                     # legacy: declared on the class
    op = _Op('op_x', [_Metro([k])])
    infer_shared_iface([op])
    assert k.SHARED_IFACE is sentinel             # never overridden


def test_else_branch_descended():
    a = _Kernel('a')
    b = _Kernel('b')
    op = _Op('op', [_Metro([_Cond(a, b)])])
    infer_shared_iface([op])
    assert a.SHARED_IFACE is op and b.SHARED_IFACE is op


def test_all_backends_borrow_the_struct():
    # The default backend is the struct OWNER, but every backend's kernels borrow
    # the operator's param struct (e.g. the fused bwd_kernel_fuse is an alternative
    # backend that still references OpAttnBwdParams). So SHARED_IFACE is set on the
    # kernels of ALL backends, not just the default.
    main = _Kernel('main')
    alt = _Kernel('alt')
    op = _Op('op', [_Metro([main]), _Metro([alt])])
    infer_shared_iface([op])
    assert main.SHARED_IFACE is op
    assert alt.SHARED_IFACE is op


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} infer_shared_iface tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
