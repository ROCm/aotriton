# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aux-kernel xref Step 8: metro/operator-level @ati.cite via the ops registry
(agent-plans/ati_aux-kernel-xref_rev0.md §4.4).

  * @ati.cite("<op>.<metro>")          -> inherit the metro's MERGED sub-kernel
                                          practices (union of all sub-kernels);
  * @ati.cite("<op>.<metro>.<kernel>") -> inherit one sub-kernel's practices;
both resolved through ops[op].get_backend(metro)[.get_kernel(kernel)].
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.describe import describe, get_kernel_spec
from registry import InterfaceRegistry, _testonly_build_kernel_description
from aotriton.template_instantiation.builder import DescriptionError
from fakekernels import attn_fwd_stub, debug_stub

MAIN_DTYPES = ['*fp16:16', '*bf16:16', '*fp32:16']


class _FakeMetro:
    """Minimal metro exposing the cite-resolution surface (get_kernel /
    iter_subkernels) over a list of built sub-kernel kdescs."""
    def __init__(self, name, subs):
        self.NAME = name
        self._subs = subs
    def iter_subkernels(self):
        return list(self._subs)
    def get_kernel(self, name):
        for s in self._subs:
            if s.NAME == name:
                return s
        raise KeyError(f'no sub-kernel {name!r}')


class _FakeOp:
    def __init__(self, name, family, backends):
        self.NAME = name
        self.FAMILY = family
        self._backends = backends
    def get_backend(self, name):
        for b in self._backends:
            if getattr(b, 'NAME', None) == name:
                return b
        raise KeyError(f'no backend {name!r}')


def _build_attn_fwd_kdesc(registry):
    # attn_fwd already ATI-described on import; build + flat-register it.
    return _testonly_build_kernel_description(attn_fwd_stub(), family='flash',
                                    triton_kernel_name='attn_fwd',
                                    registry=registry)


def _register_fwd_op():
    reg = InterfaceRegistry()
    af = _build_attn_fwd_kdesc(reg)                # also flat-registers attn_fwd
    metro = _FakeMetro('triton', [af])
    op = _FakeOp('op_attn_fwd', 'flash', [metro])
    reg.register_op(op)
    return reg, op


def _cite_debug(target, registry):
    specs = [
        ati.cite(target),
        ati.tensor('R', 'T_io', strides='stride_r?', contiguous=-1,
                   wires_to='encoded_softmax'),
        ati.scalar(['BLOCK_M', 'BLOCK_N'], options=[64]),
        # attn_fwd has a local @ati.disable; a bare-lambda override would be fatal,
        # so affirm the intentional override (disable behavior is not under test).
        ati.disable(lambda f: False,
                    I_understand_this_overrides_cited_disable=True),
    ]
    debug = debug_stub()
    describe(debug, *specs, _validate=False)
    spec = get_kernel_spec(debug)
    from aotriton.template_instantiation.ir.ops.cite import resolve_cites
    resolve_cites(spec, family='flash', op_lookup=registry.get_op)
    return _testonly_build_kernel_description(debug, family='flash',
                                              register=False,
                                              registry=registry)


def test_subkernel_cite_via_ops():
    reg, _ = _register_fwd_op()
    kdesc = _cite_debug('op_attn_fwd.triton.attn_fwd', reg)
    R = next(a for a in kdesc._built.axes if 'R' in a.arg_names)
    assert R.radix == 3                          # T_io inherited via the op path
    assert kdesc.axis_of_arg('dropout_p') is not None


def test_whole_metro_cite_via_ops():
    reg, _ = _register_fwd_op()
    kdesc = _cite_debug('op_attn_fwd.triton', reg)  # whole metro, no sub-kernel
    assert kdesc.axis_of_arg('dropout_p') is not None
    R = next(a for a in kdesc._built.axes if 'R' in a.arg_names)
    assert R.radix == 3


def test_unknown_metro_raises():
    reg, _ = _register_fwd_op()
    try:
        _cite_debug('op_attn_fwd.no_such_metro', reg)
    except DescriptionError as e:
        assert 'no_such_metro' in str(e) or 'no backend' in str(e)
        return
    raise AssertionError('expected DescriptionError for unknown metro')


def test_whole_metro_requires_built_op():
    # No op registered -> a 2-segment (whole-metro) cite cannot fall back to the
    # flat kernel registry and must raise.
    reg = InterfaceRegistry()
    _build_attn_fwd_kdesc(reg)   # kernel only, no op
    try:
        _cite_debug('op_attn_fwd.triton', reg)
    except DescriptionError as e:
        assert 'whole-metro' in str(e) or 'not a built' in str(e)
        return
    raise AssertionError('expected DescriptionError for whole-metro without op')


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} cite-metro tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
