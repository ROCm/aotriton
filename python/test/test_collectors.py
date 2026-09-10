# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Collector unification: the common `partition()` (specs/bundle.py) plus each
stack's specialised `partition_kernel/affine/flyc/operator` wrapper. Covers the
three behaviour changes that come out of widening every stack to share one
partition:

  (a) a second @ati.disable is an error on every stack kind, not just the
      Triton/kernel one (the affine collector used to silently drop it);
  (b) an @ati.tensor (or any other kind an affine stack has no use for) on an
      @ati.affine stack is now an explicit forbid() naming the kind, instead
      of silently landing in an "unexpected spec" catch-all;
  (c) flyc no longer restricts a stack to one @ati.cite -- that restriction
      was an artifact of specs/flyc.py's collector being written narrowly
      before the common vocabulary existed, not a rule flyc actually needs.

Also covers the cite-precedence contract this refactor must not disturb:
`cites` is a SEQUENCE, not a set -- the outermost (topmost in source)
@ati.cite wins a contested operand -- see specs/bundle.py's module docstring
and modules/flash/aot/bwd_kernel_fuse.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.describe import describe


def test_kernel_second_disable_raises():
    def k(Q, stride_qz, stride_qh, stride_qm, stride_qk, CAUSAL_TYPE: 'constexpr'):
        pass
    T = ati.type_var('T', dtype=['*fp16:16'])
    try:
        describe(k,
                 ati.tensor('Q', T, strides='stride_q?'),
                 ati.scalar('CAUSAL_TYPE', options=[0, 3]),
                 ati.disable(when=lambda f: True),
                 ati.disable(when=lambda f: False),
                 _validate=False)
    except AssertionError as e:
        assert 'disable' in str(e)
        return
    raise AssertionError('expected a second @ati.disable on a kernel stack to raise')


def test_affine_second_disable_raises():
    from aotriton.template_instantiation.specs.affine import partition_affine
    try:
        partition_affine([ati.disable(when=lambda f: True),
                          ati.disable(when=lambda f: False)])
    except AssertionError as e:
        assert 'disable' in str(e)
        return
    raise AssertionError('expected a second @ati.disable on an affine stack to raise')


def test_flyc_second_disable_raises():
    from aotriton.template_instantiation.specs.flyc import partition_flyc
    try:
        partition_flyc([ati.disable(when=lambda f: True),
                        ati.disable(when=lambda f: False)])
    except AssertionError as e:
        assert 'disable' in str(e)
        return
    raise AssertionError('expected a second @ati.disable on a flyc stack to raise')


def test_operator_second_disable_raises():
    from aotriton.template_instantiation.specs.operator import partition_operator
    try:
        partition_operator([ati.disable(when=lambda f: True),
                            ati.disable(when=lambda f: False)])
    except AssertionError as e:
        assert 'disable' in str(e)
        return
    raise AssertionError('expected a second @ati.disable on an operator stack to raise')


def test_affine_forbids_tensor():
    """An @ati.tensor has no meaning on an affine stack -- it has no Python
    signature to bind against. `partition_affine` must name the stack and the
    offending kind, not fall through to a generic "unexpected spec" error."""
    from aotriton.template_instantiation.specs.affine import partition_affine
    try:
        partition_affine([ati.tensor('Q', '*fp16:16')])
    except AssertionError as e:
        assert '@ati.affine' in str(e) and 'tensors' in str(e)
        return
    raise AssertionError('expected @ati.tensor on an affine stack to raise')


def test_operator_forbids_tensor():
    from aotriton.template_instantiation.specs.operator import partition_operator
    try:
        partition_operator([ati.tensor('Q', '*fp16:16')])
    except AssertionError as e:
        assert '@ati.operator' in str(e) and 'tensors' in str(e)
        return
    raise AssertionError('expected @ati.tensor on an operator stack to raise')


def test_flyc_accepts_multiple_cites():
    """The removed restriction: a flyc stack may carry several @ati.cite, same
    as a Triton kernel stack -- 'citation mode (b)', not an error."""
    from aotriton.template_instantiation.specs.flyc import partition_flyc
    c1 = ati.cite('op_x.metro.k1')
    c2 = ati.cite('op_x.metro.k2')
    b = partition_flyc([c1, c2])
    assert b.cites == [c1, c2], 'partition_flyc must keep multi-cite in source order'


def test_cite_precedence_outer_wins():
    """Several @ati.cite on one kernel: the OUTER (first in `cites`, topmost in
    source -- see specs/finalize.py's `start()` reversing `pending` back to
    source order) wins a contested operand. `resolve_cites` merges cited
    apparel with `cited_apparel.setdefault(...)`, so `cites[0]` must be tried
    first; this pins that `partition()` never reorders `cites` while doing
    so."""
    from aotriton.template_instantiation.specs.kernel import KernelDecl
    from aotriton.template_instantiation.ir.ops.cite import resolve_cites

    class P:
        def __init__(self, name):
            self.name = name

    class _Adapter:
        """`resolve_cites` looks up a cited KERNEL DESCRIPTION and reads its
        `.kernel_decl` (see ir/ops/cite.py's `_kernel_decl_of`) -- the real
        `lookup` (registry.py's InterfaceRegistry.get_kernel) returns a built
        KernelDescription, not a bare KernelDecl. This is the minimal stand-in."""
        def __init__(self, spec):
            self.kernel_decl = spec

    # Two cited kernels sharing the apparel name 'X' with DIFFERING practices
    # -- a contested operand.
    kernel_a = KernelDecl(name='kernel_a', params=[P('X')],
                          scalars=[ati.scalar('X', options=[1, 2, 3])])
    kernel_b = KernelDecl(name='kernel_b', params=[P('X')],
                          scalars=[ati.scalar('X', options=[99])])
    registry = {'kernel_a': _Adapter(kernel_a), 'kernel_b': _Adapter(kernel_b)}

    # citing declares neither X locally; cites kernel_a THEN kernel_b, which is
    # the outer/topmost-in-source order in this scenario.
    citing = KernelDecl(name='citing', params=[P('X')],
                        cites=[ati.cite('op.metro.kernel_a'),
                               ati.cite('op.metro.kernel_b')])
    resolve_cites(citing, family='fake', lookup=lambda family, name: registry.get(name))
    x = next(s for s in citing.scalars if s.arg_names == ('X',))
    assert x.options == [1, 2, 3], 'outer @ati.cite did not win the contested operand'

    # Reversed order: kernel_b now outer -> kernel_b's practice must win instead,
    # confirming the result tracks `cites` order and is not some fixed tiebreak.
    citing2 = KernelDecl(name='citing2', params=[P('X')],
                         cites=[ati.cite('op.metro.kernel_b'),
                                ati.cite('op.metro.kernel_a')])
    resolve_cites(citing2, family='fake', lookup=lambda family, name: registry.get(name))
    x2 = next(s for s in citing2.scalars if s.arg_names == ('X',))
    assert x2.options == [99], 'reordering cites did not change the winning practice'


def test_bwd_kernel_fuse_three_cite_stack_resolves():
    """The shipping backend that motivated widening `cites` to plural:
    modules/flash/aot/bwd_kernel_fuse.py (mirrored here by the fakefamily
    double) stacks THREE @ati.cite. It must still link end-to-end through the
    real two-pass Linker after the collector refactor."""
    from aotriton.codegen.linker import Linker
    fakefamily = Path(__file__).resolve().parent / 'fakefamily'
    kernels, _ops, _aff, _flyc = Linker(fakefamily).link_all_families()
    kdesc = next(k for k in kernels if k.NAME == 'bwd_kernel_fuse')
    fields = {cf.aname for cf in kdesc.func_cfields}
    # 'Out' is declared locally; 'Q' is a gap filled from the cited sub-kernels.
    assert 'Out' in fields
    assert 'Q' in fields
    assert kdesc.is_tunable is True
    # The plural side: bwd_kernel_fuse itself declares no local @ati.disable,
    # but its three cited sub-kernels between them declare five -- resolve_cites
    # must inherit all of them into `resolved_disables`, undiminished by the
    # citing kernel's own declared cardinality being 0-1. This is the gate the
    # declared/resolved split exists to keep honest.
    assert len(kdesc._built.disables) == 5


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} collector tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
