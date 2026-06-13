# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 4.2.2: AtiKernelDescription codegen surface matches the legacy attn_fwd
KernelDescription — struct cfields, perf cfields, launch arguments, and the
compiled-in feature tables."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'modules' / 'flash'))

import aot.attn_fwd as _attn_fwd_desc
attn_fwd = _attn_fwd_desc.attn_fwd
from aotriton.template_instantiation.compat import build_kernel_description
import v3python.rules.flash as F   # legacy reference for parity comparison


def _pair():
    ak = build_kernel_description(attn_fwd, family='flash')
    leg = next(k for k in F.kernels if k.NAME == 'attn_fwd')
    return ak, leg


def test_func_cfields_match():
    ak, leg = _pair()
    a = [(c.index, c.ctype, c.aname) for c in ak.func_cfields]
    l = [(c.index, c.ctype, c.aname) for c in leg.func_cfields]
    assert a == l
    assert len(a) == 46


def test_perf_cfields_match():
    ak, leg = _pair()
    a = [(c.ctype, c.aname, c.nbits) for c in ak.perf_cfields]
    l = [(c.ctype, c.aname, c.nbits) for c in leg.perf_cfields]
    assert a == l


def test_launch_arguments_match():
    ak, leg = _pair()
    a = [(x.aname, x.kind, x.expr) for x in ak.iter_launch_arguments()]
    l = [(x.aname, x.kind, x.expr) for x in leg.iter_launch_arguments()]
    assert a == l
    assert len(a) == 47
    # spot-check the three kinds
    by = {x.aname: x for x in ak.iter_launch_arguments()}
    assert by['Q'].kind == 'tensor_ptr'
    assert by['Q'].expr == 'params.Q->kparam_data_ptr()'
    assert by['stride_qz'].kind == 'tensor_stride'
    assert by['stride_qz'].expr == 'params.Q->kparam_stride(0)'
    assert by['Sm_scale'].kind == 'scalar'
    assert by['Sm_scale'].expr == 'CAST(&params.Sm_scale)'


def test_feature_tables_match():
    ak, leg = _pair()

    def feat(d):
        out = []
        for tp in d.list_functional_params():
            if not tp.emit_feature_table:
                continue
            tc = tp.repr_typed_choice
            out.append((tp.repr_name, tp.nchoices, tc.infotype,
                        tuple(c.infotext for c in tp.choices)))
        return out
    assert feat(ak) == feat(leg)


def test_dtype_variable_not_baked_even_if_member_overridden():
    # B is in the T_io axis and is override-baked; the T_io feature (named 'Q')
    # must still appear — baking is an argument property, not a type-variable one.
    ak, _ = _pair()
    names = [tp.repr_name for tp in ak.list_functional_params()
             if tp.emit_feature_table]
    assert 'Q' in names                 # the T_io dtype variable survives


def test_class_names():
    ak, _ = _pair()
    assert ak.param_class_name == 'AttnFwdParams'
    assert ak.context_class_name == 'AttnFwdContext'
    assert ak.metadata_class_name == 'AttnFwdMetadata'
    assert ak.enum_name == 'kShim_AttnFwd'


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} adapter-surface tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
