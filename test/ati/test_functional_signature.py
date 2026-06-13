# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 4.2.3: AtiFunctional signature/packing surface matches the legacy
attn_fwd functionals (the load-bearing aks2/DB-keying signatures), and the
multi-choice signature_name requirement is enforced."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'modules' / 'flash'))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.describe import describe, get_kernel_spec
from aotriton.template_instantiation.builder import build_kernel, AtiDescriptionError
from aotriton.template_instantiation.compat import build_kernel_description
from aotriton.gpu_targets import cluster_gpus
import aot.attn_fwd as _attn_fwd_desc
attn_fwd = _attn_fwd_desc.attn_fwd
try:
    import v3python.rules.flash as F   # legacy reference for parity comparison
except ModuleNotFoundError:
    F = None   # legacy reference unavailable (v3python removed) -> parity tests skip

_SIG_FIELDS = ['unified_signature', 'signature_in_func_name',
               'compact_signature_noarch', 'filepack_inzip_name',
               'tunecc_signature']


def _pair_functionals():
    ak = build_kernel_description(attn_fwd, family='flash')
    leg = next(k for k in F.kernels if k.NAME == 'attn_fwd')
    ta = cluster_gpus(['gfx942_mod0'])
    lf = {f.godel_number: f for f in leg.gen_functionals(ta)}
    mf = {f.godel_number: f for f in ak.gen_functionals(ta)}
    return lf, mf


def test_core_signatures_match_all_functionals():
    lf, mf = _pair_functionals()
    bad = []
    for g in lf:
        for fld in _SIG_FIELDS:
            if getattr(lf[g], fld) != getattr(mf[g], fld):
                bad.append((fld, g, getattr(lf[g], fld), getattr(mf[g], fld)))
    assert not bad, f'{len(bad)} signature mismatches, e.g. {bad[:3]}'


def test_filepack_paths_match():
    lf, mf = _pair_functionals()
    for g in lf:
        assert (lf[g].filepack_ondisk_path.as_posix()
                == mf[g].filepack_ondisk_path.as_posix())
        assert (lf[g].full_flatzip_path.as_posix()
                == mf[g].full_flatzip_path.as_posix())


def test_signature_name_is_explicit_repr_not_dtype_var():
    _, mf = _pair_functionals()
    f = mf[246]
    # The T_io variable is recorded under 'Q' (its signature_name), never 'T_io'.
    assert "Q='*bf16:16'" in f.unified_signature
    assert 'T_io' not in f.unified_signature


def test_pp_arg_doc_matches_legacy():
    # The prepare_arguments per-arg (is_constexpr, comment) must match legacy for
    # every launch argument of every functional. Covers the override comment cases:
    # tensor-degraded-to-0, literal 0, and VarRef (Hdim -> deferred choice list).
    lf, mf = _pair_functionals()
    leg = next(k for k in F.kernels if k.NAME == 'attn_fwd')
    largs = [la.aname for la in leg.iter_launch_arguments()]
    bad = []
    for g in lf:
        for a in largs:
            if lf[g].pp_arg_doc(a) != mf[g].pp_arg_doc(a):
                bad.append((g, a, lf[g].pp_arg_doc(a), mf[g].pp_arg_doc(a)))
    assert not bad, f'{len(bad)} pp_arg_doc mismatches, e.g. {bad[:3]}'


def test_multichoice_shared_var_requires_signature_name():
    # A multi-choice variable spanning >1 arg without signature_name -> error.
    def k(a, b):
        pass
    T = ati.tensor_dtype('T', dtype=['*fp16:16', '*bf16:16'])   # 2 choices, no sig name
    describe(k, ati.tensor('a', T, rank=2), ati.tensor('b', T, rank=2),
             _validate=False)
    try:
        build_kernel(get_kernel_spec(k))
    except AtiDescriptionError as e:
        assert 'signature_name' in str(e)
        return
    raise AssertionError('expected signature_name requirement error')


def test_single_choice_shared_var_exempt():
    # A single-choice variable is trivial -> no signature_name needed.
    def k(a, b):
        pass
    T = ati.tensor_dtype('T', dtype=['*fp16:16'])               # 1 choice
    describe(k, ati.tensor('a', T, rank=2), ati.tensor('b', T, rank=2),
             _validate=False)
    bk = build_kernel(get_kernel_spec(k))           # must not raise
    assert bk is not None


def main():
    if F is None:
        print('SKIP: v3python legacy reference unavailable; parity test skipped.')
        return 0
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} functional-signature tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
