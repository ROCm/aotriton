# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 4.2.1: the full ATI attn_fwd description matches the legacy
KernelDescription on per-argument resolved signatures across all functionals."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'modules' / 'flash'))

import aot.attn_fwd as _attn_fwd_desc
attn_fwd = _attn_fwd_desc.attn_fwd
from aotriton.template_instantiation.compat import build_kernel_description
from aotriton.gpu_targets import cluster_gpus
try:
    import v3python.rules.flash as F   # legacy reference for parity comparison
except ModuleNotFoundError:
    F = None   # legacy reference unavailable (v3python removed) -> parity tests skip

# perf params live in the schema, not the functional choice space
_PERF = {'PERSISTENT_TYPE', 'GRID_CU_MULTIP', 'BLOCK_M', 'BLOCK_N',
         'PRE_LOAD_V', 'NUM_XCDS'}


def _setup():
    ak = build_kernel_description(attn_fwd, family='flash')
    legacy = next(k for k in F.kernels if k.NAME == 'attn_fwd')
    ta = cluster_gpus(['gfx942_mod0'])
    return ak, legacy, ta


def test_describe_completeness_passes():
    # The stacked-@ form finalizes with full validation on import: all 74 params
    # must be claimed or @ati.kernel would have raised.
    assert attn_fwd.__ati__ is not None


def test_godel_set_matches_legacy():
    ak, legacy, ta = _setup()
    mine = sorted(f.godel_number for f in ak.gen_functionals(ta))
    leg = sorted(f.godel_number for f in legacy.gen_functionals(ta))
    assert mine == leg == list(range(576))


def test_resolved_signatures_match_legacy():
    ak, legacy, ta = _setup()
    leg = {f.godel_number: f for f in legacy.gen_functionals(ta)}
    mine = {f.godel_number: f for f in ak.gen_functionals(ta)}
    args = [a for a in legacy.ARGUMENTS if a not in _PERF]
    mismatches = []
    for g in leg:
        ld = leg[g].build_complete_tc_dict()
        md = mine[g].resolved
        for a in args:
            lv = str(ld[a].triton_compile_signature)
            mv = str(md[a].triton_compile_signature)
            if lv != mv:
                mismatches.append((a, lv, mv, g))
    assert not mismatches, f'{len(mismatches)} mismatches, e.g. {mismatches[:5]}'


def main():
    if F is None:
        print('SKIP: v3python legacy reference unavailable; parity test skipped.')
        return 0
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} attn_fwd parity tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
