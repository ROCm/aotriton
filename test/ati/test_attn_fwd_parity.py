# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 4.2.1: the full ATI attn_fwd description matches the legacy
KernelDescription on per-argument resolved signatures across all functionals."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tritonsrc'))
sys.path.insert(0, str(REPO / 'modules'))

from fwd_kernel import attn_fwd
from flash.attn_fwd_ati import describe_attn_fwd
from v3python.template_instantiation.compat import build_kernel_description
from v3python.gpu_targets import cluster_gpus
import v3python.rules.flash as F

# perf params live in the schema, not the functional choice space
_PERF = {'PERSISTENT_TYPE', 'GRID_CU_MULTIP', 'BLOCK_M', 'BLOCK_N',
         'PRE_LOAD_V', 'NUM_XCDS'}


def _setup():
    describe_attn_fwd(attn_fwd)
    ak = build_kernel_description(attn_fwd, family='flash')
    legacy = next(k for k in F.kernels if k.NAME == 'attn_fwd')
    ta = cluster_gpus(['gfx942_mod0'])
    return ak, legacy, ta


def test_describe_completeness_passes():
    # describe_attn_fwd uses no _validate=False: all 74 params must be claimed.
    describe_attn_fwd(attn_fwd)
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
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} attn_fwd parity tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
