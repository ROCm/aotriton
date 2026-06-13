# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aux-kernel xref Step 11: debug_simulate_encoded_softmax ported to ATI
(agent-plans/ati_aux-kernel-xref_exec0.md Step 11).

The description declares only R (wired to encoded_softmax) + schema-only perf and
cites the fwd metro's key kernel for everything else. This test exercises the port
directly (independent of the golden): the borrowed struct, the wiring, the inherited
gap arguments, untunability, and the dead-block philox rank."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'modules' / 'flash'))

import aotriton.template_instantiation as ati

# The flash family build (aot package) builds attn_fwd (cited) then debug (citing),
# resolving debug's @ati.cite against the registered key kernel. We assert on that
# real, family-built debug kdesc — rebuilding it would re-run resolve_cites over a
# spec that already carries the inherited cited disable.
import aot


def _build():
    return next(k for k in aot.kernels
                if k.NAME == 'debug_simulate_encoded_softmax')


def test_struct_fields_use_apparel_and_are_complete():
    kdesc = _build()
    fields = {cf.aname: cf.ctype for cf in kdesc.func_cfields}
    assert 'R' not in fields                      # real name hidden
    assert fields['encoded_softmax'] == 'const TensorView<4>*'
    # gaps inherited from the cite, in the struct
    for g in ('dropout_p', 'Num_head_q', 'Max_seqlen_q', 'Max_seqlen_k',
              'philox_seed_ptr', 'philox_offset1', 'philox_offset2'):
        assert g in fields, f'{g} missing from struct'
    # philox is rank 0 (correct; the legacy dead-block <4> quirk is gone)
    assert fields['philox_seed_ptr'] == 'const TensorView<0>*'


def test_wiring_recorded():
    kdesc = _build()
    assert kdesc.apparel_of('R') == 'encoded_softmax'
    assert kdesc.real_of('encoded_softmax') == 'R'


def test_untunable_schema_only():
    kdesc = _build()
    assert kdesc.is_tunable is False
    names = [pp.name for pp in kdesc.tune.schema.params]
    assert names == ['BLOCK_M', 'BLOCK_N']


def test_only_multichoice_axis_is_tio():
    kdesc = _build()
    multi = [a for a in kdesc.axes_multi]
    # encoded_softmax (T_io) is the only multi-choice axis; everything else trivial.
    assert len(multi) == 1
    assert multi[0].repr_arg == 'R'
    assert multi[0].radix == 3


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} port-debug tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
