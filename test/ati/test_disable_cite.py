# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aux-kernel xref Step 7: @ati.disable inheritance through @ati.cite + the
override guard-rail (agent-plans/ati_aux-kernel-xref_rev0.md §4.5).

  * no local disable      -> inherit the cited target's predicate;
  * local disable         -> REPLACES the cited one (local > cited);
  * extend the cited one  -> callable class + super().__call__;
  * bare-callable + cite  -> FATAL error (it would silently drop the cited
                             disable) unless I_understand_this_overrides_cited_disable=True.
"""

import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'modules' / 'flash'))
sys.path.insert(0, str(REPO / 'modules' / 'flash' / 'kernel'))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation import registry
from aotriton.template_instantiation.describe import describe, get_kernel_spec
from aotriton.template_instantiation.compat import build_kernel_description
from aotriton.template_instantiation.ops.cite import resolve_cites
from aotriton.template_instantiation.builder import AtiDescriptionError

import aot.attn_fwd as _attn_fwd_desc
attn_fwd = _attn_fwd_desc.attn_fwd
from dropout_rng import debug_simulate_encoded_softmax

MAIN_DTYPES = ['*fp16:16', '*bf16:16', '*fp32:16']


# The cited disable, written as a callable class so it can be extended.
class CitedDisabled:
    def __call__(self, f):
        return f.arch == 'gfx1100'


class ExtendedDisabled(CitedDisabled):
    def __call__(self, f):
        if super().__call__(f):
            return True
        return f.arch == 'gfx950'


def _register_attn_fwd_with_disable():
    # attn_fwd is already ATI-described (stacked-@ on import); REPLACE its disables
    # with our known callable-class one so the cite-inheritance assertions are
    # deterministic.
    spec_obj = get_kernel_spec(attn_fwd)
    spec_obj.disables = [ati.disable(CitedDisabled())]
    return build_kernel_description(attn_fwd, family='flash',
                                    source_path='tritonsrc/flash.py',
                                    triton_kernel_name='attn_fwd')


def _citing_specs(*extra):
    return [
        ati.cite('op_attn_fwd.triton.attn_fwd'),
        ati.tensor('R', 'T_io', strides='stride_r?', contiguous=-1,
                   wires_to='encoded_softmax'),
        ati.scalar(['BLOCK_M', 'BLOCK_N'], options=[64]),
        *extra,
    ]


def _resolve_citing(*extra):
    describe(debug_simulate_encoded_softmax, *_citing_specs(*extra), _validate=False)
    spec = get_kernel_spec(debug_simulate_encoded_softmax)
    return resolve_cites(spec, family='flash')


def test_inherits_cited_disable_when_absent():
    registry.clear('flash'); _register_attn_fwd_with_disable()
    spec = _resolve_citing()                 # no local disable
    assert len(spec.disables) == 1
    assert isinstance(spec.disables[0].when, CitedDisabled)


def test_local_callable_class_replaces_no_warning():
    registry.clear('flash'); _register_attn_fwd_with_disable()
    with warnings.catch_warnings():
        warnings.simplefilter('error')       # any warning -> failure
        spec = _resolve_citing(ati.disable(ExtendedDisabled()))
    # local replaces; extension is the author's responsibility via super()
    assert isinstance(spec.disables[0].when, ExtendedDisabled)


def test_bare_lambda_override_is_fatal():
    registry.clear('flash'); _register_attn_fwd_with_disable()
    try:
        _resolve_citing(ati.disable(lambda f: f.arch == 'gfx950'))
    except AtiDescriptionError as e:
        assert 'cited disable' in str(e)
        return
    raise AssertionError('expected a FATAL error for a bare-lambda disable + cite')


def test_affirmed_bare_lambda_no_warning():
    registry.clear('flash'); _register_attn_fwd_with_disable()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _resolve_citing(ati.disable(
            lambda f: f.arch == 'gfx950',
            I_understand_this_overrides_cited_disable=True))


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} disable-cite tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
