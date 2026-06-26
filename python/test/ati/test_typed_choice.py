# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for the TypedChoice value wrapper (executive plan Step 1.1)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aotriton.template_instantiation.ir import TypedChoice


def test_type_var():
    c = TypedChoice.parse('*fp16:16')
    assert c.is_tensor
    assert not c.is_constexpr
    assert c.triton_compile_signature == '*fp16:16'
    assert c.type_enum == 'DType::kFloat16'


def test_scalar_type():
    c = TypedChoice.parse('i32')
    assert not c.is_tensor
    assert not c.is_constexpr
    assert c.triton_compile_signature == 'i32'
    assert c.itype == 'int32_t'


def test_constexpr_int():
    c = TypedChoice.parse(0)
    assert c.is_constexpr
    assert not c.is_tensor
    assert c.triton_compile_signature == 0
    assert c.itype == 'int8_t'        # GuessInt picks the smallest fitting type


def test_constexpr_bool():
    c = TypedChoice.parse(False)
    assert c.is_constexpr
    assert c.triton_compile_signature is False


def test_tensor_rank_specialization():
    base = TypedChoice.parse('*bf16:16')
    r4 = base.with_rank(4)
    assert r4.itype == 'const TensorView<4>*'
    r2 = base.with_rank(2)
    assert r2.itype == 'const TensorView<2>*'
    # element type preserved across specialization
    assert r4.triton_compile_signature == '*bf16:16'
    assert r4.type_enum == 'DType::kBFloat16'


def test_rank_suffix_in_type_string():
    # '*fp32:16[2]' — tensor pointer with explicit rank baked into the type string.
    c = TypedChoice.parse('*fp32:16[2]')
    assert c.is_tensor
    assert c.triton_compile_signature == '*fp32:16'   # suffix stripped from sig
    assert c.itype == 'const TensorView<2>*'          # rank settled

    # '*u64[0]' — strideless pointer (rank 0).
    c0 = TypedChoice.parse('*u64[0]')
    assert c0.is_tensor
    assert c0.triton_compile_signature == '*u64'
    assert c0.itype == 'const TensorView<0>*'

    # Without suffix, rank is None (settled later via Axis.choice_for_arg).
    c_norank = TypedChoice.parse('*fp32:16')
    assert c_norank.is_tensor
    assert c_norank.triton_compile_signature == '*fp32:16'


def test_with_rank_noop_on_scalar():
    c = TypedChoice.parse('i32')
    assert c.with_rank(4) is c


def test_equality_and_hash():
    assert TypedChoice.parse('*fp16:16') == TypedChoice.parse('*fp16:16')
    assert TypedChoice.parse(0) != TypedChoice.parse(1)
    assert TypedChoice.parse('*fp16:16') != TypedChoice.parse('*bf16:16')
    s = {TypedChoice.parse('i32'), TypedChoice.parse('i32'), TypedChoice.parse('*fp16:16')}
    assert len(s) == 2


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} TypedChoice tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
