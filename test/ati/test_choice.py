# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for the Choice value wrapper (executive plan Step 1.1)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aotriton.template_instantiation.ir import Choice


def test_tensor_dtype():
    c = Choice.parse('*fp16:16')
    assert c.is_tensor
    assert not c.is_constexpr
    assert c.triton_compile_signature == '*fp16:16'
    assert c.type_enum == 'DType::kFloat16'


def test_scalar_type():
    c = Choice.parse('i32')
    assert not c.is_tensor
    assert not c.is_constexpr
    assert c.triton_compile_signature == 'i32'
    assert c.itype == 'int32_t'


def test_constexpr_int():
    c = Choice.parse(0)
    assert c.is_constexpr
    assert not c.is_tensor
    assert c.triton_compile_signature == 0
    assert c.itype == 'int8_t'        # GuessInt picks the smallest fitting type


def test_constexpr_bool():
    c = Choice.parse(False)
    assert c.is_constexpr
    assert c.triton_compile_signature is False


def test_tensor_rank_specialization():
    base = Choice.parse('*bf16:16')
    r4 = base.with_rank(4)
    assert r4.itype == 'const TensorView<4>*'
    r2 = base.with_rank(2)
    assert r2.itype == 'const TensorView<2>*'
    # element type preserved across specialization
    assert r4.triton_compile_signature == '*bf16:16'
    assert r4.type_enum == 'DType::kBFloat16'


def test_with_rank_noop_on_scalar():
    c = Choice.parse('i32')
    assert c.with_rank(4) is c


def test_equality_and_hash():
    assert Choice.parse('*fp16:16') == Choice.parse('*fp16:16')
    assert Choice.parse(0) != Choice.parse(1)
    assert Choice.parse('*fp16:16') != Choice.parse('*bf16:16')
    s = {Choice.parse('i32'), Choice.parse('i32'), Choice.parse('*fp16:16')}
    assert len(s) == 2


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} Choice tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
