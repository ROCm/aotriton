#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl
from masked_load_store import mstore2d
from typing import Tuple

@triton.jit
def composed_offs_1d(
        D0 : tl.constexpr,
        D1 : tl.constexpr,
        D2 : tl.constexpr
):
    x = tl.arange( 0 +  0, D0 +  0 +  0) if D0 > 0 else 0
    y = tl.arange( 0 + D0, D0 + D1 +  0) if D1 > 0 else 0
    z = tl.arange(D0 + D1, D0 + D1 + D2) if D2 > 0 else 0
    return (x, y, z)


@triton.jit
def composed_zeros_2d(
        M  : tl.constexpr,
        D0 : tl.constexpr,
        D1 : tl.constexpr,
        D2 : tl.constexpr,
        dtype=tl.float32
):
    x = tl.zeros([M, D0], dtype=dtype) if D0 > 0 else 0
    y = tl.zeros([M, D1], dtype=dtype) if D1 > 0 else 0
    z = tl.zeros([M, D2], dtype=dtype) if D2 > 0 else 0
    return (x, y, z)

@triton.jit
def composed_ptrs(
        T,
        stride_0, stride_1, stride_2, stride_3,
        offset_0, offset_1, offset_2,
        D0 : tl.constexpr,
        D1 : tl.constexpr,
        D2 : tl.constexpr,
        TRANSPOSED: tl.constexpr = False
):
    T += offset_0 * stride_0 + offset_1 * stride_1
    d0, d1, d2 = composed_offs_1d(D0, D1, D2)
    if TRANSPOSED:
        x = T + d0[:, None] * stride_3 + offset_2[None, :] * stride_2
        y = T + d1[:, None] * stride_3 + offset_2[None, :] * stride_2
        z = T + d2[:, None] * stride_3 + offset_2[None, :] * stride_2
        return (x, y, z)
    else:
        x = T + offset_2[:, None] * stride_2 + d0[None, :] * stride_3
        y = T + offset_2[:, None] * stride_2 + d1[None, :] * stride_3
        z = T + offset_2[:, None] * stride_2 + d2[None, :] * stride_3
        return (x, y, z)


@triton.jit
def composed_load(
        x : tl.tensor,
        y : tl.tensor,
        z : tl.tensor,
        ROWS,
        D0 : tl.constexpr,
        D1 : tl.constexpr,
        D2 : tl.constexpr,
        i_rows,
        i_cols,
        other,
        PADDED_ROW : tl.constexpr,
        PADDED_COL : tl.constexpr,
        TRANSPOSED : tl.constexpr = False
):
    COLS0, COLS1, COLS2 = composed_offs_1d(D0, D1, D2)
    if TRANSPOSED:
        ptrs_mask = ROWS[None, :] < i_rows if PADDED_ROW or PADDED_COL else None
        # x_mask is only needed when D0 > 0 and D1 == 0 and D2 == 0
        x_mask = ptrs_mask & (COLS0[:, None] < i_cols) if PADDED_COL and D1 == 0 else ptrs_mask
        y_mask = ptrs_mask & (COLS1[:, None] < i_cols) if PADDED_COL and D2 == 0 else ptrs_mask
        z_mask = ptrs_mask & (COLS2[:, None] < i_cols) if PADDED_COL and True    else ptrs_mask
    else:
        ptrs_mask = ROWS[:, None] < i_rows if PADDED_ROW or PADDED_COL else None
        # x_mask is only needed when D0 > 0 and D1 == 0 and D2 == 0
        x_mask = ptrs_mask & (COLS0[None, :] < i_cols) if PADDED_COL and D1 == 0 else ptrs_mask
        y_mask = ptrs_mask & (COLS1[None, :] < i_cols) if PADDED_COL and D2 == 0 else ptrs_mask
        z_mask = ptrs_mask & (COLS2[None, :] < i_cols) if PADDED_COL and True    else ptrs_mask
    x = tl.load(x) if x_mask is None else tl.load(x, mask=x_mask, other=other)
    y = tl.load(y) if y_mask is None else tl.load(y, mask=y_mask, other=other)
    z = tl.load(z) if z_mask is None else tl.load(z, mask=z_mask, other=other)
    return (x, y, z)


@triton.jit
def composed_advance(
        x, y, z,
        advance,
        D0 : tl.constexpr,
        D1 : tl.constexpr,
        D2 : tl.constexpr
):
    x = x + advance
    y = y + advance if D1 > 0 else y
    z = z + advance if D2 > 0 else z
    return (x, y, z)


@triton.jit
def composed_to(
        x, y, z,
        dtype,
):
    return (x.to(dtype), y.to(dtype), z.to(dtype))


# Inner dot
#   lhs: (M, (K0, K1, K2))
#   rhs: ((K0, K1, K2), N)
#   acc: (M, N)
@triton.jit
def composed_dot_both(
        lx, ly, lz,
        rx, ry, rz,
        acc,
        D0 : tl.constexpr,
        D1 : tl.constexpr,
        D2 : tl.constexpr
):
    acc = tl.dot(lx, rx, acc=acc)
    acc = tl.dot(ly, ry, acc=acc) if D1 > 0 else acc
    acc = tl.dot(lz, rz, acc=acc) if D2 > 0 else acc
    return acc


# Inner dot, broadcasting lhs to composed rhs
#   lhs: (M, N)
#   rhs: (N, (K0, K1, K2))
#   acc: (M, (K0, K1, K2))
@triton.jit
def composed_dot_rhs(
        lhs,
        rx, ry, rz,
        ax, ay, az,
        D0 : tl.constexpr,
        D1 : tl.constexpr,
        D2 : tl.constexpr
):
    ax = tl.dot(lhs, rx, acc=ax)
    ay = tl.dot(lhs, ry, acc=ay) if D1 > 0 else ay
    az = tl.dot(lhs, rz, acc=az) if D2 > 0 else az
    return (ax, ay, az)


# Element-wise mul, broadcasting rhs to composed lhs
@triton.jit
def composed_mul_lhs(
        lx, ly, lz,
        rhs,
        D0 : tl.constexpr,
        D1 : tl.constexpr,
        D2 : tl.constexpr
):
    lx = lx * rhs
    ly = ly * rhs if D1 > 0 else ly
    lz = lz * rhs if D2 > 0 else lz
    return (lx, ly, lz)

@triton.jit
def composed_store(
        regs0, regs1, regs2,
        REG_ROWS  : tl.constexpr,
        REG_COLS0 : tl.constexpr,
        REG_COLS1 : tl.constexpr,
        REG_COLS2 : tl.constexpr,
        o_base,
        o_start_row,
        o_start_col,
        o_rows,
        o_cols,
        stride_row,
        stride_col,
):
    mstore2d(regs0,
             REG_ROWS,
             REG_COLS0,
             o_base=o_base,
             o_start_row=o_start_row,
             o_start_col=o_start_col,
             o_rows=o_rows,
             o_cols=o_cols,
             stride_row=stride_row,
             stride_col=stride_col)
    if REG_COLS1 > 0:
        mstore2d(regs1,
                 REG_ROWS,
                 REG_COLS1,
                 o_base=o_base,
                 o_start_row=o_start_row,
                 o_start_col=o_start_col + REG_COLS0,
                 o_rows=o_rows,
                 o_cols=o_cols,
                 stride_row=stride_row,
                 stride_col=stride_col)
    if REG_COLS2 > 0:
        mstore2d(regs2,
                 REG_ROWS,
                 REG_COLS2,
                 o_base=o_base,
                 o_start_row=o_start_row,
                 o_start_col=o_start_col + REG_COLS0 + REG_COLS1,
                 o_rows=o_rows,
                 o_cols=o_cols,
                 stride_row=stride_row,
                 stride_col=stride_col)