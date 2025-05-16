#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl

@triton.jit
def closed_interval_isect(a_s, a_e, b_s, b_e):
    if (b_s > a_e or a_s > b_e) or (a_s > a_e):
        return 0, -1
    o_s = max(a_s, b_s)
    o_e = min(a_e, b_e)
    return o_s, o_e

@triton.jit
def is_closed_interval_empty(a_s, a_e) -> bool:
    return a_s > a_e

@triton.jit
def closed_interval_size(a_s, a_e) -> bool:
    return max(0, a_e - a_s + 1)

# Convenience function to load with optional boundary checks.
# "First" is the major dim, "second" is the minor dim.
@triton.jit
def load_fn(ptrs, offset_first, offset_second, _in_boundary_first, _in_boundary_second):
    boundary_first = _in_boundary_first
    boundary_second = _in_boundary_second
    """
    # Debugging GPU segfault
    boundary_first = 0
    boundary_second = 0
    mask = (offset_first[:, None] < boundary_first) & \
           (offset_second[None, :] < boundary_second)
    return tl.load(ptrs, mask=mask, other=0.0)
    """
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & \
               (offset_second[None, :] < boundary_second)
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor

@triton.jit
def mload1d(
        REGS : tl.constexpr,
        i_base,
        i_start,
        i_nums,
):
    offs = tl.arange(0, REGS) + i_start
    i_ptrs = i_base + offs
    # return tl.load(i_base + offs)
    overflow = i_start + REGS - i_nums
    # if overflow <= 0:
    #     return tl.load(i_ptrs)
    i_ptrs_mask = tl.full([REGS], 1, dtype=tl.int1)
    i_ptrs_mask = i_ptrs_mask & (offs < i_nums)
    return tl.load(i_ptrs, mask=i_ptrs_mask, other=0.0)

@triton.jit
def mload2d(
        REG_ROWS : tl.constexpr,
        REG_COLS : tl.constexpr,
        i_base,
        i_start_row,
        i_start_col,
        i_rows,
        i_cols,
        stride_row,
        stride_col,
):
    off_rows = tl.arange(0, REG_ROWS) + i_start_row
    off_cols = tl.arange(0, REG_COLS) + i_start_col
    i_ptrs = i_base + off_rows[:, None] * stride_row + off_cols[None, :] * stride_col
    row_overflow = i_start_row + REG_ROWS - i_rows
    col_overflow = i_start_col + REG_COLS - i_cols
    # if row_overflow <= 0 and col_overflow <= 0:
    # if NOCHECK:
    #     return tl.load(i_ptrs)
    i_ptrs_mask = tl.full([REG_ROWS, REG_COLS], 1, dtype=tl.int1)
    if row_overflow > 0:
        i_ptrs_mask = i_ptrs_mask & (off_rows[:, None] < i_rows)
    if col_overflow > 0:
        i_ptrs_mask = i_ptrs_mask & (off_cols[None, :] < i_cols)
    return tl.load(i_ptrs, mask=i_ptrs_mask, other=0.0)

@triton.jit
def mstore2d(
        registers,
        REG_ROWS : tl.constexpr,
        REG_COLS : tl.constexpr,
        o_base,
        o_start_row,
        o_start_col,
        o_rows,
        o_cols,
        stride_row,
        stride_col,
):
    off_rows = tl.arange(0, REG_ROWS) + o_start_row
    off_cols = tl.arange(0, REG_COLS) + o_start_col
    o_ptrs = o_base + off_rows[:, None] * stride_row + off_cols[None, :] * stride_col
    o_ptrs_mask = tl.full([REG_ROWS, REG_COLS], 1, dtype=tl.int1)
    row_overflow = o_start_row + REG_ROWS - o_rows
    if row_overflow > 0:
        o_ptrs_mask = o_ptrs_mask & (off_rows[:, None] < o_rows)
    col_overflow = o_start_col + REG_COLS - o_cols
    if col_overflow > 0:
        o_ptrs_mask = o_ptrs_mask & (off_cols[None, :] < o_cols)
    tl.store(o_ptrs, registers, mask=o_ptrs_mask)
    # Do not return variables unused by the caller
    # See: https://github.com/triton-lang/triton/issues/5768
    # return o_ptrs, o_ptrs_mask
