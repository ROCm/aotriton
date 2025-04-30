# Copyright ©2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import numpy as np

SIGNATURE_TO_C = {
    'fp32'    : 'float',
    '*fp32'   : 'const float*',
    '*fp16'   : 'const __fp16*',
    '*bf16'   : 'const __bf16*',
    'i8'      : 'int8_t',
    'i16'     : 'int16_t',
    'i32'     : 'int32_t',
    'i64'     : 'int64_t',
    'u8'      : 'uint8_t',
    'u16'     : 'uint16_t',
    'u32'     : 'uint32_t',
    'u64'     : 'uint64_t',
    np.int8   : 'int8_t',
    np.int16  : 'int16_t',
    np.int32  : 'int32_t',
    np.int64  : 'int64_t',
    np.uint8  : 'uint8_t',
    np.uint16 : 'uint16_t',
    np.uint32 : 'uint32_t',
    np.uint64 : 'uint64_t',
}
C_SIZE = {
    'bool'              : 1,
    'const __bf16*'     : 8,
    'const __fp16*'     : 8,
    'const float*'      : 8,
    'float'             : 4,
    'int8_t'            : 1,
    'int16_t'           : 2,
    'int32_t'           : 4,
    'int64_t'           : 8,
    'uint8_t'           : 1,
    'uint16_t'          : 2,
    'uint32_t'          : 4,
    'uint64_t'          : 8,
}

