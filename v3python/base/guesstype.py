
from .ttype import *

MAP_NP_TO_TTYPE = {
}

MAP_TO_TTYPE = {
    np.bool   : bool_t,
    np.int8   : int8_t,
    np.int16  : int16_t,
    np.int32  : int32_t,
    np.int64  : int64_t,
    np.uint8  : uint8_t,
    np.uint16 : uint16_t,
    np.uint32 : uint32_t,
    np.uint64 : uint64_t,
    'fp32'    : float_t,
    'i8'      : int8_t,
    'i16'     : int16_t,
    'i32'     : int32_t,
    'i64'     : int64_t,
    'u8'      : uint8_t,
    'u16'     : uint16_t,
    'u32'     : uint32_t,
    'u64'     : uint64_t,
    'u64:8'   : stride_a8,
    'u64:16'  : stride_a16,
    # '*fp32'   : tensor_type,
    # '*fp16'   : tensor_type,
    # '*bf16'   : tensor_type,
}

def create_anyrank_tensor():
    # split = tstring[1:].split(':')
    # baset = split[0]
    # alignment = 0 if len(split) <= 1 else split[1]
    return TI(nbits=64, tensor_rank=-1)

# choices are list of strings
def guess_ttype(choices):
    # explicitly stated
    if isinstance(choices, np.ndarray):
        return MAP_NP_TO_TTYPE(choices.dtype.type)
    # Scalars
    if all([isinstance(v, bool) for v in choices]):
        return bool_t
    if all([isinstance(v, int) for v in choices]):
        if max(choices) < 16:  # Leave with some margin
            return int8_t
        if max(choices) < 8192:
            return int16_t
        return int32_t
    if all([v.startswith('*') for v in choices]):
        return create_generic_tensor_type()
    if isinstance(choices[0], ConditionalValue):
        return choices[0].resolve_ttype()
    tstring = conditional_or_tstring
    return MAP_TSTRING_TO_TTYPE[tstring]
    assert False, f"Cannot guess valued parameter type from {choices=}"


