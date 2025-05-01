
FACTORY = {
    np.bool   : bool_t,
    np.int8   : int8_t,
    np.int16  : int16_t,
    np.int32  : int32_t,
    np.int64  : int64_t,
    np.uint8  : uint8_t,
    np.uint16 : uint16_t,
    np.uint32 : uint32_t,
    np.uint64 : uint64_t,
    int       : guess_int,
    bool      : bool_t
}

def parse_choices(choices):
    if isinstance(choices, np.ndarray):
        factory = FACTORY[choices.dtype.type]
        return [ factory(v, choices) for v in choices ]
    for t in [bool, int]:
        factory = FACTORY[t]
        if all([isinstance(v, t) for v in choices]):
            return [ factory(v, choices) for v in choices ]
