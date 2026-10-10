# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Causal-mask translation shared by the flash tuning stack.

Moved out of aotriton.tune.gpu_utils (a family-neutral module) because this
logic is flash-specific (modular-tune.md §3b/step 12); also de-duplicates the
copy of WindowValue that used to live at the top of reference.py.

Torch-free: safe to import outside a GPU container.
"""

# Note: we don't use Enum class because accessing the integer requires using
#       `.value` property, which makes the code verbose.
class CausalType:
    NONE = 0
    TOP_LEFT = 1
    BOTTOM_RIGHT = 2
    WINDOWED = 3


class WindowValue:
    NONE = 0
    TOP_LEFT_ALIGNED = -2147483647       # 0x80000001. Special value for varlen
    BOTTOM_RIGHT_ALIGNED = -2147483646   # 0x80000002. Special value for varlen


def translate_causal(causal, v3_api):
    window_left, window_right = 0, 0
    if isinstance(causal, tuple):
        assert v3_api, 'Only V3_API supports windowed attention (causal = tuple([window_left, window_right]))'
        window_left, window_right = causal
        causal_type = CausalType.WINDOWED
    elif isinstance(causal, bool):
        causal_type = CausalType.WINDOWED if causal else CausalType.NONE
        if causal:
            # PyTorch SDPA's default causal mask is top-left aligned (upper-left triangle).
            # FA backend always uses bottom-right aligned internally, but for correctness
            # testing we match torch's SDPA convention here.
            window_left = WindowValue.TOP_LEFT_ALIGNED
            window_right = WindowValue.TOP_LEFT_ALIGNED
    else:
        assert causal in [CausalType.NONE, CausalType.TOP_LEFT, CausalType.BOTTOM_RIGHT]
        assert v3_api, 'CausalType.TOP_LEFT/BOTTOM_RIGHT variant is supported thru windowed attention, which requires V3 API'
        if causal == CausalType.TOP_LEFT:
            causal_type = CausalType.WINDOWED
            window_left = WindowValue.TOP_LEFT_ALIGNED
            window_right = WindowValue.TOP_LEFT_ALIGNED
        elif causal == CausalType.BOTTOM_RIGHT:
            causal_type = CausalType.WINDOWED
            window_left = WindowValue.BOTTOM_RIGHT_ALIGNED
            window_right = WindowValue.BOTTOM_RIGHT_ALIGNED
        else:
            causal_type = causal
    return causal_type, window_left, window_right
