# SPDX-License-Identifier: Apache-2.0

"""The one input-metadata class every flyc attention kernel takes.

Keeps the `fmha_` prefix because arch modules import it by bare name; see
`fmha_knobs.py` for the naming rule.

flydsl-free: the code generator builds one per functional
(`modules/flash/aot/_flyc_common.flyc_input_metadata`), and it runs with no
flydsl installed.
"""

from dataclasses import dataclass

# AOTriton's CAUSAL_TYPE.
CAUSAL_NONE = 0
CAUSAL_TOP_LEFT = 1
CAUSAL_BOTTOM_RIGHT = 2
CAUSAL_WINDOW = 3


@dataclass(frozen=True)
class FmhaInputMetadata:
    """What to compute, for the forward, dQ and dK/dV alike, on every arch.

    Set by the caller, never by a tuning policy. A field an arch's kernel does
    not read is ignored there.
    """

    num_heads: int
    # The caller's real head dims. AOT passes the functional's tile here.
    head_dim: int
    # "bf16" or "f16".
    dtype_str: str
    # AOTriton's CAUSAL_TYPE: 0 none, 1 top-left, 2 bottom-right, 3 window.
    # The only masking field; `causal` and `window` derive from it.
    causal_type: int
    num_kv_heads: int | None = None  # None: same as num_heads
    head_dim_v: int | None = None  # None: same as head_dim
    sm_scale: float | None = None  # None: 1/sqrt(real head dim)
    bias: bool = False
    dropout: bool = False
    # dK/dV: whether LSE and Delta use the (T, H) layout rather than (H, T).
    # A tuning record only: kernels decode the layout from VarlenBits at run
    # time, so a build serves both.
    lse_layout_th: bool = False

    # The compiled tiles and whether the real head dims are narrower. They are
    # a functional's BLOCK_DMODEL and PADDED_HEAD: AOT fills them, a JIT caller
    # gets them from the dispatcher's `plan`.
    block_dmodel: int | None = None
    block_dmodel_v: int | None = None
    padded_head: bool | None = None

    def __post_init__(self):
        if self.causal_type not in (CAUSAL_NONE, CAUSAL_TOP_LEFT, CAUSAL_BOTTOM_RIGHT, CAUSAL_WINDOW):
            raise ValueError(f"causal_type must be 0, 1, 2 or 3, got {self.causal_type!r}")

    @property
    def causal(self) -> bool:
        """Any causal mask, windowed or not."""
        return self.causal_type != CAUSAL_NONE

    @property
    def window(self) -> bool:
        """A sliding window: a left bound on top of the causal right one."""
        return self.causal_type == CAUSAL_WINDOW
