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

# The runtime values a per-call rule may read (`ati.derives` yielded by a
# tuning module's `variants`), each with the range of the C++ value it is wired
# to: every value C++ can hold must select a build. `hdim_qk` is the call's Q/K
# head dim, the sequence lengths are the call's maxima.
_INT32 = (-2**31, 2**31 - 1)
RUNTIME_VALUES = {
    'hdim_qk': _INT32,
    'seqlen_q': _INT32,
    'seqlen_k': _INT32,
}


@dataclass(frozen=True, kw_only=True)
class FmhaInputMetadata:
    """What to compute, for the forward, dQ and dK/dV alike, on every arch.

    Set by the caller, never by a tuning policy. A field an arch's kernel does
    not read is ignored there.
    """

    num_heads: int
    # "bf16" or "f16".
    dtype_str: str
    # AOTriton's CAUSAL_TYPE: 0 none, 1 top-left, 2 bottom-right, 3 window.
    # The only masking field; `causal` and `window` derive from it.
    causal_type: int

    # The compiled tiles: a functional's BLOCK_DMODEL, which AOT fills. Required
    # but nullable: a JIT caller passes None and gets the tile from `plan`.
    block_dmodel: int | None
    block_dmodel_v: int | None = None  # filled: block_dmodel
    # Whether calls may bring head dims narrower than the tile: a functional's
    # PADDED_HEAD in AOT, derived by `plan` in JIT.
    padded_head: bool | None = None
    # The caller's real head dims. They only make a difference in JIT, where
    # `plan` derives the tile from them; AOT leaves them None, since calls pass
    # the real head dims at run time.
    head_dim: int | None = None
    head_dim_v: int | None = None  # filled: head_dim

    num_kv_heads: int | None = None  # filled: num_heads
    sm_scale: float | None = None  # None: the kernel derives 1/sqrt of the head dim
    bias: bool = False
    dropout: bool = False
    # dK/dV: whether LSE and Delta use the (T, H) layout rather than (H, T).
    # A tuning record only: kernels decode the layout from VarlenBits at run
    # time, so a build serves both.
    lse_layout_th: bool = False

    def __post_init__(self):
        """The one place input metadata is checked and its defaults filled.

        A default is filled only when its source is known, so a JIT caller's
        tile fields stay None until `plan` sets `block_dmodel` through
        `dataclasses.replace`, which runs this again. Replacing the tile of
        metadata whose V tile was already filled keeps that V tile.
        """
        if self.causal_type not in (CAUSAL_NONE, CAUSAL_TOP_LEFT, CAUSAL_BOTTOM_RIGHT, CAUSAL_WINDOW):
            raise ValueError(f"causal_type must be 0, 1, 2 or 3, got {self.causal_type!r}")
        if self.dtype_str not in ("bf16", "f16"):
            raise ValueError(f"dtype_str must be 'bf16' or 'f16', got {self.dtype_str!r}")
        if self.block_dmodel is None and self.head_dim is None:
            raise ValueError("set block_dmodel (AOT) or head_dim (JIT, for plan)")
        # Frozen: defaults are filled through object.__setattr__.
        if self.num_kv_heads is None:
            object.__setattr__(self, "num_kv_heads", self.num_heads)
        if self.num_heads % self.num_kv_heads:
            raise ValueError(f"num_heads ({self.num_heads}) must be a multiple of num_kv_heads "
                             f"({self.num_kv_heads}): each KV head owns a whole group of query heads")
        if self.head_dim_v is None and self.head_dim is not None:
            object.__setattr__(self, "head_dim_v", self.head_dim)
        if self.block_dmodel_v is None and self.block_dmodel is not None:
            object.__setattr__(self, "block_dmodel_v", self.block_dmodel)
        # The real head dims, when known, have to fit the compiled tiles.
        if self.head_dim is not None and self.block_dmodel is not None:
            if self.head_dim > self.block_dmodel:
                raise ValueError(f"head_dim {self.head_dim} does not fit block_dmodel {self.block_dmodel}")
            if self.head_dim_v > self.block_dmodel_v:
                raise ValueError(f"head_dim_v {self.head_dim_v} does not fit block_dmodel_v {self.block_dmodel_v}")

    @property
    def causal(self) -> bool:
        """Any causal mask, windowed or not."""
        return self.causal_type != CAUSAL_NONE

    @property
    def window(self) -> bool:
        """A sliding window: a left bound on top of the causal right one."""
        return self.causal_type == CAUSAL_WINDOW
