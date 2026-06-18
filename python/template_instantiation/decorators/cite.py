# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
`ati.cite` — a STRING reference to a metro (or one of its sub-kernels) whose
instantiation practices the citing kernel inherits for shared arguments (rev0 §4.4).
"""


class CiteSpec:
    """One @ati.cite("<op>.<metro>"[.<kernel>]): a STRING reference to a metro (or
    one of its sub-kernels) whose instantiation practices the current kernel pulls
    in for any argument it shares by apparel name (agent-plans/
    ati_aux-kernel-xref_rev0.md §4.4). String-only to avoid circular imports between
    sibling kernel modules; resolved at build time against the `ops` registry."""

    __slots__ = ('target',)

    def __init__(self, target):
        assert isinstance(target, str) and target, (
            '@ati.cite needs a string "<op>.<metro>" or "<op>.<metro>.<kernel>" '
            f'(objects are disallowed to avoid circular imports), got {target!r}')
        parts = target.split('.')
        assert len(parts) in (2, 3), (
            f'@ati.cite target {target!r} must be "<op>.<metro>" or '
            f'"<op>.<metro>.<kernel>"')
        self.target = target

    @property
    def op_name(self):
        return self.target.split('.')[0]

    @property
    def metro_name(self):
        return self.target.split('.')[1]

    @property
    def kernel_name(self):
        """The cited sub-kernel name, or None for a whole-metro cite."""
        parts = self.target.split('.')
        return parts[2] if len(parts) == 3 else None

    def __call__(self, kernel):
        """Stacked-@ form: accumulate this cite onto the kernel below it."""
        from ..specs.finalize import accumulate_spec
        return accumulate_spec(self, kernel)

    def __repr__(self):
        return f'CiteSpec({self.target!r})'


def cite(target):
    """Cite a metro (or one of its sub-kernels) to inherit its instantiation
    practices (rev0 §4.4):

      @ati.cite("op_attn_fwd.triton.attn_fwd")   # one sub-kernel
      @ati.cite("op_attn_bwd.triton_split")      # whole metro (merged interface)
    """
    return CiteSpec(target)
