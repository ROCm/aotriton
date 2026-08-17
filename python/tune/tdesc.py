# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import itertools
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from dataclasses import asdict, dataclass, fields

'''
A dual-purpose class for task dispatch and GPU worker execution.

IMPORTANT — lazy kernel initialization rule for get_impl():
    Subclasses of TuningDescription MUST NOT import kernel modules or
    instantiate kernel objects at module level or in __init__. Kernel modules
    typically import torch (via reference.py), which is unavailable outside
    GPU containers. dispatch_tasks.py instantiates every TuneDesc subclass at
    startup (to build argparse subparsers from get_entry_choices()), so a
    top-level torch import breaks dispatch on machines without torch.

    A subclass typically owns one lazily-imported "provider" module per
    tuning level (e.g. flash's level_kernel.py / level_op.py) and dispatches
    get_impl()/_do_probe_all_impls() to the right one based on the DSL name's
    prefix (see ImplSelector.split_dsl_name() below). Always import the
    provider module lazily, inside the method that needs it -- never at
    module level or in __init__:

        # modules/<family>/tune/desc.py
        class MyTune(TuningDescription):
            def get_impl(self, name):
                level, iface = ImplSelector.split_dsl_name(name)
                if level == 'kernel':
                    from . import level_kernel as provider
                elif level == 'op':
                    from . import level_op as provider
                else:
                    raise ValueError(f"{type(self).__name__} has no tuning "
                                      f"level {level!r} (from impl {name!r})")
                return provider.get_impl(iface)
'''


@dataclass
class ImplSelector:
    """Unified impl-selector DSL (modular-tune.md §4.2, Revision note 3):
    `[<tuning_level> '.'] <iface_name> '=' <impl_index>`.

    A single three-field dataclass shared by every tuning level of every
    family. `tuning_level` defaults to 'kernel' and its prefix is omitted by
    as_text() in that case, so plain kernel-level selector text is unchanged
    from before ('attn_fwd=3'); op-level selectors are prefixed
    ('op.attn_fwd=1').

    IMPORTANT: the `op.` (or any non-kernel level's) prefix is surface syntax
    ONLY -- it must never reach storage. `TuningDescription.list_impls()`
    returns DSL-spelled names (prefix included); `iface_name` on this
    dataclass is always bare (never e.g. 'attn_fwd_op').

    `ImplSelector` is the DSL/wire parser used by `testrun`, `exaid` and
    `localq.handlers` -- it is deliberately NOT part of `TuningDescription`'s
    API (Revision note 3 item 3/4): `list_impls`/`get_impl`/`probe_all_impls`
    all take the plain DSL name string (e.g. 'attn_fwd' or 'op.attn_fwd'),
    never an `ImplSelector` instance. `split_dsl_name()` below is the shared
    primitive both sides use: `ImplSelector.parse_text()` uses it to split the
    LHS of `name=index`, and `TuningDescription` implementations use it
    directly (with no `=index` present) to dispatch `get_impl()`/
    `probe_all_impls()` on the level prefix.
    """
    tuning_level: str = 'kernel'
    iface_name: str = ''
    impl_index: int = -1

    @staticmethod
    def split_dsl_name(name: str) -> tuple[str, str]:
        """Split a bare DSL name (no '=index'), e.g. 'attn_fwd' or
        'op.attn_fwd', into (tuning_level, iface_name). Absent prefix means
        'kernel' -- the unmarked, default level. rpartition (not split) so a
        future dotted iface name does not misparse."""
        level, _, iface = name.rpartition('.')
        return level or 'kernel', iface

    @property
    def dsl_name(self) -> str:
        """DSL name with the level prefix applied but no '=index' -- what
        `TuningDescription.get_impl()`/`probe_all_impls()` take, e.g.
        'attn_fwd' or 'op.attn_fwd'."""
        prefix = '' if self.tuning_level == 'kernel' else f'{self.tuning_level}.'
        return f'{prefix}{self.iface_name}'

    @staticmethod
    def parse_text(line: str) -> "ImplSelector":
        lhs, index = line.split('=')
        level, iface = ImplSelector.split_dsl_name(lhs)
        return ImplSelector(tuning_level=level, iface_name=iface, impl_index=int(index))

    def as_text(self) -> str:
        return f'{self.dsl_name}={self.impl_index}'


class TuningDescription(ABC):
    """Lists and resolves all impls of a family directly, keyed by DSL name
    (modular-tune.md Revision note 3 item 1). There is no intermediate
    per-level strategy object: each family's `TuningDescription` subclass
    (e.g. flash's `FlashTune`) implements `list_impls`/`get_impl`/
    `_do_probe_all_impls`/`probe_impl_desc` itself, dispatching internally on
    the DSL name's prefix (`ImplSelector.split_dsl_name()`).

    No constructor argument: a `TuningDescription` instance is not scoped to
    one tuning level. `FlashTune()` takes no `level=` -- `list_impls()`
    reports every level's impls, DSL-spelled, and `get_impl()`/
    `probe_all_impls()` resolve whichever DSL name they are given.
    """

    @property
    @abstractmethod
    def ENTRY_CLASS(self):
        pass

    @property
    @abstractmethod
    def INPUT_METADATA(self):
        pass

    '''
    get_entry_choices:
        Return an ENTRY_CLASS instance where each field contains a list of possible choices.

        Returns:
            An instance of ENTRY_CLASS where each field is a list of values rather than a single value.

        Example:
            For FlashEntry, instead of:
                FlashEntry(dtype='float16', hdim=32, ...)
            Return:
                FlashEntry(dtype=['float16', 'bfloat16', 'float32'],
                          hdim=[16, 32, 48, 64, ...], ...)

        Note:
            This violates the type hints of ENTRY_CLASS (e.g., dtype: str becomes list[str]),
            but it's only used for parameter space definition, not actual entry instances.
    '''
    @abstractmethod
    def get_entry_choices(self):
        pass

    '''
    validate_entry:
        Validate if an entry combination is valid.

        Args:
            entry: An ENTRY_CLASS instance to validate

        Returns:
            True if the entry is valid, False otherwise

        Subclasses can override this to skip invalid parameter combinations.
    '''
    def validate_entry(self, entry) -> bool:
        return True

    '''
    validate_hw_feature:
        Validate if an entry is supported on a specific architecture.

        Args:
            arch: Target GPU architecture string (e.g., 'gfx942', 'gfx1100')
            entry: An ENTRY_CLASS instance to validate

        Returns:
            (supported: bool, reason: str)
            supported is True if the entry is valid for this arch.
            reason is a human-readable explanation when supported is False.

        Subclasses override this to reject hardware-unsupported configurations.
        Unlike validate_entry (which is arch-independent), this is called per
        (arch, entry) pair in task_config_gen and skips unsupported combinations.
    '''
    def validate_hw_feature(self, arch: str, entry) -> tuple[bool, str]:
        return True, ''

    '''
    generate_entries_from_choices:
        Generate entry instances from choices.

        Args:
            choices: An ENTRY_CLASS instance where each field is a list of allowed values.
                    If None, uses get_entry_choices().

        Yields:
            ENTRY_CLASS instances with single values (proper type-conforming instances)

        This method can be implemented generically in the base class since it just
        does cartesian product of all choice lists.
    '''
    def generate_entries_from_choices(self, choices=None):
        if choices is None:
            choices = self.get_entry_choices()

        # Get field names and their choice lists
        field_names = [f.name for f in fields(choices)]
        choice_lists = [getattr(choices, f.name) for f in fields(choices)]

        # Generate cartesian product
        for value_tuple in itertools.product(*choice_lists):
            entry = self.ENTRY_CLASS(*value_tuple)
            if self.validate_entry(entry):
                yield entry

    '''
    generate_entries:
        Generate an entry object that can uniquely locate a entry in the tuning
        table (sans Arch/GPU selection, which is handled in upper layer)

        This is now implemented as a convenience wrapper around the two-step process.
        Subclasses can override this if they need custom logic, but typically should
        just implement get_entry_choices() instead.

    Note:
        An entry will be extended into Input Metadata object, which contains
        additional fields like batch sizes and PRNG seeds.
        This step should be handled inside run_test()
    '''
    def generate_entries(self):
        return self.generate_entries_from_choices()

    @abstractmethod
    def list_impls(self, entry, arch: str | None = None) -> list[str]:
        """DSL-spelled interface names covering every tuning level this
        family supports, e.g. ['attn_fwd', 'bwd_kernel_dk_dv',
        'bwd_kernel_dq', 'bwd_kernel_fuse', 'op.attn_fwd', 'op.attn_bwd'] for
        flash. Unprefixed names are the 'kernel' level (the DSL's unmarked
        default); every other level is prefixed `f'{level}.'`.

        Must be answerable without a GPU/torch/pyaotriton -- this is pure
        entry-based enumeration (Revision note 3 item 4)."""
        pass

    @abstractmethod
    def get_impl(self, name: str):
        """Resolve one DSL-spelled name (e.g. 'attn_fwd' or 'op.attn_fwd',
        never an ImplSelector) to its impl object, lazily importing whichever
        provider module owns that level (see the module docstring above).

        MUST use lazy initialization -- import torch/pyaotriton-dependent
        modules inside this method, not at module level.

        Implementations should attempt the resolution unconditionally (no
        up-front "does this process have the right library" check) and only
        translate a resulting ImportError into a clearer message naming the
        impl and both libraries -- callers (testrun's interactive `probe`,
        `exaid`) decide whether/how to route around a failure; get_impl()
        itself must not refuse a name it merely suspects will fail."""
        pass

    @abstractmethod
    def probe_impl_desc(self, kernel, args) -> dict:
        """Extract impl_desc from a probing run's extargs.

        Called by run_single_benchmark after kernel.direct_call(direct_inputs, args)
        with probe=True. Returns a JSON-serialisable dict that uniquely identifies
        the chosen implementation (e.g., {psels, copts} for HSACO kernels,
        {backend_index} for op backends).

        DELIBERATELY overlaps with probe_all_impls()/_do_probe_all_impls() --
        do not "optimise" this away by threading probe_all_impls()'s psels/
        copts through the dispatcher-to-GPU-worker fanout message instead of
        calling this. That trade was considered and rejected (Revision note 3
        item 6):
          (a) it double-confirms the impl_desc actually run, independent of
              whatever the earlier enumeration pass said would be there;
          (b) it keeps the dispatcher<->GPU-worker IPC to a single
              `impl_index` integer on the wire, with no need to serialise
              psels/copts through it; and
          (c) it is what lets the DSL be a name plus a bare integer at all --
              `op.attn_fwd=1` / `bwd_kernel_dk_dv=10` stay writable by hand
              precisely because the full identity is recovered here, at run
              time on the GPU worker, from the extargs actually used -- not
              carried on the wire. Without this method the selector would
              have to carry psels/copts to mean anything.

        Since this method's 2-argument (kernel, args) signature carries no
        explicit level, implementations dispatch by inspecting `args`/
        `kernel` themselves (duck typing) -- consistent with (c) above: full
        identity recovery from what was actually run, not from external state.

        Args:
            kernel: the impl object returned by get_impl()
            args: the extargs object returned by create_extargs(probe=True)
        """
        pass

    @abstractmethod
    def _do_probe_all_impls(self, entry, im, which_impl: str, pt: Path) -> list[dict]:
        """Enumerate the candidate implementation variants (e.g. HSACO
        indices for a kernel-level impl, backend indices for an op-level
        impl) for the DSL name `which_impl`. One dict per candidate, in
        impl_index order. Was `TuningDescription._do_probe_backends`."""
        pass

    def probe_all_impls(self, root: Path, which_impl: str) -> list[dict]:
        """Was `probe_backends` (Revision note 3 item 5, keeps the "probe"
        terminology). `which_impl` is a DSL-spelled name, e.g. 'attn_fwd' or
        'op.attn_fwd' -- as returned by `list_impls()`."""
        entry, tests = self.get_entry(root, and_tests=True)
        test = tests[0]
        im = self.INPUT_METADATA.from_dict(test["input_metadata"])
        pt = Path(test["pt_file"])
        return self._do_probe_all_impls(entry, im, which_impl, pt)

    @abstractmethod
    def _gen_ref(self, entry, root: Path, extra_ims: list = []):  # Gen [tname: str, input_metadata, pt: Path]
        """
        Inputs:
            entry: an object to describe an entry in tuning database.
            root: the root path to store tensors of testing cases (PLURAL).
        Outputs:
            tname: testing case name, ideally should be consistent with the .pt file name
            input_metadata: the entry object, with extra/translated arguments that's necessary to launch the kernel.
                e.g. batch/nheads will be filled to a reasonable number.
            pt: the actual .pt tensor path.
        Note:
            input_metadata may still contain fields to be translated for kernel use, e.g., .sm_scale = 'l1'.
            The .pt file must only contain arguments for kernel use directly.
        """
        pass

    def prepare_data(self, entry, root: Path, extra_ims: list = []):
        def iterate_test():
            for tname, im, pt in self._gen_ref(entry, root, extra_ims):
                yield {'test_name': tname, 'input_metadata' : asdict(im), 'pt_file': pt.as_posix()}
        with open(root / 'entry.json', 'w') as f:
            json.dump({'entry' : asdict(entry), 'tests': list(iterate_test()) }, f)

    # TODO: Move certain backend neutral logic here
    @abstractmethod
    def run_single_test(self,
                        input_metadata,
                        pt: Path,
                        which_impl) -> list[float]:  # L1 error
        """
        Args:
            which_impl: an ImplSelector instance.
        Returns:
            L1 error per test case.
        """
        pass

    @abstractmethod
    def run_single_benchmark(self,
                             input_metadata,
                             pt: Path,
                             which_impl) -> tuple[dict, list[float]]:
        """
        Args:
            which_impl: an ImplSelector instance.
        Returns:
            (impl_desc, times) where impl_desc is a JSON-serialisable dict
            and times is [median, p20, p80] latencies in ms.
        """
        pass

    def get_entry(self, root: Path, *, and_tests=False):
        with open(root / 'entry.json') as f:
            ej = json.load(f)
        entry = self.ENTRY_CLASS.from_dict(ej['entry'])
        if and_tests:
            return entry, ej['tests']
        else:
            return entry

    def benchmark(self, root: Path, which_impl: ImplSelector):
        """
        Output:
            entry: ENTRY_CLASS, describes an entry in tuning table
            impl_desc: json { .psels, .copts }
            adiffs: (tft, adiff, ref_error) from gpu_utils.target_fudge_factor()
            times: float[3], from do_bench(fn, quantiles=(0.5, 0.2, 0.8))
            bim: INPUT_METADATA, "benchmark_input_metadata"
        """
        entry, tests = self.get_entry(root, and_tests=True)
        def gen():
            for t in tests:
                im = self.INPUT_METADATA.from_dict(t['input_metadata'])
                pt = t['pt_file']
                yield t['test_name'], im, pt
        adiffs = {tname : self.run_single_test(im, pt, which_impl) for tname, im, pt in gen()}
        for _, bim, pt in gen():
            impl_desc, times = self.run_single_benchmark(bim, pt, which_impl)
            break
        return entry, impl_desc, adiffs, times, bim
