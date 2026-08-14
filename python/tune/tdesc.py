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
    Subclasses of TuningLevel (see below) MUST NOT import kernel modules or
    instantiate kernel objects at module level or in __init__. Kernel modules
    typically import torch (via reference.py), which is unavailable outside
    GPU containers. dispatch_tasks.py imports every TuneDesc subclass at
    startup (to build argparse subparsers), so a top-level torch import
    breaks dispatch on machines without torch.

    Always initialize kernel objects lazily inside get_impl() on first call:

        class MyLevel(TuningLevel):
            _kernel_dict = None

            def get_impl(self, name):
                if self._kernel_dict is None:
                    from .kernels import KernelA, KernelB  # lazy import
                    self._kernel_dict = {'a': KernelA(), 'b': KernelB()}
                return self._kernel_dict[name]
'''


@dataclass
class ImplSelector:
    """Unified impl-selector DSL (modular-tune.md §4.2):
    `[<tuning_level> '.'] <iface_name> '=' <impl_index>`.

    Replaces the old, per-module FlashKernelSelector ('kernel_name=hsaco_index')
    and FlashOpBackendSelector ('op_name=backend_index') pair with a single
    three-field dataclass shared by every tuning level of every family.
    `tuning_level` defaults to 'kernel' and its prefix is omitted by
    as_text() in that case, so plain kernel-level selector text is unchanged
    from before ('attn_fwd=3'); op-level selectors are prefixed
    ('op.attn_fwd=1').

    IMPORTANT: the `op.` (or any non-kernel level's) prefix is surface syntax
    ONLY -- it must never reach storage. `TuningLevel.list_impls()`/`get_impl()`
    speak bare interface names; `iface_name` here is always bare too (never
    e.g. 'attn_fwd_op').
    """
    tuning_level: str = 'kernel'
    iface_name: str = ''
    impl_index: int = -1

    @staticmethod
    def parse_text(line: str) -> "ImplSelector":
        lhs, index = line.split('=')
        level, _, iface = lhs.rpartition('.')
        return ImplSelector(tuning_level=level or 'kernel', iface_name=iface, impl_index=int(index))

    def as_text(self) -> str:
        prefix = '' if self.tuning_level == 'kernel' else f'{self.tuning_level}.'
        return f'{prefix}{self.iface_name}={self.impl_index}'


class TuningLevel(ABC):
    """Strategy object encapsulating what differs between tuning levels of the
    same family (modular-tune.md §4.1) -- e.g. flash's kernel-level (HSACO
    variant selection, needs the tuning-instrumented pyaotriton build) vs.
    op-level (backend selection, needs only the plain testing build):

      - list_impls / get_impl: which bare interface names exist, and how to
        resolve one to its impl object.
      - enumerate_variants (was TuningDescription._do_probe_backends): how to
        enumerate the candidate implementation variants for one interface.
      - impl_desc (was TuningDescription.probe_impl_desc): how to describe
        the variant actually selected by a probing run.

    Subclasses MUST NOT import kernel modules (torch/pyaotriton) at module
    level or in __init__ -- always lazily inside get_impl() on first call
    (see the module docstring above).
    """

    NAME: str = ''
    # Whether this level needs the *tuning*-instrumented pyaotriton library
    # build (KernelControl / kernel_fine_control), as opposed to the plain
    # testing library build. TuningDescription checks this up front, at
    # construction time (§4.5), so a level/library mismatch fails fast
    # instead of surfacing as a deep ImportError on the first get_impl() call.
    REQUIRES_TUNING_LIB: bool = True

    def __init__(self, tune: "TuningDescription"):
        self.tune = tune

    @abstractmethod
    def list_impls(self, entry, arch: str | None = None) -> list[str]:
        """Bare interface names only -- e.g. ['attn_fwd', 'bwd_kernel_dk_dv',
        'bwd_kernel_dq', 'bwd_kernel_fuse'] for flash's kernel level,
        ['attn_fwd', 'attn_bwd'] for flash's op level. The `op.` prefix used
        in the ImplSelector DSL is surface syntax; it must never appear in a
        name returned here (highest-risk area #1 of modular-tune.md)."""
        pass

    @abstractmethod
    def get_impl(self, name: str):
        """Return the impl object for a bare interface name. Callers never
        pass an ImplSelector here -- TuningDescription.get_impl() unwraps
        that before delegating."""
        pass

    @abstractmethod
    def enumerate_variants(self, entry, im, which_impl: str, pt: Path) -> list[dict]:
        """Was TuningDescription._do_probe_backends. Enumerate the candidate
        implementation variants (HSACO indices for a kernel level, backend
        indices for an op level) for `which_impl`."""
        pass

    @abstractmethod
    def impl_desc(self, kernel, args) -> dict:
        """Was TuningDescription.probe_impl_desc. Extract impl_desc from a
        probing run's extargs (e.g. {psels, copts} for HSACO kernels,
        {backend_index} for op backends)."""
        pass


class TuningDescription(ABC):
    # tuning_level name -> TuningLevel subclass. Subclasses (one per family)
    # must populate this, e.g. {'kernel': FlashKernelLevel, 'op': FlashOpLevel}.
    LEVELS: dict[str, type[TuningLevel]] = {}

    def __init__(self, level: str = 'kernel'):
        try:
            level_cls = self.LEVELS[level]
        except KeyError:
            raise ValueError(
                f"Unknown tuning level {level!r} for {type(self).__name__}; "
                f"available: {sorted(self.LEVELS)}")
        self.tuning_level = level
        self.level = level_cls(self)

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

    def list_impls(self, entry, arch: str | None = None) -> list[str]:
        return self.level.list_impls(entry, arch=arch)

    def get_impl(self, name: str | ImplSelector):
        """Return the impl object for the given name or selector.
        Accepts either a plain str name or an ImplSelector instance --
        iface_name is extracted from the selector before delegating to
        self.level (TuningLevel implementations only ever see bare names).
        MUST use lazy initialization (import torch-dependent modules inside
        this method, not at module level) -- see self.level's implementation.
        """
        if isinstance(name, ImplSelector):
            name = name.iface_name
        return self.level.get_impl(name)

    def probe_impl_desc(self, kernel, args) -> dict:
        """Extract impl_desc from a probing run's extargs.

        Called by run_single_benchmark after kernel.direct_call(direct_inputs, args)
        with probe=True. Returns a JSON-serialisable dict that uniquely identifies
        the chosen implementation (e.g., {psels, copts} for HSACO kernels,
        {backend_index} for op backends).

        Args:
            kernel: the impl object returned by get_impl()
            args: the extargs object returned by create_extargs(probe=True)
        """
        return self.level.impl_desc(kernel, args)

    def probe_backends(self, root: Path, which_impl: str) -> list[dict]:
        entry, tests = self.get_entry(root, and_tests=True)
        test = tests[0]
        im = self.INPUT_METADATA.from_dict(test["input_metadata"])
        pt = Path(test["pt_file"])
        return self._do_probe_backends(entry, im, which_impl, pt)

    def _do_probe_backends(self, entry, im, which_impl: str, pt: Path) -> list[dict]:
        return self.level.enumerate_variants(entry, im, which_impl, pt)

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
