# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from aotriton.tune.tdesc import TuningDescription, ImplSelector
from aotriton.tune.utils import asdict_shallow, sanitize_value
from .entry import FlashEntry, FlashInputMetadata
from dataclasses import asdict
import dataclasses
from pathlib import Path
import gc
import sys

'''
CAVEAT about imports
FlashTune (and everything it imports at module scope: entry.py, tdesc.py,
utils.py, dataclasses/pathlib/gc) is a dual purpose class, which may
not have torch/pyaotriton/dacite packages installed in the environment.
`dacite`/`from_dict` are imported lazily inside run_single_test() /
run_single_benchmark() below, alongside their own lazy `import torch` --
those are the only two methods that need dacite, and both already require
a live GPU anyway.

dispatch_tasks.py instantiates FlashTune() for EVERY registered tuning module
at CLI startup (to build argparse subparsers from get_entry_choices()), so
this file must stay torch-free at import time and at construction time --
FlashTune() takes no arguments and does no per-level resolution up front.
Only the provider module a given DSL name actually needs (level_kernel.py /
level_op.py, both torch/pyaotriton-heavy) is imported, lazily, from inside
get_impl()/_do_probe_all_impls() below (F3's lazy-import boundary) -- never at
construction time.
'''


class FlashTune(TuningDescription):
    ENTRY_CLASS = FlashEntry
    INPUT_METADATA = FlashInputMetadata

    def _provider_for(self, name: str):
        """Resolve the DSL name `name` (e.g. 'attn_fwd' or 'op.attn_fwd') to
        (provider_module, bare_iface_name), lazily importing exactly the
        provider module the prefix calls for -- resolving an 'op.*' name must
        never import level_kernel (needs the tuning-lib-only KernelControl)
        and vice versa (F3)."""
        level, iface_name = ImplSelector.split_dsl_name(name)
        if level == 'kernel':
            from . import level_kernel as provider
        elif level == 'op':
            from . import level_op as provider
        else:
            raise ValueError(
                f"FlashTune has no tuning level {level!r} (from impl {name!r}); "
                f"expected 'kernel' (unprefixed) or 'op.'-prefixed")
        return provider, iface_name

    def list_impls(self, entry, arch: str | None = None) -> list[str]:
        """DSL-spelled names covering both levels: bare names from the kernel
        level (its the DSL's unmarked default), 'op.'-prefixed names from the
        op level. Pure entry-based enumeration -- imports level_kernel/level_op
        only for their (torch-free) list_impls() functions, same lazy-import
        boundary as get_impl()."""
        from . import level_kernel, level_op
        names = list(level_kernel.list_impls(entry, arch=arch))
        names += [f'op.{n}' for n in level_op.list_impls(entry, arch=arch)]
        return names

    def get_impl(self, name: str):
        provider, iface_name = self._provider_for(name)
        try:
            return provider.get_impl(iface_name)
        except ImportError as e:
            level, _ = ImplSelector.split_dsl_name(name)
            if level == 'kernel':
                needed, have = 'the tuning library (installed/<arch>/lib)', 'the testing library'
            else:
                needed, have = 'the testing library (installed/test/<arch>/lib)', 'the tuning library'
            raise ImportError(
                f"cannot resolve {name!r}: {level}-level tuning needs {needed}; "
                f"this process appears to have {have} loaded instead ({e})"
            ) from e

    def probe_impl_desc(self, kernel, args) -> dict:
        # Duck-type on AttnOptionsWrapperOp's distinctive `backend_index`
        # property (see level_op.py) rather than threading an explicit level
        # through this fixed 2-argument signature -- see the docstring on
        # TuningDescription.probe_impl_desc() for why this method exists
        # instead of reusing probe_all_impls()'s enumeration output.
        from . import level_kernel, level_op
        if hasattr(args, 'backend_index'):
            return level_op.impl_desc(kernel, args)
        return level_kernel.impl_desc(kernel, args)

    def _do_probe_all_impls(self, entry, im, which_impl: str, pt) -> list[dict]:
        provider, iface_name = self._provider_for(which_impl)
        return provider.enumerate_variants(entry, im, iface_name, pt)

    def get_entry_choices(self):
        return FlashEntry(
            dtype=['float16', 'bfloat16', 'float32'],
            hdim=[16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512],
            seqlen_q=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
            seqlen_k=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
            causal=[False, True],
            dropout_p=[0.0, 0.5],
            bias_type=[0, 1]
        )

    def validate_entry(self, entry: FlashEntry) -> bool:
        # Skip combinations where causal=True and bias_type != 0
        if entry.causal and entry.bias_type != 0:
            return False
        return True

    def validate_hw_feature(self, arch: str, entry: FlashEntry) -> tuple[bool, str]:
        # gfx11xx (RDNA 3, 32-lane wavesize) lacks LDS/register resources for hdim > 256.
        # The code generator also disables these combinations in _common.py; reject here
        # to avoid dispatching tuning tasks that would produce no compiled kernels.
        if arch.startswith('gfx11') and entry.hdim > 256:
            return False, (f'arch {arch} does not support hdim={entry.hdim} '
                           f'(gfx11xx maximum is 256; larger hdim exceeds LDS/register limits)')
        if arch == 'gfx1250' and entry.hdim > 256:
            return False, (f'arch {arch} does not support hdim={entry.hdim} '
                           f'(no shipped candidate is numerically accurate at hdim > 256)')
        if arch == 'gfx1250' and entry.hdim & (entry.hdim - 1) != 0:
            return False, (f'arch {arch} does not support hdim={entry.hdim} '
                           f'(NPOT head dims are disabled at compile time; see flash_disabled)')
        return True, ''

    def _gen_ref(self, entry: FlashEntry, data_root: Path, extra_ims: list = []):
        import torch
        from aotriton.tune.gpu_utils import device_ctx
        with device_ctx():
            yield from self._do_gen_ref(entry, data_root)
            for idx, im in enumerate(extra_ims):
                # Numbered after _do_gen_ref's last case (07_binning_irregular_both).
                tname = f'{8 + idx:02d}_utextra'
                yield self._write_ref_no_clamp(im, data_root, tname)

    def _clamp_memory_usage(self, im: FlashInputMetadata) -> FlashInputMetadata:
        '''
        Clamp batch size and number of heads to avoid OOM.
        Based on clamp_memory_usage from test/tune_flash.py.
        '''
        from aotriton.tune.gpu_utils import get_total_memory_from_amdsmi
        import math

        vram_cap_gb = get_total_memory_from_amdsmi()
        if vram_cap_gb is None:
            # Cannot determine VRAM, return unchanged
            return im

        # Extract values
        batch = im.BATCH if isinstance(im.BATCH, int) else 3
        is_gqa = not isinstance(im.N_HEADS, int)
        n_heads = im.N_HEADS[0] if is_gqa else im.N_HEADS
        d_head = im.hdim if isinstance(im.hdim, int) else im.hdim[0]
        seqlen_q = im.seqlen_q
        seqlen_k = im.seqlen_k
        dropout_p = im.dropout_p
        dtype = im.dtype
        bias_type = im.bias_type

        # FWD+BWD, dominated by what the REFERENCE materialises rather than by
        # the kernel under test.
        #
        # Two terms, because they scale differently and mixing them was the old
        # formula's mistake: it multiplied everything by d_head. Only q/k/v/o/do
        # and the three gradients are O(B*H*S*D); every attention-shaped tensor
        # is O(B*H*Sq*Sk) with no d_head at all, and at long sequence that is
        # essentially the whole bill. For bf16 hdim=80 8192x8192 with bias the
        # d_head-scaled tensors are 0.15 GB against 30 GB of attention-shaped
        # ones -- 0.5% -- so the old cost tracked the negligible half and
        # predicted 21.9 GB for a ~30 GB entry, which cleared a 24 GB cap and
        # clamped nothing.
        #
        # `causal` is deliberately absent: sdpa_math masks a full Sq x Sk score
        # matrix rather than skipping blocks, so causal saves no reference
        # memory. The old code bound it and never used it.
        def current_cost():
            elem = 4 if dtype == 'float32' else 2
            hp = 8 if dtype == 'float32' else 4   # reference's high-precision dtype
            # reference.py runs sdpa_math twice (input dtype and hp) plus
            # sdpa_logsumexp in hp; each retains scores+probs for backward.
            bytes_per_elem = 2 * elem + 4 * hp
            if bias_type != 0:
                bytes_per_elem += 2 * elem + 2 * hp   # bias/db, and their hp copies
            if dropout_p > 0.0:
                bytes_per_elem += elem + 1            # encoded_softmax + bool mask
            attn_gb = (bytes_per_elem * batch * n_heads * seqlen_q * seqlen_k
                       / (1024 ** 3))
            qkvo_gb = (8 * elem * batch * n_heads * max(seqlen_q, seqlen_k) * d_head
                       / (1024 ** 3))
            return 2.0 * (attn_gb + qkvo_gb)  # x2: aim to use at most 50% of VRAM

        # Halve rather than step through fixed rungs. The old ladder's first
        # three rungs were min(n_heads, 24/12/6), no-ops against the default
        # N_HEADS=5, and it bottomed out at batch=2/n_heads=2 with no way to go
        # lower however far over the cap it still was.
        #
        # GQA cannot go below 2 query heads: the map below needs a (q, k) pair,
        # and at n_heads=1 none of its branches fire, which would silently turn
        # a GQA case into a non-GQA one.
        min_heads = 2 if is_gqa else 1
        while current_cost() > vram_cap_gb and n_heads > min_heads:
            n_heads = max(min_heads, n_heads // 2)
        while current_cost() > vram_cap_gb and batch > 1:
            batch = max(1, batch // 2)
        if current_cost() > vram_cap_gb:
            # Nothing left to shrink; seqlen and hdim are the entry's identity.
            print(f'[aotriton] WARNING: {im.as_posix()} still needs about '
                  f'{current_cost() / 2.0:.1f} GB at batch={batch} '
                  f'n_heads={n_heads}, against {vram_cap_gb:.1f} GB of VRAM. '
                  f'Expect an out-of-memory failure.', file=sys.stderr)
        if is_gqa:
            if n_heads >= 24:
                n_heads = (24, 8)
            elif n_heads >= 12:
                n_heads = (12, 4)
            elif n_heads >= 6:
                n_heads = (6, 2)
            elif n_heads >= 3:
                n_heads = (3, 1)
            elif n_heads >= 2:
                n_heads = (2, 1)

        # Update im if values changed
        if batch != im.BATCH or n_heads != im.N_HEADS:
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            return dataclasses.replace(im, BATCH=batch, N_HEADS=n_heads)
        return im

    def _do_gen_ref(self, entry: FlashEntry, data_root: Path):
        '''
        Pre-condition: called with device_ctx()
        '''
        im = FlashInputMetadata(**asdict(entry))
        im = self._clamp_memory_usage(im)
        yield self._write_ref(im, data_root, '00_benchmark')

        gqa = dataclasses.replace(im, N_HEADS=(10, 2))
        gqa = self._clamp_memory_usage(gqa)
        yield self._write_ref(gqa, data_root, '01_gqa')

        ihdim = dataclasses.replace(im, hdim=im.hdim - 8)
        yield self._write_ref(ihdim, data_root, '02_irregular_hdim')

        irregular_seqlen = dataclasses.replace(im,
                                               seqlen_q=im.seqlen_q - 7,
                                               seqlen_k=im.seqlen_k - 7)
        yield self._write_ref(irregular_seqlen, data_root, '03_irregular_seqlen')

        irregular_both = dataclasses.replace(ihdim,
                                             seqlen_q=ihdim.seqlen_q - 7,
                                             seqlen_k=ihdim.seqlen_k - 7)
        yield self._write_ref(irregular_both, data_root, '04_irregular_both')

        bshd = dataclasses.replace(irregular_seqlen, storage_flip=(1,2))
        yield self._write_ref(bshd, data_root, '05_bshd')

        # Binning-boundary coverage: a seqlen past the entry's own bin exercises
        # the `le` binning clamp, which sparse tuning tables (e.g. gfx1250's
        # three-entry tech preview) lean on for everything between entries.
        # Re-clamped like 01_gqa: doubling both seqlens quadruples the attention
        # matrix, and the largest entries are already tuned at seqlen 8192.
        binning_seqlen = dataclasses.replace(im,
                                             seqlen_q=2 * im.seqlen_q,
                                             seqlen_k=2 * im.seqlen_k)
        binning_seqlen = self._clamp_memory_usage(binning_seqlen)
        yield self._write_ref(binning_seqlen, data_root, '06_binning_seqlen')

        binning_irregular_both = dataclasses.replace(im,
                                                     seqlen_q=2 * im.seqlen_q - 37,
                                                     seqlen_k=2 * im.seqlen_k - 37,
                                                     hdim=im.hdim - 8)
        binning_irregular_both = self._clamp_memory_usage(binning_irregular_both)
        yield self._write_ref(binning_irregular_both, data_root, '07_binning_irregular_both')
        # TODO: varlen tests

    def _write_ref(self,
                   im: FlashInputMetadata,
                   root: Path,
                   tname: str):
        '''
        Pre-condition: called with device_ctx()
        '''
        import torch
        if im.qkh > 2048 * 2048 * 64:
            gc.collect()
            torch.cuda.empty_cache()
        # print(f'{tname=} {im=}')
        from .reference import SdpaReference
        ref_kernel = SdpaReference()
        bidi_inputs = ref_kernel.generate_inputs(im)
        bidi_inputs, outputs = ref_kernel(im, bidi_inputs, None)
        d = {
            "bidi_inputs" : asdict_shallow(bidi_inputs),
            "bidi_outputs" : asdict_shallow(outputs),
        }
        pt = (root / tname).with_suffix('.pt')
        torch.save(d, pt)
        return tname, im, pt

    def _write_ref_no_clamp(self,
                            im: FlashInputMetadata,
                            root: Path,
                            tname: str):
        '''Like _write_ref but skips _clamp_memory_usage — extra IMs come from real
        pytest runs so their shapes are known to fit in VRAM.
        Pre-condition: called with device_ctx()
        '''
        import torch
        if im.qkh > 2048 * 2048 * 64:
            gc.collect()
            torch.cuda.empty_cache()
        from .reference import SdpaReference
        ref_kernel = SdpaReference()
        bidi_inputs = ref_kernel.generate_inputs(im)
        bidi_inputs, outputs = ref_kernel(im, bidi_inputs, None)
        d = {
            "bidi_inputs" : asdict_shallow(bidi_inputs),
            "bidi_outputs" : asdict_shallow(outputs),
        }
        pt = (root / tname).with_suffix('.pt')
        torch.save(d, pt)
        return tname, im, pt

    def run_single_test(self,
                        im: FlashInputMetadata,
                        pt: Path,
                        which_impl):
        import torch
        from dacite import from_dict
        from aotriton.tune.utils import dacite_tuple
        from aotriton.tune.gpu_utils import device_ctx, default_device_string
        with device_ctx():
            kernel = self.get_impl(which_impl.dsl_name)
            args = kernel.create_extargs(which_impl=which_impl, dtype=im.dtype)
            d = torch.load(pt, map_location=default_device_string(), mmap=True)
            inputs = from_dict(data_class=kernel.PT_INPUT_CLASS, data=d["bidi_inputs"], config=dacite_tuple)
            direct_inputs = kernel.prepare_directs(im, inputs)
            kernel.fill_nan_to_outputs(direct_inputs)
            outputs, err = kernel.direct_call(direct_inputs, args)
            refs = from_dict(data_class=kernel.PT_REF_CLASS, data=d["bidi_outputs"], config=dacite_tuple)
            result = kernel.compare(outputs, refs)
            early = kernel.check_early_reject_results(result, err)
            if early is not None:
                result = early
            if im.qkh > 2048 * 2048 * 64:
                gc.collect()
                torch.cuda.empty_cache()
            return sanitize_value(result)

    def run_single_benchmark(self,
                             im: FlashInputMetadata,
                             pt: Path,
                             which_impl):
        import torch
        from dacite import from_dict
        from aotriton.tune.utils import dacite_tuple
        from aotriton.tune.gpu_utils import do_bench, device_ctx, default_device_string
        with device_ctx():
            kernel = self.get_impl(which_impl.dsl_name)
            args = kernel.create_extargs(which_impl=which_impl, probe=True, dtype=im.dtype)
            d = torch.load(pt, map_location=default_device_string(), mmap=True)
            inputs = from_dict(data_class=kernel.PT_INPUT_CLASS, data=d["bidi_inputs"], config=dacite_tuple)
            direct_inputs = kernel.prepare_directs(im, inputs)
            kernel.direct_call(direct_inputs, args)
            impl_desc = self.probe_impl_desc(kernel, args)
            args.disable_probing()
            def fn():
                kernel.direct_call(direct_inputs, args)
            times = do_bench(fn, quantiles=(0.5, 0.2, 0.8))
            if im.qkh > 2048 * 2048 * 64:
                gc.collect()
                torch.cuda.empty_cache()
            return sanitize_value(impl_desc), sanitize_value(times)
