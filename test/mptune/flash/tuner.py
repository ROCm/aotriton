#!/usr/bin/env python
# Copyright © 2024-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import math
from ..core import (
    MonadAction,
    MonadMessage,
    Monad,
    MonadService,
    TunerService as BaseTunerService,
    ProfilerEarlyExit as PEE,
)
DEFAULT_PHILOX_SEED = 0x1BF52
DEFAULT_PHILOX_OFFSET_1 = 0x1D4000
DEFAULT_PHILOX_OFFSET_2 = 0x000B42
DEFAULT_PHILOX_OFFSET = DEFAULT_PHILOX_OFFSET_1 + DEFAULT_PHILOX_OFFSET_2
INTEGRITY_CHECK_PER_HSACO = bool(int(os.getenv('INTEGRITY_CHECK_PER_HSACO', default='0')))
# GFX950's compiler has problem handling irregulars. Must be tested with irregulars
CPPTUNE_FLASH_DEFENSIVE_SEQLENS = bool(int(os.getenv('CPPTUNE_FLASH_DEFENSIVE_SEQLENS', default='0')))

class TunerMonad(Monad):
    def service_factory(self):
        return TunerService(self._args, self)

class FakeExtArgs:
    def __init__(self, factory):
        self.capi_object = factory()

class TunerService(BaseTunerService):

    def _parse_skip(self, tup):
        BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, return_encoded_softmax, dtype, bias_type = tup
        from ..core import CPPTUNE_SKIP_KERNELS
        skip_fwd = 'attn_fwd' in CPPTUNE_SKIP_KERNELS
        if 'bwd_kernel_dk_dv' in CPPTUNE_SKIP_KERNELS and 'bwd_kernel_dq' in CPPTUNE_SKIP_KERNELS:
            skip_split_bwd = True
        else:
            skip_split_bwd = False

        if 'bwd_kernel_fuse' in CPPTUNE_SKIP_KERNELS:
            skip_fused_bwd = True
        else:
            skip_fused_bwd = False

        if seqlen_q < 16 or seqlen_q > 1024:
            skip_fused_bwd = True
        if seqlen_k < 16 or seqlen_k > 1024:
            skip_fused_bwd = True

        skip_bwd = skip_split_bwd and skip_fused_bwd
        return skip_fwd, skip_bwd, skip_split_bwd, skip_fused_bwd

    def create_ctx_cache(self, tup):
        '''
        Defer the import to GPU process
        '''
        import torch
        from _common_test import SdpaContext, SdpaParams
        from aotriton_flash import (
            debug_simulate_encoded_softmax,
            hipError_t,
        )

        torch.cuda.empty_cache()
        BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, return_encoded_softmax, dtype, bias_type = tup
        dtype = getattr(torch, dtype)
        if isinstance(N_HEADS, int):
            Q_HEADS = K_HEADS = N_HEADS
        else:
            Q_HEADS, K_HEADS = N_HEADS
        if CPPTUNE_FLASH_DEFENSIVE_SEQLENS:
            seqlen_q -= 7
            seqlen_k -= 7
        skip_fwd, skip_bwd, skip_split_bwd, skip_fused_bwd = self._parse_skip(tup)
        '''
        Create reference dropout_mask
        '''
        if dropout_p > 0.0:
            rdims = (BATCH, Q_HEADS, seqlen_q, seqlen_k)
            r = torch.empty(rdims, device=self._gpu_device, dtype=torch.float32)
            philox_seed = torch.tensor([DEFAULT_PHILOX_SEED], device=r.device, dtype=torch.uint64)
            philox_offset1 = torch.tensor([DEFAULT_PHILOX_OFFSET_1], device=r.device, dtype=torch.uint64)
            philox_offset2 = DEFAULT_PHILOX_OFFSET_2
            debug_simulate_encoded_softmax(r, dropout_p, philox_seed, philox_offset1, philox_offset2)
            mask = r >= 0
            torch.cuda.synchronize()
            del r
        else:
            mask = None
        sdpa_params = SdpaParams(causal=causal, sm_scale=sm_scale, dropout_p=dropout_p, dropout_mask=mask)
        ctx = SdpaContext(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype,
                          bias_type=bias_type, storage_flip=None, device=self._gpu_device)
        ## For reproducible values
        ctx.create_ctx_tensors()
        if not skip_bwd:
            ctx.create_bwd_tensors()
        ctx.create_ref_inputs(target_gpu_device=self._gpu_device)
        ctx.set_require_grads(skip_db=True if bias_type == 0 else False)
        self._cached_ctx = ctx
        self._cached_params = sdpa_params

    def profile(self, request):
        a = self._args
        import torch
        from aotriton_flash import IGNORE_BACKWARD_IMPORT
        from aotriton_flash import (
            attn_fwd,
            FwdExtraArguments,
            hipError_t,
        )
        from ..core import cpp_autotune_gen, KernelOutput, AutotuneResult

        payload = request.payload
        tup = payload.tup
        tid = request.task_id
        BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, return_encoded_softmax, dtype, bias_type = tup
        encoded_softmax = None
        dtype = getattr(torch, dtype)

        if causal and bias_type != 0:
            raise PEE(request.make_skip(self.monad, 'FA does not support accept casual=True when bias_type != 0.'))
        if a.dry_run:
            raise PEE(request.make_dryrun(self.monad))

        ctx, sdpa_params = self.hit_cache(tup)

        q = ctx.dev_tensors[0]

        if dropout_p > 0.0:
            philox_seed = torch.tensor([DEFAULT_PHILOX_SEED], device=q.device, dtype=torch.uint64)
            philox_offset1 = torch.tensor([DEFAULT_PHILOX_OFFSET_1], device=q.device, dtype=torch.uint64)
            philox_offset2 = DEFAULT_PHILOX_OFFSET_2
            philox_seed_output = torch.tensor([0], device=q.device, dtype=torch.uint64)
            philox_offset_output = torch.tensor([0], device=q.device, dtype=torch.uint64)
        else:
            nulltensor = torch.empty([0], device=q.device, dtype=torch.uint64)
            philox_seed = nulltensor
            philox_offset1 = nulltensor
            philox_offset2 = 0
            philox_seed_output = nulltensor
            philox_offset_output = nulltensor

        def subless_sub_extarg_accessor(extargs, i):
            return extargs

        skip_fwd, skip_bwd, skip_split_bwd, skip_fused_bwd = self._parse_skip(tup)

        # ref_out is kept in the ctx
        _ = ctx.compute_ref_forward(sdpa_params)
        ctx.save_integrity_checksum()

        def integrity_check_and_restore():
            integrity, who = ctx.check_integrity()
            if not integrity:
                ctx.restore_integrity(who, sdpa_params)
            return integrity
        def fwd_func(extargs : 'CppTuneWrapper', is_testing):
            # Faulty kernel may rewrite any tensor
            q, k, v, b = ctx.dev_tensors
            o, M = ctx.ctx_tensors
            L = M  # alias
            if causal:
                atomic = torch.zeros([1], device=q.device, dtype=torch.int32)
            else:
                atomic = torch.empty([0], device=q.device, dtype=torch.int32)
            if is_testing:
                o.fill_(float('nan'))
                M.fill_(float('nan'))
                philox_seed_output.fill_(0)
                philox_offset_output.fill_(0)
                # print(f'{atomic=}')
            # print(f'Calling fwd_func with {extargs.force_kernel_index=}')
            args = (q, k, v, b, sm_scale, M, o,
                    dropout_p, philox_seed, philox_offset1, philox_offset2,
                    philox_seed_output, philox_offset_output,
                    encoded_softmax, causal, atomic,
                    extargs.capi_object if extargs is not None else None)
            try:
                ret = attn_fwd(*args)
            except Exception as e:
                self.report_exception(e)
                return 1, [KernelOutput(hip_status=hipError_t.hipErrorLaunchFailure,
                                        output_tensors=None)]
            if INTEGRITY_CHECK_PER_HSACO and is_testing:
                integrity, who = ctx.check_integrity()
                if not integrity:
                    ret = hipError_t.hipErrorDeinitialized
                    ctx.restore_integrity(who, sdpa_params)
            return 1, [KernelOutput(hip_status=ret,
                                    output_tensors=[o, philox_seed_output, philox_offset_output])]
        # print(f'{payload.kig_dict=}')
        if not skip_fwd:
            # if skip_fwd, run manually to use default tuning db
            yield from cpp_autotune_gen(FwdExtraArguments,
                                        subless_sub_extarg_accessor,
                                        ['attn_fwd'],
                                        fwd_func,
                                        [self.fwd_validator],
                                        kernel_index_progress_dict=payload.kig_dict,
                                        integrity_checker=integrity_check_and_restore)
        # Early exit when both bwd are disabled
        # Skipping of individual kernels is handled in cpp_autotune_gen directly
        if skip_bwd:
            return
        integrity, who = ctx.check_integrity()
        if not integrity:
            print('ctx.restore_integrity')
            ctx.restore_integrity(who, sdpa_params)
        extargs = FakeExtArgs(FwdExtraArguments)
        extargs.capi_object.force_kernel_index = payload.kig_dict['attn_fwd'].last_success_kernel
        fwd_func(extargs, is_testing=True)

        ctx.compute_backward(None, None, ref_only=True)
        dout = ctx.ddev_tensors[0]
        ctx.save_integrity_checksum()

        bwd_validators = (self.bwd_dkdv_validator, self.bwd_dqdb_validator)
        def generic_bwd_func(extargs, is_testing, bwd_operator, split):
            q, k, v, b = ctx.dev_tensors
            dq, dk, dv, db, delta = ctx.bwd_tensors
            o, M = ctx.ctx_tensors
            L = M  # alias
            if is_testing:
                dk.fill_(float('nan'))
                dv.fill_(float('nan'))
                dq.fill_(float('nan'))
                if db is not None:
                    db.fill_(float('nan'))
            if split:
                args = (q, k, v, b, sm_scale, o, dout, dq, dk, dv, db, None, L, delta,
                        dropout_p, philox_seed_output, philox_offset_output, 0,
                        causal, extargs.capi_object)
            else:
                args = (q, k, v, b, sm_scale, o, dout, dq, dk, dv, db, L,
                        dropout_p, philox_seed_output, philox_offset_output, 0,
                        causal, extargs.capi_object)
            try:
                ret = bwd_operator(*args)
            except Exception as e:
                self.report_exception(e)
                ret = hipError_t.hipErrorLaunchFailure
                if split:
                    return 2, [KernelOutput(hip_status=ret, output_tensors=None),
                               KernelOutput(hip_status=ret, output_tensors=None),
                              ]
                else:
                    return 1, [KernelOutput(hip_status=ret, output_tensors=None)]
            if INTEGRITY_CHECK_PER_HSACO and is_testing:
                integrity, who = ctx.check_integrity()
                if not integrity:
                    ret = hipError_t.hipErrorDeinitialized
                    ctx.restore_integrity(who, sdpa_params)
            if split:
                return 2, [KernelOutput(hip_status=ret, output_tensors=[dk,dv]),
                           KernelOutput(hip_status=ret, output_tensors=[dq,db]),
                          ]
            else:
                return 1, [KernelOutput(hip_status=ret, output_tensors=[dk,dv,dq,db])]
        def bwd_sub_extarg_accessor(bwd_extargs : 'BwdExtraArguments', i):
            if i == 0:
                return bwd_extargs.dkdv
            if i == 1:
                return bwd_extargs.dqdb
            assert False

        if not skip_split_bwd:
            from aotriton_flash import (
                attn_bwd,
                BwdExtraArguments,
            )
            def bwd_func(extargs, is_testing):
                return generic_bwd_func(extargs, is_testing, attn_bwd, split=True)
            yield from cpp_autotune_gen(BwdExtraArguments, bwd_sub_extarg_accessor,
                                        ['bwd_kernel_dk_dv', 'bwd_kernel_dq'],
                                        bwd_func,
                                        bwd_validators,
                                        kernel_index_progress_dict=payload.kig_dict,
                                        integrity_checker=integrity_check_and_restore)
        if not skip_fused_bwd:
            from aotriton_flash import (
                attn_bwd_fused,
                FusedBwdExtraArguments,
            )
            def bwd_func(extargs, is_testing):
                return generic_bwd_func(extargs, is_testing, attn_bwd_fused, split=False)
            yield from cpp_autotune_gen(FusedBwdExtraArguments,
                                        subless_sub_extarg_accessor,
                                        ['bwd_kernel_fuse'],
                                        bwd_func,
                                        [self.bwd_fused_validator],
                                        kernel_index_progress_dict=payload.kig_dict,
                                        integrity_checker=integrity_check_and_restore)

    def fwd_validator(self, kernel_outputs : 'List[KernelOutput]', atr : 'AutotuneResult'):
        tri_out, philox_seed, philox_offset = kernel_outputs[0].output_tensors
        is_allclose, adiffs, _, _, tft = self._cached_ctx.validate_with_reference(tri_out, None, no_backward=True,
                                                                                  return_target_fudge_factors=True)
        # Note: philox_seed/offset_output is only updated when causal=True. We
        #       need a better closure solution for validator
        # TODO: Check philox
        # philox_correct = (philox_seed == DEFAULT_PHILOX_SEED) and (philox_offset == DEFAULT_PHILOX_OFFSET)
        atr.ut_passed = is_allclose
        atr.adiffs = adiffs
        atr.target_fudge_factors = {'out' : tft['out']}
        return atr

    def bwd_fused_validator(self, kernel_outputs : 'List[KernelOutput]', atr : 'AutotuneResult'):
        tri_dk, tri_dv, tri_dq, tri_db = kernel_outputs[0].output_tensors
        dout_tensors = (tri_dq, tri_dk, tri_dv, tri_db)
        _, _, grads_allclose, grads_adiff, tft = self._cached_ctx.validate_with_reference(None, dout_tensors,
                                                                                          no_forward=True,
                                                                                          return_target_fudge_factors=True)
        dq_allclose, dk_allclose, dv_allclose, db_allclose = grads_allclose
        dq_adiff, dk_adiff, dv_adiff, db_adiff = grads_adiff
        atr.ut_passed = dq_allclose and dk_allclose and dv_allclose and db_allclose
        atr.adiffs = [dk_adiff, dv_adiff, dq_adiff, db_adiff]
        atr.target_fudge_factors = {'dk' : tft['k'], 'dv' : tft['v'], 'dq' : tft['q'], 'db' : tft['b']}
        return atr

    def bwd_both_validator(self, kernel_outputs : 'List[KernelOutput]'):
        tri_dk, tri_dv, = kernel_outputs[0].output_tensors
        tri_dq, tri_db, = kernel_outputs[1].output_tensors
        dout_tensors = (tri_dq, tri_dk, tri_dv, tri_db)
        _, _, grads_allclose, grads_adiff, tft = self._cached_ctx.validate_with_reference(None, dout_tensors,
                                                                                          no_forward=True,
                                                                                          return_target_fudge_factors=True)
        dq_allclose, dk_allclose, dv_allclose, db_allclose = grads_allclose
        ref_dq, ref_dk, ref_dv, ref_db = self._cached_ctx.dref_tensors
        return dk_allclose and dv_allclose, dq_allclose and db_allclose, grads_adiff, tft

    def bwd_dkdv_validator(self, kernel_outputs : 'List[KernelOutput]', atr : 'AutotuneResult'):
        dkdv, dqdb, grads_adiff, tft = self.bwd_both_validator(kernel_outputs)
        dq_adiff, dk_adiff, dv_adiff, db_adiff = grads_adiff
        # if not dkdv:
        #     print(f'{grads_adiff=} {tft=}')
        atr.ut_passed = dkdv
        atr.adiffs = [dk_adiff, dv_adiff]
        atr.target_fudge_factors = {'dk' : tft['k'], 'dv' : tft['v']}
        return atr

    def bwd_dqdb_validator(self, kernel_outputs : 'List[KernelOutput]', atr : 'AutotuneResult'):
        dkdv, dqdb, grads_adiff, tft = self.bwd_both_validator(kernel_outputs)
        dq_adiff, dk_adiff, dv_adiff, db_adiff = grads_adiff
        # if not dqdb:
        #     print(f'{grads_adiff=}')
        atr.ut_passed = dqdb
        atr.adiffs = [dq_adiff, db_adiff]
        atr.target_fudge_factors = {'dq' : tft['q'], 'db' : tft['b']}
        return atr
