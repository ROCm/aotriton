#!/usr/bin/env python
# Copyright © 2024-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import math
from ..core import (
    MonadAction,
    MonadMessage,
    Monad,
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

class BenchmarkMonad(Monad):
    def service_factory(self):
        return BenchmarkService(self._args, self)

class BenchmarkService(BaseTunerService):

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
        BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, return_encoded_softmax, dtype, bias_type, op_backend = tup
        dtype = getattr(torch, dtype)
        if CPPTUNE_FLASH_DEFENSIVE_SEQLENS:
            seqlen_q -= 7
            seqlen_k -= 7
        '''
        Create reference dropout_mask
        '''
        if dropout_p > 0.0:
            rdims = (BATCH, N_HEADS, seqlen_q, seqlen_k)
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
        from ..core import cpp_autotune_gen, KernelOutput, AutotuneResult, do_bench

        payload = request.payload
        tup = payload.tup
        tid = request.task_id
        # print(tup)
        BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, return_encoded_softmax, dtype, bias_type, op_backend = tup
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

        # ref_out is kept in the ctx
        _ = ctx.compute_ref_forward(sdpa_params)
        def fwd_func(extargs : 'CppTuneWrapper', is_testing):
            # Faulty kernel may rewrite any tensor
            q, k, v, b = ctx.dev_tensors
            o, M = ctx.ctx_tensors
            L = M  # alias
            if is_testing:
                o.fill_(float('nan'))
                M.fill_(float('nan'))
                philox_seed_output.fill_(0)
                philox_offset_output.fill_(0)
            if causal:
                atomic = torch.zeros([1], device=q.device, dtype=torch.int32)
            else:
                atomic = torch.empty([0], device=q.device, dtype=torch.int32)
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
            return 1, [KernelOutput(hip_status=ret,
                                    output_tensors=[o, philox_seed_output, philox_offset_output])]
        fwd_func(None, is_testing=True)

        ctx.compute_backward(None, None, ref_only=True)
        dout = ctx.ddev_tensors[0]

        from aotriton_flash import (
            attn_bwd,
            attn_options,
            lazy_dq_acc,
        )
        def generic_bwd_func(extargs, is_testing, bwd_operator):
            q, k, v, b = ctx.dev_tensors
            dq, dk, dv, db, delta = ctx.bwd_tensors
            dq_acc = lazy_dq_acc(dq)
            o, M = ctx.ctx_tensors
            L = M  # alias
            if is_testing:
                dk.fill_(float('nan'))
                dv.fill_(float('nan'))
                dq.fill_(float('nan'))
                if db is not None:
                    db.fill_(float('nan'))
            CALL_OPERATOR=True
            args = (q, k, v, b, sm_scale, o, dout, dq, dk, dv, db, dq_acc, L, delta,
                    dropout_p, philox_seed_output, philox_offset_output, 0,
                    causal, extargs, CALL_OPERATOR)
            try:
                ret = bwd_operator(*args)
            except Exception as e:
                self.report_exception(e)
                ret = hipError_t.hipErrorLaunchFailure
                return 1, [KernelOutput(hip_status=ret, output_tensors=None)]
            return 1, [KernelOutput(hip_status=ret, output_tensors=[dk,dv,dq,db])]

        atr = AutotuneResult()
        atr.kernel_index = op_backend
        atr.total_number_of_kernels = 3
        extargs = attn_options()
        extargs.force_backend_index = op_backend
        def func(is_testing=False):
            return generic_bwd_func(extargs, is_testing, bwd_operator=attn_bwd)
        atr = do_bench(func, atr, rep=250, validator=self.bwd_fused_validator)
        yield op_backend, atr, None

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
