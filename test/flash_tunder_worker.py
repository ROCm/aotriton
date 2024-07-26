#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from tuner_common import MonadAction, MonadMessage, Monad, MonadService
from tuner_worker import TunerWorker, TunerService

class FlashTunerWorker(TunerWorker):
    def service_factory():
        return FlashTunerService(self._args, self)

class FlashTunerService(TunerService):

    def create_ctx_cache(self, tup):
        '''
        Defer the import to GPU process
        '''
        import torch
        from _common_test import SdpaContext, SdpaParams
        from attn_torch_function import (
            DEFAULT_PHILOX_SEED,
            DEFAULT_PHILOX_OFFSET,
        )
        from aotriton_flash import (
            debug_fill_dropout_rng,
            hipError_t,
        )

        torch.cuda.empty_cache()
        BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, return_encoded_softmax, dtype, bias_type = tup
        philox_seed = DEFAULT_PHILOX_SEED
        philox_offset = DEFAULT_PHILOX_OFFSET
        '''
        Create reference dropout_mask
        '''
        if dropout_p > 0.0:
            rdims = (BATCH, N_HEADS, seqlen_q, seqlen_k)
            r = torch.empty(rdims, device=self._gpu_device, dtype=torch.float32)
            debug_fill_dropout_rng(r, philox_seed, philox_offset)
            mask = r > dropout_p
            torch.cuda.synchronize()
            del r
        else:
            mask = None
        ctx = SdpaContext(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype,
                          bias_type=bias_type, storage_flip=None, device=self._gpu_device)
        ctx.create_ctx_tensors()
        ctx.create_bwd_tensors()
        ctx.create_ref_inputs(target_gpu_device=self._gpu_device)
        ctx.set_require_grads(skip_db=True if bias_type == 0 else False)
        sdpa_params = SdpaParams(causal=causal, sm_scale=sm_scale, dropout_p=dropout_p, dropout_mask=mask)
        self._cached_ctx = ctx
        self._cached_params

    def profile(self, request):
        import torch
        from attn_torch_function import (
            DEFAULT_PHILOX_SEED,
            DEFAULT_PHILOX_OFFSET,
        )
        from aotriton_flash import (
            attn_fwd,
            attn_bwd,
            debug_fill_dropout_rng,
            FwdExtraArguments,
            BwdExtraArguments,
            hipError_t,
        )
        from cpp_autotune import cpp_autotune_gen

        tup = request.tup
        tid = request.task_id
        BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, return_encoded_softmax, dtype, bias_type = tup
        philox_seed = DEFAULT_PHILOX_SEED
        philox_offset = DEFAULT_PHILOX_OFFSET

        if self._arch == 'gfx1100':
            if seqlen_q > 4096 and seqlen_k > 4096:
                yield request.skip('Navi kernels triggers "MES failed to response msg=" kernel error when handling large inputs.')
                return
        if seqlen_q > 8192 and seqlen_k > 8192:
            N_HEADS = 1
        if causal and seqlen_q != seqlen_k:
            yield request.skip('FA does not support accept casual=True when seqlen_q != seqlen_k.')
            return
        if causal and bias_type != 0:
            yield request.skip('FA does not support accept casual=True when bias_type != 0.')
            return
        if a.dry_run:
            yield request.dry_run()
            return

        ctx = self.hit_cache(tup)

        q, k, v, b = ctx.dev_tensors
        # TODO: unify with attn_torch_function
        o, M = ctx.ctx_tensors

        def fwd_sub_extarg_accessor(fwd_extargs : FwdExtraArguments, i):
            return fwd_extargs

        def fwd_func(extargs, is_testing):
            # print(f'{is_testing=}')
            if is_testing:
                o.fill_(float('nan'))
            args = (q, k, v, b, sm_scale, M, o,
                    dropout_p, philox_seed, philox_offset, encoded_softmax, causal,
                    extargs.capi_object)
            try:
                ret = attn_fwd(*args)
            except Exception as e:
                self.report_exception(e)
                return 1, [KernelOutput(hip_status=hipError_t.hipErrorLaunchFailure,
                                        output_tensors=None)]
            return 1, [KernelOutput(hip_status=ret, output_tensors=[o])]

        yield from cpp_autotune_gen(FwdExtraArguments,
                                    fwd_sub_extarg_accessor,
                                    ['attn_fwd'],
                                    fwd_func,
                                    [self.fwd_validator],
                                    kernel_index_progress=request.kernel_index_progress)

        dq, dk, dv, db, delta = ctx.bwd_tensors
        dout = torch.randn_like(q)
        ctx.compute_backward(None, dout, ref_only=True)

        def bwd_func(extargs, is_testing):
            if is_testing:
                dk.fill_(float('nan'))
                dv.fill_(float('nan'))
                dq.fill_(float('nan'))
                if db is not None:
                    db.fill_(float('nan'))
            args = (q, k, v, b, sm_scale, o, do, dq, dk, dv, db, L, delta,
                    dropout_p, philox_seed, philox_offset, causal,
                    extargs.capi_object)
            try:
                ret = attn_bwd(*args)
            except Exception as e:
                self.report_exception(e)
                return hipError_t.hipErrorLaunchFailure, None
            return 2, [KernelOutput(hip_status=ret, output_tensors=[dk,dv]),
                       KernelOutput(hip_status=ret, output_tensors=[dq,db]),
                      ]
        def bwd_sub_extarg_accessor(bwd_extargs : BwdExtraArguments, i):
            if i == 0:
                return bwd_extargs.dkdv
            if i == 1:
                return bwd_extargs.dqdb
            assert False
        bwd_validators = (self.bwd_dkdv_validator, self.bwd_dqdb_validator)
        yield from cpp_autotune_gen(BwdExtraArguments, bwd_sub_extarg_accessor,
                                    ['bwd_kernel_dk_dv', 'bwd_kernel_dq'],
                                    bwd_func,
                                    bwd_validators,
                                    kernel_index_progress_dict=request.kig_dict)

    def fwd_validator(self, kernel_outputs : 'List[KernelOutput]'):
        tri_out, = kernel_outputs[0].output_tensors
        is_allclose, adiff, _, _ = self._cached_ctx.validate_with_reference(tri_out, None, no_backward=True)
        return is_allclose

    def bwd_both_validator(self, kernel_outputs : 'List[KernelOutput]'):
        tri_dk, tri_dv, = kernel_outputs[0].output_tensors
        tri_dq, tri_db, = kernel_outputs[1].output_tensors
        dout_tensors = (tri_dq, tri_dk, tri_dv, tri_db)
        _, _, grads_allclose, grads_adiff = self._cached_ctx.validate_with_reference(None, dout_tensors, no_forward=True)
        dq_allclose, dk_allclose, dv_allclose, db_allclose = grads_allclose
        ref_dq, ref_dk, ref_dv, ref_db = self._cached_ctx.dref_tensors
        return dk_allclose and dv_allclose, dq_allclose and db_allclose

    def bwd_dkdv_validator(self, kernel_outputs : 'List[KernelOutput]'):
        dkdv, dqdb = self.bwd_both_validator(kernel_outputs)
        return dkdv

    def bwd_dqdb_validator(self, kernel_outputs : 'List[KernelOutput]'):
        dkdv, dqdb = self.bwd_both_validator(kernel_outputs)
        return dqdb
