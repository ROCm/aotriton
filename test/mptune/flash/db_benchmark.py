#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..core import (
    MonadAction,
    MonadMessage,
    Monad,
    MonadService,
    DbService as BaseDbService,
)
import numpy as np

class DbBenchmarkMonad(Monad):
    def service_factory(self):
        return DbBenchmarkAccessor(self._args, self)

class DbBenchmarkAccessor(BaseDbService):
    KERNEL_FAMILY = 'FLASH'

    def constrcut_inputs(self, request):
        payload = request.payload
        tup = payload.tup
        BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, return_encoded_softmax, dtype, bias_type, op_backend = tup
        # TODO: Use proper solution
        # Duct taped solution for tuning database
        if D_HEAD not in [16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256]:
            head_dim_rounded = 2 ** (D_HEAD - 1).bit_length()
            head_dim_rounded = max(16, head_dim_rounded)
        else:
            head_dim_rounded = D_HEAD
        inputs = {
            'Q_dtype': str(dtype),
            'BATCH' : BATCH,
            'N_HEADS': N_HEADS,
            'D_HEAD': D_HEAD,
            'Max_seqlen_q': seqlen_q,
            'Max_seqlen_k': seqlen_k,
            'CAUSAL_TYPE': causal,
            'RETURN_ENCODED_SOFTMAX': return_encoded_softmax,
            'BLOCK_DMODEL': head_dim_rounded,
            'ENABLE_DROPOUT' : dropout_p > 0.0,
            'PADDED_HEAD' : head_dim_rounded != D_HEAD,
            'BIAS_TYPE'     : bias_type,
            'USE_ALIBI'     : False,
            'INT8'          : False,
            'INT8_KV'       : False,
            'USE_P_SCALE'   : False,
            'op_backend'    : op_backend,
        }
        return inputs

    def analysis_result(self, request):
        return request.payload.perf_number

    def translate_config(self, kernel_name, inputs, atr : 'AutotuneResult'):
        tuning_result = super().translate_config(kernel_name, inputs, atr)
        BATCH = inputs['BATCH']
        H = inputs['N_HEADS']
        N_CTX_Q = inputs['Max_seqlen_q']
        N_CTX_K = inputs['Max_seqlen_k']
        D_HEAD = inputs['D_HEAD']
        flops_per_matmul = 2. * BATCH * H * N_CTX_Q * N_CTX_K * D_HEAD
        total_flops = 2 * flops_per_matmul
        ms = atr.time
        tuning_result['TFLOPS'] = total_flops / ms * 1e-9
        return tuning_result
