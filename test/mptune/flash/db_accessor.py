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

class DbMonad(Monad):
    def service_factory():
        return DbAccessor(self._args, self)

class DbAccessor(BaseDbService):
    KERNEL_FAMILY = 'FLASH'

    def constrcut_inputs(self, request):
        tup = request.tup
        BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, return_encoded_softmax, dtype, bias_type = tup
        head_dim_rounded = 2 ** (D_HEAD - 1).bit_length()
        head_dim_rounded = max(16, head_dim_rounded)
        inputs = {
            'Q_dtype': str(dtype),
            'N_HEADS': N_HEADS,
            'D_HEAD': D_HEAD,
            'max_seqlen_q': seqlen_q,
            'max_seqlen_k': seqlen_k,
            'CAUSAL': causal,
            'RETURN_ENCODED_SOFTMAX': return_encoded_softmax,
            'BLOCK_DMODEL': head_dim_rounded,
            'ENABLE_DROPOUT' : dropout_p > 0.0,
            'PADDED_HEAD' : head_dim_rounded != D_HEAD,
            'BIAS_TYPE' : bias_type,
        }
        return inputs

    def analysis_result(self, request):
        return request.perf_number

