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
    def service_factory(self):
        return DbAccessor(self._args, self)

class DbAccessor(BaseDbService):
    KERNEL_FAMILY = 'FLASH'

    def constrcut_inputs(self, request):
        payload = request.payload
        tup = payload.tup
        BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, return_encoded_softmax, dtype, bias_type = tup
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
        }
        return inputs

    def analysis_result(self, request):
        return request.payload.perf_number
