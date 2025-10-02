# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import torch

def cdiv(x, div):
    return (x + div - 1) // div

def round_to_8x(n):
    return 8 * cdiv(n, 8)

'''
Note: logsumexp tensor is independent of dropout
'''
def sdpa_logsumexp(query, key, value, attn_mask=None,
                   is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    m_i, _ = torch.max(attn_weight, dim=-1, keepdim=True)
    lse = m_i + torch.sum(torch.exp(attn_weight - m_i), dim=-1, keepdim=True)
    return lse.reshape((query.shape[0] * query.shape[1], query.shape[2]))
    # attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    # return attn_weight @ value

