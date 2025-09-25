# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import torch

def cdiv(x, div):
    return (x + div - 1) // div

def round_to_8x(n):
    return 8 * cdiv(n, 8)

def elike(t: torch.Tensor | None) -> torch.Tensor | None:
    return torch.empty_like(t) if t is not None else None

def adiff(golden: torch.Tensor | None,
          lowp: torch.Tensor | None) -> float | None:
    if golden is None or lowp is None:
        assert lowp is None
        return None
    return (golden, torch.max(torch.abs(golden - lowp)).item())

def sdpa_logsumexp(query, key, value, attn_mask=None, dropout_p=0.0,
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
    m_i = torch.max(attn_weight, dim=-1)
	return m_i + torch.sum(torch.exp(attn_weight - m_i), dim=-1, keep_dims=True)
    # attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    # return attn_weight @ value

