import os
import numpy as np
import torch
import math
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from attn_torch_function import attention, AttentionExtraArgs

def IENV(name, default):
    val = bool(int(os.getenv(name, default=str(default))))
    globals()[name] = val

IENV('USE_FP16', 1)
IENV('USE_MATH', 0)
IENV('CAST_FP32', 0)
IENV('USE_FULL_TENSOR', 0)

def dup(t, dtype=None):
    return torch.tensor(t, dtype=t.dtype if dtype is None else dtype, device=t.device, requires_grad=t.requires_grad)

def argmax(t):
    return np.unravel_index(t.argmax().cpu().numpy(), t.shape)

print(f"{USE_FP16=}")
print(f"{USE_MATH=}")
print(f"{CAST_FP32=}")
print(f"{USE_FULL_TENSOR=}")

if USE_FP16:
    query = torch.load('query_fp16.tensor', weights_only=True)
    key = torch.load('key_fp16.tensor', weights_only=True)
    value = torch.load('value_fp16.tensor', weights_only=True)
    grad = torch.load('grad_fp16.tensor', weights_only=True)
else:
    query = torch.load('query.tensor', weights_only=True)
    key = torch.load('key.tensor', weights_only=True)
    value = torch.load('value.tensor', weights_only=True)
    grad = torch.load('grad.tensor', weights_only=True)

if CAST_FP32:
    query = dup(query, dtype=torch.float32)
    key = dup(key, dtype=torch.float32)
    value = dup(value, dtype=torch.float32)
    grad = dup(grad, dtype=torch.float32)

print(f'{query.shape=} {query.dtype=} {query.device=}')
print(f'{key.shape=} {key.dtype=}')
print(f'{value.shape=} {value.dtype=}')
print(f'{grad.shape=} {grad.dtype=}')

print('query.max(): ', query.max())
print('key.max(): ', key.max())
print('value.max(): ', value.max())
print('output.grad.max(): ', grad.max())
is_causal = False
dropout_p = 0.0
sm_scale = 1.0 / math.sqrt(query.size(-1))
ext = AttentionExtraArgs(return_encoded_softmax=dropout_p > 0.0,
                         autotune=False,
                         return_autotune=False,
                         fillnan=True)
if USE_FULL_TENSOR:
    if USE_MATH:
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p, is_causal=is_causal)
    else:
        output, encoded_softmax, _ = attention(query, key, value, None, is_causal, sm_scale, dropout_p, ext)
    # output = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p, is_causal=False)
    output.backward(grad)
    print('output.max(): ',output.max())
    print('output.argmax(): ', argmax(output))
    print(f'{output[0,6,3059]=}')
    print('query.grad.max(): ',query.grad.max())
    print('key.grad.max(): ',key.grad.max())
    print('value.grad.max(): ',value.grad.max())
    print('value.grad.argmax(): ', argmax(value.grad))
    print(f'{value.grad[0,6,3059]=}')
    wherenan = torch.nonzero(value.grad.isnan())
    print(wherenan)
    exit()

BAD_HEAD = 6
BAD_SEQ = 3059
# SEQ_BLOCK = 1024
# for SEQ_BLOCK in [16, 32, 64, 128, 256, 512, 1024]:
# Note: nan produced when SEQ_BLOCK = 128
#       inf produced when SEQ_BLOCK = 64
#       32 is the last value that produces normal results
SEQ_BLOCK = 64
# for OFFSET in range(0, BAD_SEQ % SEQ_BLOCK):
# Note: OFFSET = 0 can produce inf
OFFSET = 0
# for DHEAD_HI in [16, 32, 64]:
# inf produced with DHEAD_HI = 64
DHEAD_HI = 64
# for DHEAD_LO in [0, 32, 48]:
# inf produced with DHEAD_LO = 0
DHEAD_LO = 0
if not USE_FULL_TENSOR:  # Placeholder of "for"
# for SEQ_BLOCK in [16, 32, 64]:  # Bugged if SEQ_BLOCK != 32, weird
    # print(f'========================={SEQ_BLOCK=}==========================')
    # print(f'========================={OFFSET=}==========================')
    # print(f'========================={DHEAD_HI=}==========================')
    print(f'========================={DHEAD_LO=}==========================')
    BAD_SEQ_LO = BAD_SEQ // SEQ_BLOCK * SEQ_BLOCK
    BAD_SEQ_HI = BAD_SEQ_LO + SEQ_BLOCK

    BAD_SEQ_LO += OFFSET
    BAD_SEQ_HI += OFFSET

    BAD_SEQ_SUBIDX = BAD_SEQ - BAD_SEQ_LO
    print(f"{BAD_SEQ_LO=}:{BAD_SEQ_HI=} {BAD_SEQ=} -> {BAD_SEQ_SUBIDX=}")
    # BAD_SEQ_LO = None
    # BAD_SEQ_HI = None

    iso_query = dup(query[:, BAD_HEAD:BAD_HEAD+1, BAD_SEQ_LO:BAD_SEQ_HI, DHEAD_LO:DHEAD_HI])
    iso_key = dup(key[:, BAD_HEAD:BAD_HEAD+1, BAD_SEQ_LO:BAD_SEQ_HI, DHEAD_LO:DHEAD_HI])
    iso_value = dup(value[:, BAD_HEAD:BAD_HEAD+1, BAD_SEQ_LO:BAD_SEQ_HI, DHEAD_LO:DHEAD_HI])
    iso_grad = dup(grad[:, BAD_HEAD:BAD_HEAD+1, BAD_SEQ_LO:BAD_SEQ_HI, DHEAD_LO:DHEAD_HI])
    print(f'{iso_query.shape=}')
    print(f'{iso_key.shape=}')
    print(f'{iso_value.shape=}')
    print(f'{iso_grad.shape=}')

    if USE_MATH:
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            iso_output = F.scaled_dot_product_attention(iso_query, iso_key, iso_value, dropout_p=dropout_p, is_causal=is_causal)
    else:
        iso_output, iso_encoded_softmax, _ = attention(iso_query, iso_key, iso_value, None, is_causal, sm_scale, dropout_p, ext)
    iso_output.backward(iso_grad)
    print(f'{iso_output.max()=}')
    print(f'{iso_output[0,0,BAD_SEQ_SUBIDX]=}')
    print(f'{iso_query.grad.max()=}')
    print(f'{iso_key.grad.max()=}')
    print(f'{iso_value.grad.max()=}')
    print(f'{iso_value.grad[0, 0, BAD_SEQ_SUBIDX]=}')
    print()
