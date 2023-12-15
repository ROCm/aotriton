namespace aotriton::v2::flash {

hipError_t attn_fwd_launcher(TritonKernel kernel,
                             {{LAUNCHER_PARAMETERS}},
                             hipStream_t stream)
{
    uint64_t seqlen_q = Q.size(2);
    uint64_t seqlen_k = K.size(2);
    {{LET_TENSOR_STRIDE_ARGUMENTS}};
    {{LET_KERNEL_ARGUMENTS}};
    dim3 grid { cdiv(seqlen_q, kernel.get_BLOCK_N()), Q.size(0) * Q.size(1), 1 };
    return kernel.invoke(grid, args, stream);
}

}
