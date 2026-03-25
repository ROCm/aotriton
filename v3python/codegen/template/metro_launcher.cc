hipError_t [[launcher_func_name]](const [[context_class_name]]& context,
                                  Gpu gpu,
                                  hipStream_t stream) {
    hipError_t err;
    const attn_options* options = context.call_options;
#if AOTRITON_BUILD_FOR_TUNING
    using KIV = attn_options::KernelIndexValue;
#endif
[[lookup_every_kernel]]
[[launch_every_kernel]]
}
