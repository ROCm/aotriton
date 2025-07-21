hipError_t [[launcher_func_name]](const [[context_class_name]]& context,
                                  Gpu gpu,
                                  hipStream_t stream) {
    hipError_t err;
[[lookup_every_kernel]]
[[launch_every_kernel]]
}
