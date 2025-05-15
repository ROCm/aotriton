hipError_t [[launcher_func_name]](const [[context_class_name]]& context,
                                  Gpu gpu,
                                  hipStream_t stream) {
    [[backend_context_name]] bcontext;
    bcontext.params = context.params;
    hipError_t err;
    err = bcontext.lookup_optimal(gpu);
    if (err != hipSuccess)
        return err;
    err = bcontext.launch(stream);
    return err;
}
