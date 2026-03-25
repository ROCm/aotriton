template<typename ContextClass>
hipError_t context_lookup_helper(const attn_options* options,
                                 bool& launch_condition,
                                 ContextClass& bcontext,
                                 int kernel_slot_index) {
#if AOTRITON_BUILD_FOR_TUNING
  using KCV = attn_options::KernelControlValue;
  int control_value = KCV::Auto;
  if (options) {
    control_value = options->kernel_fine_control[kernel_slot_index];
    // Handle SkipAndQueryKernelNumber: skip execution but record we need to query
    if (control_value == KCV::SkipAndQueryKernelNumber) {
      launch_condition = false;
    } else if (control_value >= 0) {
      bcontext.force_kernel_index = control_value;
    }
  }
#endif
  bcontext.params = context.params;
  err = bcontext.lookup_optimal(gpu);
  if (err != hipSuccess)
    return err;
#if AOTRITON_BUILD_FOR_TUNING
  // No need to check options, control_value == Skip IFF options presents
  if (control_value == KCV::SkipAndQueryKernelNumber) {
    options->kernel_fine_control[kernel_slot_index] = bcontext._total_number_of_kernels;
  }
#endif
}

hipError_t [[launcher_func_name]](const [[context_class_name]]& context,
                                  Gpu gpu,
                                  hipStream_t stream) {
    hipError_t err;
    const attn_options* options = context.call_options;
#if AOTRITON_BUILD_FOR_TUNING
    using KCV = attn_options::KernelControlValue;
#endif
[[lookup_every_kernel]]
[[launch_every_kernel]]
}
