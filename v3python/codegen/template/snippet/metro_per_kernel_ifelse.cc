  // TODO: Replace with std::variant
  [[backend_context_name]] bcontext[[nth_kernel]]_if;
  [[else_context_name]] bcontext[[nth_kernel]]_else;
  bool condition[[nth_kernel]] = ([[condition]]);
#if AOTRITON_BUILD_FOR_TUNING
  if (options) {
    int force_kernel_index[[nth_kernel]] = options->force_kernel_indices[[[kernel_slot_index]]];
    bool skip[[nth_kernel]] = (force_kernel_index[[nth_kernel]] == KIV::Skip);
    condition[[nth_kernel]] &&= !skip[[nth_kernel]];
    if (condition[[nth_kernel]] && force_kernel_index[[nth_kernel]] >= 0) {
      // Note: Both if and else branches share the same force_kernel_index
      bcontext[[nth_kernel]]_if.force_kernel_index = force_kernel_index[[nth_kernel]];
      bcontext[[nth_kernel]]_else.force_kernel_index = force_kernel_index[[nth_kernel]];
    }
  }
#endif
  if (condition[[nth_kernel]]) {
    bcontext[[nth_kernel]]_if.params = context.params;
    err = bcontext[[nth_kernel]]_if.lookup_optimal(gpu);
    if (err != hipSuccess)
      return err;
  } else {
    bcontext[[nth_kernel]]_else.params = context.params;
    err = bcontext[[nth_kernel]]_else.lookup_optimal(gpu);
    if (err != hipSuccess)
      return err;
  }
