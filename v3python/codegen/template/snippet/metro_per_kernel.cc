  [[backend_context_name]] bcontext[[nth_kernel]];
  bool condition[[nth_kernel]] = ([[condition]]);
#if AOTRITON_BUILD_FOR_TUNING
  if (options) {
    int force_kernel_index[[nth_kernel]] = options->force_kernel_indices[[[kernel_slot_index]]];
    bool skip[[nth_kernel]] = (force_kernel_index[[nth_kernel]] == KIV::Skip);
    condition[[nth_kernel]] &&= !skip[[nth_kernel]];
    if (condition[[nth_kernel]] && force_kernel_index[[nth_kernel]] >= 0) {
      bcontext[[nth_kernel]].force_kernel_index = force_kernel_index[[nth_kernel]];
    }
  }
#endif
  if (condition[[nth_kernel]]) {
    bcontext[[nth_kernel]].params = context.params;
    err = bcontext[[nth_kernel]].lookup_optimal(gpu);
    if (err != hipSuccess)
        return err;
  }
