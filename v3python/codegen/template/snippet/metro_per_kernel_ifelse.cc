  // TODO: Replace with std::variant
  [[backend_context_name]] bcontext[[nth_kernel]]_if;
  [[else_context_name]] bcontext[[nth_kernel]]_else;
  bool condition[[nth_kernel]] = ([[condition]]);
#if AOTRITON_BUILD_FOR_TUNING
  if (options) {
    int force_kernel_index[[nth_kernel]]_if = options->force_kernel_indices[[[kernel_slot_index]]];
    int force_kernel_index[[nth_kernel]]_else = options->force_kernel_indices[[[else_kernel_slot_index]]];
    bool skip[[nth_kernel]]_if = (force_kernel_index[[nth_kernel]]_if == KIV::Skip);
    bool skip[[nth_kernel]]_else = (force_kernel_index[[nth_kernel]]_else == KIV::Skip);
    // Apply skip logic: if both branches are skipped, skip the whole conditional
    if (skip[[nth_kernel]]_if && skip[[nth_kernel]]_else) {
      condition[[nth_kernel]] = false;
    } else if (skip[[nth_kernel]]_if) {
      condition[[nth_kernel]] = false;  // Force else branch
    } else if (skip[[nth_kernel]]_else) {
      condition[[nth_kernel]] = true;   // Force if branch
    }
    // Apply force_kernel_index to respective branches
    if (condition[[nth_kernel]] && force_kernel_index[[nth_kernel]]_if >= 0) {
      bcontext[[nth_kernel]]_if.force_kernel_index = force_kernel_index[[nth_kernel]]_if;
    }
    if (!condition[[nth_kernel]] && force_kernel_index[[nth_kernel]]_else >= 0) {
      bcontext[[nth_kernel]]_else.force_kernel_index = force_kernel_index[[nth_kernel]]_else;
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
