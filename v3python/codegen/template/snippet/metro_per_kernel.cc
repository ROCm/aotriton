  [[backend_context_name]] bcontext[[nth_kernel]];
  bool condition[[nth_kernel]] = ([[condition]]);
  if (condition[[nth_kernel]]) {
    // For branchless kernel, just modify condition[[nth_kernel]] directly.
    err = context_lookup_helper(context,
                                options,
                                condition[[nth_kernel]],
                                bcontext[[nth_kernel]],
                                [[kernel_slot_index]],
                                gpu);
    if (err != hipSuccess)
        return err;
  }
