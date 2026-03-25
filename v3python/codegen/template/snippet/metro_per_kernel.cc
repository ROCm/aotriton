  [[backend_context_name]] bcontext[[nth_kernel]];
  bool condition[[nth_kernel]] = ([[condition]]);
  {
    err = context_lookup_helper(options,
                                condition[[nth_kernel]],
                                bcontext[[nth_kernel]],
                                [[kernel_slot_index]]);
    if (err != hipSuccess)
        return err;
  }
