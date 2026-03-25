  // TODO: Replace with std::variant
  [[backend_context_name]] bcontext[[nth_kernel]];
  [[else_context_name]] bcontext[[nth_kernel]]_else;
  // It is possible to launch neither according to SkipAndQueryKernelNumber
  // hence we need launch_condition* variables.
  bool condition[[nth_kernel]] = ([[condition]]);
  bool launch_condition[[nth_kernel]] = condition[[nth_kernel]];
  bool launch_condition[[nth_kernel]]_else = !condition[[nth_kernel]];
  if (condition[[nth_kernel]]) {
    err = context_lookup_helper(context,
                                options,
                                launch_condition[[nth_kernel]],
                                bcontext[[nth_kernel]],
                                [[kernel_slot_index]],
                                gpu);
    if (err != hipSuccess)
      return err;
  } else {
    err = context_lookup_helper(context,
                                options,
                                launch_condition[[nth_kernel]]_else,
                                bcontext[[nth_kernel]]_else,
                                [[kernel_slot_index_else]],
                                gpu);
    if (err != hipSuccess)
      return err;
  }
