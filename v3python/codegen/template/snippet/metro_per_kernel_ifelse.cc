  // TODO: Replace with std::variant
  [[backend_context_name]] bcontext[[nth_kernel]]_if;
  [[else_context_name]] bcontext[[nth_kernel]]_else;
  // It is possible to launch neither according to SkipAndQueryKernelNumber
  bool launch_condition[[nth_kernel]]_if = condition[[nth_kernel]];
  bool launch_condition[[nth_kernel]]_else = !condition[[nth_kernel]];
  if (condition[[nth_kernel]]) {
    bcontext[[nth_kernel]]_if.params = context.params;
    err = context_lookup_helper(options,
                                launch_condition[[nth_kernel]]_if,
                                bcontext[[nth_kernel]]_if,
                                [[kernel_slot_index_if]]);
    if (err != hipSuccess)
      return err;
  } else {
    bcontext[[nth_kernel]]_else.params = context.params;
    err = context_lookup_helper(options,
                                launch_condition[[nth_kernel]]_else,
                                bcontext[[nth_kernel]]_else,
                                [[kernel_slot_index_else]]);
    if (err != hipSuccess)
      return err;
  }
