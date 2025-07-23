  // TODO: Replace with std::variant
  [[backend_context_name]] bcontext[[nth_kernel]]_if;
  [[else_context_name]] bcontext[[nth_kernel]]_else;
  bool condition[[nth_kernel]] = ([[condition]]);
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
