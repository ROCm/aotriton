  // TODO: Replace with std::variant
  bool condition[[nth_kernel]] = ([[condition]]);
  [[backend_context_name]] bcontext[[nth_kernel]](context, condition[[nth_kernel]]);
  err = bcontext[[nth_kernel]].lookup_optimal(gpu);
  if (err != hipSuccess)
    return err;
  [[else_context_name]] bcontext[[nth_kernel]]_else(context, !condition[[nth_kernel]]);
  err = bcontext[[nth_kernel]]_else.lookup_optimal(gpu);
  if (err != hipSuccess)
    return err;
