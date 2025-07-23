  [[backend_context_name]] bcontext[[nth_kernel]];
  bool condition[[nth_kernel]] = ([[condition]]);
  if (condition[[nth_kernel]]) {
    bcontext[[nth_kernel]].params = context.params;
    err = bcontext[[nth_kernel]].lookup_optimal(gpu);
    if (err != hipSuccess)
        return err;
  }
