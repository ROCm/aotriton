  [[backend_context_name]] bcontext[[nth_kernel]](context, [[condition]]);
  err = bcontext[[nth_kernel]].lookup_optimal(gpu);
  if (err != hipSuccess)
    return err;
