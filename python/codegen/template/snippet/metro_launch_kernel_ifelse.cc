  // Conditional skip is inside launch
  // TODO: a new pattern to ensure Conditional skip (must not use virtual function)
  err = bcontext[[nth_kernel]].launch(stream);
  if (err != hipSuccess)
    return err;
  err = bcontext[[nth_kernel]]_else.launch(stream);
  if (err != hipSuccess)
    return err;
