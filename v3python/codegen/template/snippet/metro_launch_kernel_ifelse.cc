  err = bcontext[[nth_kernel]].launch(stream);
  if (err != hipSuccess)
    return err;
  err = bcontext[[nth_kernel]]_else.launch(stream);
  if (err != hipSuccess)
    return err;
